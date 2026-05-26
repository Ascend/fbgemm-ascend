/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#ifndef SIMT_KERNEL_H
#define SIMT_KERNEL_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

namespace emb_inplace_update_kernel {

// SparseType encoding (must match fbgemm SparseType enum)
constexpr uint8_t SPARSE_TY_FP32 = 0;
constexpr uint8_t SPARSE_TY_FP16 = 1;
constexpr uint8_t SPARSE_TY_INT8 = 2;
constexpr uint8_t SPARSE_TY_INT4 = 3;
constexpr uint8_t SPARSE_TY_INT2 = 4;
constexpr uint8_t SPARSE_TY_BF16 = 5;
constexpr uint8_t SPARSE_TY_FP8 = 6;

// 昇腾 SIMT 暂不支持 UVM，placement 仅保留 DEVICE 取值用于 host 侧校验。
constexpr int32_t PLACEMENT_DEVICE = 0;

// 计算单行 padding 后的字节数。
//   D：embedding 维度（元素个数，非字节）
//   weight_ty : 元素量化类型（SparseType 枚举）
//   row_alignment: 行对齐字节数（<=1 表示不对齐）
//   r：对齐前的单行字节数
// 量化类型 INT8/INT4/INT2 每行额外携带 4B 元数据（fbgemm 约定的 scale+zero）。
__simt_callee__ __aicore__ inline int32_t PaddedRowSizeInBytes(int32_t D, uint8_t weight_ty, int32_t row_alignment)
{
    int32_t r = 0;
    switch (weight_ty) {
        case SPARSE_TY_FP32:
            r = D * 4;
            break;
        case SPARSE_TY_FP16:
            r = D * 2;
            break;
        case SPARSE_TY_INT8:
            r = D + 4;
            break;
        case SPARSE_TY_INT4:
            r = D / 2 + 4;
            break;
        case SPARSE_TY_INT2:
            r = D / 4 + 4;
            break;
        case SPARSE_TY_BF16:
            r = D * 2;
            break;
        case SPARSE_TY_FP8:
            r = D;
            break;
        default:
            r = 0;
            break;
    }
    if (row_alignment <= 1) {
        return r;
    }
    return ((r + row_alignment - 1) / row_alignment) * row_alignment;
}

// SIMT warp-cooperative kernel for emb_inplace_update.
//
// 模型与 fbgemm GPU 实现对齐：每个 warp（32 lane）协作处理一条更新记录，
// warp 内 32 个 lane 以 stride=32 步长协作搬运一行 `Db` 字节。
//
// 拷贝粒度：每 lane 16 字节（CANN 内置 float4 向量），与 GPU uint4 完全对齐。
// 单 warp 单轮搬运量 = 32 × 16 = 512 字节，最大化 HBM 带宽利用率。
//
// 线程身份：
//   threadIdx = blockIdx.x * threadNum + tid
//   warpId    = tid / WARP_SIZE
//   laneId    = tid % WARP_SIZE
//   recordIdx = blockIdx.x * (threadNum / WARP_SIZE) + warpId   ← 一条记录
//
// 拷贝策略（lane 间分工）：
//   16B 主循环：lane d 处理 dst[16d..16d+15] = src[16d..16d+15]，d += 32
//   tail 处理： lane 0 单独处理 0..15 字节的尾巴（8/4/2/1 降级）
//
// 昇腾 SIMT 暂不支持 UVM 特性：仅处理 placement == DEVICE(0) 的记录，
// 非 DEVICE 的记录由整个 warp 一致跳过（不报错、不写入）。
constexpr int32_t WARP_SIZE = 32;

template <typename RowIdType>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void SimtEmbInplaceUpdateMultiThread(
    __gm__ uint8_t* dev_weights, __gm__ uint8_t* /*uvm_weights*/, __gm__ int32_t* weights_placements,
    __gm__ int64_t* weights_offsets, __gm__ uint8_t* weights_tys, __gm__ int32_t* D_offsets,
    __gm__ uint8_t* update_weights, __gm__ int32_t* update_table_indices, __gm__ RowIdType* update_row_indices,
    __gm__ int64_t* update_offsets, int64_t N, int32_t row_alignment)
{
    int32_t tid = static_cast<int32_t>(AscendC::Simt::GetThreadIdx<0>());
    int32_t blockIdx = static_cast<int32_t>(AscendC::Simt::GetBlockIdx());
    int32_t threadNum = static_cast<int32_t>(AscendC::Simt::GetThreadNum<0>());
    int32_t blockNum = AscendC::Simt::GetBlockNum();

    int32_t warpsPerBlock = threadNum / WARP_SIZE;
    int32_t warpId = tid / WARP_SIZE;
    int32_t laneId = tid % WARP_SIZE;
    int64_t totalWarps = static_cast<int64_t>(blockNum) * warpsPerBlock;

    // 每 warp 处理 stride=totalWarps 的若干条记录
    int64_t recordStart = static_cast<int64_t>(blockIdx) * warpsPerBlock + warpId;
    int64_t kRecPerWarp = (N + totalWarps - 1) / totalWarps;

    for (int64_t i = 0; i < kRecPerWarp; ++i) {
        int64_t idx = recordStart + i * totalWarps;
        if (idx >= N) {
            return;
        }

        int32_t t = update_table_indices[idx];

        // 非 DEVICE 表：整个 warp 一致跳过（所有 lane 读到相同的 placement[t]）
        if (weights_placements[t] != PLACEMENT_DEVICE) {
            continue;
        }

        int64_t r = static_cast<int64_t>(update_row_indices[idx]);
        int32_t D = D_offsets[t + 1] - D_offsets[t];
        uint8_t ty = weights_tys[t];
        int32_t Db = PaddedRowSizeInBytes(D, ty, row_alignment);
        if (Db <= 0) {
            continue;
        }

        int64_t srcOff = update_offsets[idx];
        int64_t dstOff = weights_offsets[t] + static_cast<int64_t>(Db) * r;

        __gm__ uint8_t* src = update_weights + srcOff;
        __gm__ uint8_t* dst = dev_weights + dstOff;

        // ---- warp 协作 16B 向量化拷贝（与 fbgemm GPU uint4 完全对齐）----
        // host 侧已强制 row_alignment 为 16 的倍数，因此 Db 必为 16 的倍数，
        // src/dst 地址天然 16B 对齐，无需运行时检查，无需 tail 处理。
        auto vec_dst = reinterpret_cast<__gm__ float4*>(dst);
        auto vec_src = reinterpret_cast<__gm__ float4*>(src);
        int32_t numChunks16 = Db / 16;
        for (int32_t d = laneId; d < numChunks16; d += WARP_SIZE) {
            vec_dst[d] = vec_src[d];
        }
    }
}

}  // namespace emb_inplace_update_kernel

#endif  // SIMT_KERNEL_H
