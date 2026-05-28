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
#ifndef DIRECT_MAPPED_LRU_CACHE_INSERT_BYTE_SIMT_KERNEL_H
#define DIRECT_MAPPED_LRU_CACHE_INSERT_BYTE_SIMT_KERNEL_H

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

#include "cache_constants.h"
#include "padded_row.h"

using namespace AscendC;

// 线程块与 Warp 配置
constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t WARP_SIZE = 32;

namespace DirectMappedLruCacheInsertByte {

// SIMT 向量计算入口：每个 warp 将一个未命中的索引对应的权重数据从 UVM 写入 Cache
template <typename IndexT>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void InsertByteSmallDataCompute(
    __gm__ uint8_t* weights, __gm__ int64_t* cache_hash_size_cumsum, __gm__ int32_t* cache_index_table_map,
    __gm__ int64_t* weights_offsets, __gm__ uint8_t* weights_tys, __gm__ int32_t* d_offsets,
    __gm__ int64_t* lxu_cache_state, __gm__ uint8_t* lxu_cache_weights, int64_t time_stamp, __gm__ int64_t* lru_state,
    __gm__ IndexT* linear_cache_indices, __gm__ int64_t* lxu_cache_miss_timestamp, __gm__ int32_t* cache_sets,
    bool gather_cache_stats, __gm__ int32_t* uvm_cache_stats, int32_t num_cache_sets, int32_t lxu_cache_row_bytes,
    int32_t row_alignment, int32_t N)
{
    // 线程索引计算
    const int32_t warpId = threadIdx.x / WARP_SIZE;
    const int32_t laneId = threadIdx.x % WARP_SIZE;
    const int32_t numWarpsPerBlock = MAX_THREADS_PER_BLOCK / WARP_SIZE;
    const int32_t bid = blockIdx.x;
    int32_t grid = gridDim.x;
    if (grid <= 0) {
        grid = 1;
    }

    // 主循环：按 warp 粒度遍历所有索引
    for (int32_t pos = bid * numWarpsPerBlock + warpId; pos < N; pos += grid * numWarpsPerBlock) {
        const int32_t cache_set = cache_sets[pos];

        // 跳过未分配的 cache set
        if (cache_set == -1) {
            continue;
        }

        // 已被当前时间戳标记为命中的跳过
        if (lru_state[cache_set] == time_stamp) {
            continue;
        }

        // 统计 direct-mapped 冲突未命中次数
        if (gather_cache_stats && laneId == 0) {
            atomicAdd(&uvm_cache_stats[fbgemm_compat::UvmCacheStatsIndex::num_conflict_unique_misses], 1);
        }

        // 通过索引映射表查找表 ID、权重类型、偏移量和维度
        const int64_t insert_idx = static_cast<int64_t>(linear_cache_indices[pos]);
        const int32_t t_insert = cache_index_table_map[insert_idx];
        const uint8_t w_ty_insert = weights_tys[t_insert];
        const int64_t idx_insert = insert_idx - cache_hash_size_cumsum[t_insert];
        const int64_t weights_offset_insert = weights_offsets[t_insert];
        const int32_t D_start_insert = d_offsets[t_insert];
        const int32_t D_end_insert = d_offsets[t_insert + 1];
        const int32_t D_insert = D_end_insert - D_start_insert;

        // 计算行字节数（对齐后）
        const int32_t D_insert_bytes = fbgemm_compat::PaddedRowSizeBytes(D_insert, w_ty_insert, row_alignment);

        // 源地址（UVM）和目标地址（Cache）
        __gm__ uint8_t* uvm_row = weights + weights_offset_insert + idx_insert * static_cast<int64_t>(D_insert_bytes);
        __gm__ uint8_t* cache_row =
            lxu_cache_weights + static_cast<int64_t>(cache_set) * static_cast<int64_t>(lxu_cache_row_bytes);

        // Warp 内各线程并行拷贝权重行数据
        for (int32_t d = laneId; d < D_insert_bytes; d += WARP_SIZE) {
            cache_row[d] = uvm_row[d];
        }

        // 仅 lane 0 更新 cache 状态和 LRU 时间戳
        if (laneId == 0) {
            lxu_cache_state[cache_set] = insert_idx;
            lru_state[cache_set] = time_stamp;
        }
    }
}

}  // namespace DirectMappedLruCacheInsertByte

#endif  // DIRECT_MAPPED_LRU_CACHE_INSERT_BYTE_SIMT_KERNEL_H
