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
#ifndef DIRECT_MAPPED_LRU_CACHE_FIND_UNCACHED_SIMT_KERNEL_H
#define DIRECT_MAPPED_LRU_CACHE_FIND_UNCACHED_SIMT_KERNEL_H

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

namespace DirectMappedLruCacheFindUncachedSimt {

// 线程块与 Warp 配置
constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t WARP_SIZE = 32;

// UVM 统计索引：NUM_CALLS = 调用次数，NUM_REQUESTED_INDICES = 请求的索引数
constexpr int32_t UVM_NUM_CALLS = 0;
constexpr int32_t UVM_NUM_REQUESTED_INDICES = 1;

// MurmurHash3 64-bit finalizer，将线性索引映射到 cache_set
__simt_callee__ inline uint32_t CacheSlot(int64_t hIn, int32_t c)
{
    uint64_t h = static_cast<uint64_t>(hIn);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return static_cast<uint32_t>(h % static_cast<uint32_t>(c));
}

// SIMT 向量计算入口：每个 warp 处理一个索引，通过 atomicMax 竞争 direct-mapped 插入权
template <typename IndexT>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void FindUncachedSmallDataCompute(
    __gm__ IndexT* linear_cache_indices, int32_t N, int64_t max_indices, __gm__ int64_t* lxu_cache_state, int32_t C,
    __gm__ int32_t* cache_sets, int64_t time_stamp, __gm__ int64_t* lru_state, bool gather_cache_stats,
    __gm__ int32_t* uvm_cache_stats, int64_t uvm_len, __gm__ int64_t* lxu_cache_miss_timestamp)
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

    // UVM 调用统计：第一个 block 的线程 0 记录调用次数和索引请求数
    if (gather_cache_stats) {
        if (bid == 0 && threadIdx.x == 0) {
            atomicAdd(&uvm_cache_stats[UVM_NUM_CALLS], 1);
            atomicAdd(&uvm_cache_stats[UVM_NUM_REQUESTED_INDICES], N);
        }
    }

    // 主循环：按 warp 粒度遍历所有索引
    for (int32_t n = bid * numWarpsPerBlock + warpId; n < N; n += grid * numWarpsPerBlock) {
        const int64_t idx = static_cast<int64_t>(linear_cache_indices[n]);

        // 检查是否为无效/被删除的索引（等于 max_indices 哨兵值）
        if (idx == max_indices) {
            if (laneId == 0) {
                cache_sets[n] = -1;
            }
            continue;
        }

        // MurmurHash3 计算该索引对应的 cache_set（所有 lane 结果相同）
        const int32_t cache_set = static_cast<int32_t>(CacheSlot(idx, C));

        // 命中判断：所有 lane 读同一个 cache line 的状态
        const bool found = (lxu_cache_state[cache_set] == idx);

        // 命中时更新 LRU 时间戳
        if (found) {
            lru_state[cache_set] = time_stamp;
        }

        // 后续操作仅 lane 0 执行，避免重复写入
        if (laneId != 0) {
            continue;
        }

        if (found) {
            // 命中：标记为不插入
            cache_sets[n] = -1;
        } else {
            // 未命中：通过 atomicMax 竞争插入权，胜出者获得 cache_set
            auto old = atomicMax(&lxu_cache_miss_timestamp[cache_set], time_stamp + 1);
            if (old < time_stamp + 1) {
                cache_sets[n] = cache_set;
            } else {
                cache_sets[n] = -1;
            }
        }
    }
}

}  // namespace DirectMappedLruCacheFindUncachedSimt

#endif  // DIRECT_MAPPED_LRU_CACHE_FIND_UNCACHED_SIMT_KERNEL_H
