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

#ifndef LRU_CACHE_FIND_UNCACHED_SIMT_KERNEL_H
#define LRU_CACHE_FIND_UNCACHED_SIMT_KERNEL_H

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

namespace LruCacheFindUncachedSimt {

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t WARP_SIZE = 32;

constexpr int32_t UVM_NUM_CALLS = 0;
constexpr int32_t UVM_NUM_REQUESTED_INDICES = 1;
constexpr int32_t UVM_NUM_UNIQUE_INDICES = 2;
constexpr int32_t UVM_NUM_UNIQUE_MISSES = 3;

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

/// Warp 内是否存在任一线程 v 为真（等价 __any_sync 语义，mask 全参与）。
__simt_callee__ inline bool WarpAny(bool v)
{
    int32_t b = v ? 1 : 0;
#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1) {
        b |= asc_shfl_xor(b, off);
    }
    return b != 0;
}

template <typename IndexT>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK)
inline void FindUncachedSmallDataCompute(
    __gm__ IndexT* unique_indices,
    int64_t unique_indices_numel,
    __gm__ int32_t* N_unique,
    int64_t max_indices,
    int32_t nbuf,
    __gm__ int64_t* lxu_cache_state,
    __gm__ int32_t* cache_sets,
    int64_t time_stamp,
    __gm__ int64_t* lru_state,
    bool gather_cache_stats,
    __gm__ int32_t* uvm_cache_stats,
    int64_t uvm_len,
    bool lock_cache_line,
    __gm__ int32_t* lxu_cache_locking_counter,
    int64_t lock_len,
    int32_t C,
    int32_t W)
{
    const int32_t warpId = threadIdx.x / WARP_SIZE;
    const int32_t laneId = threadIdx.x % WARP_SIZE;
    const int32_t numWarpsPerBlock = MAX_THREADS_PER_BLOCK / WARP_SIZE;
    const int32_t bid = blockIdx.x;
    int32_t grid = gridDim.x;
    if (grid <= 0) {
        grid = 1;
    }

    int32_t kEff = *N_unique;
    if (kEff < 0) {
        kEff = 0;
    }
    if (kEff > nbuf) {
        kEff = nbuf;
    }

    const int64_t cw = static_cast<int64_t>(C) * static_cast<int64_t>(W);

    if (gather_cache_stats) {
        // 与 CUDA：仅 blockIdx==0 且单线程写 num_calls / requested / unique_indices
        if (bid == 0 && threadIdx.x == 0) {
            if (uvm_len > static_cast<int64_t>(UVM_NUM_UNIQUE_MISSES)) {
                atomicAdd(&uvm_cache_stats[UVM_NUM_CALLS], 1);
                atomicAdd(
                    &uvm_cache_stats[UVM_NUM_REQUESTED_INDICES],
                    static_cast<int32_t>(unique_indices_numel));
                atomicAdd(&uvm_cache_stats[UVM_NUM_UNIQUE_INDICES], kEff);
            }
        }
    }

    int32_t n_misses = 0;

    // 本核 warp 起始偏移 bid * numWarpsPerBlock + warpId，跨核步进 grid * numWarpsPerBlock
    for (int32_t n = bid * numWarpsPerBlock + warpId; n < kEff; n += grid * numWarpsPerBlock) {
        const int64_t idx = static_cast<int64_t>(unique_indices[n]);
        if (idx == max_indices) {
            continue;
        }

        const int32_t cache_set = static_cast<int32_t>(CacheSlot(idx, C));

        bool found = false;
        if (laneId < W) {
            const int64_t pos = static_cast<int64_t>(cache_set) * static_cast<int64_t>(W) + laneId;
            found = (lxu_cache_state[pos] == idx);
        }

        if (found) {
            const int64_t pos = static_cast<int64_t>(cache_set) * static_cast<int64_t>(W) + laneId;
            const bool already_locked = (lru_state[pos] == time_stamp);
            lru_state[pos] = time_stamp;
            if (lock_cache_line && lock_len == cw && !already_locked) {
                lxu_cache_locking_counter[pos] += 1;
            }
        }

        if (!WarpAny(found)) {
            if (laneId == 0) {
                cache_sets[n] = cache_set;
                ++n_misses;
            }
        }
    }

    if (gather_cache_stats && laneId == 0) {
        if (uvm_len > static_cast<int64_t>(UVM_NUM_UNIQUE_MISSES)) {
            atomicAdd(&uvm_cache_stats[UVM_NUM_UNIQUE_MISSES], n_misses);
        }
    }
}

}  // namespace LruCacheFindUncachedSimt

#endif  // LRU_CACHE_FIND_UNCACHED_SIMT_KERNEL_H
