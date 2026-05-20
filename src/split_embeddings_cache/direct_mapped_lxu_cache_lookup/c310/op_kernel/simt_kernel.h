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

#ifndef DIRECT_MAPPED_LXU_CACHE_LOOKUP_SIMT_KERNEL_H
#define DIRECT_MAPPED_LXU_CACHE_LOOKUP_SIMT_KERNEL_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t WARP_SIZE = 32;

constexpr int32_t UVM_NUM_MISSES = 5;

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

namespace DirectMappedLxuCacheLookupSimt {

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtDirectMappedLxuCacheLookupMultiThread(
    __gm__ T* linear_cache_indices, __gm__ int64_t* lxu_cache_state, int64_t invalid_index, bool gather_cache_status,
    __gm__ int32_t* uvm_cache_stats, __gm__ T* lxu_cache_locations, int32_t indices, int32_t slots, int32_t uvm_len)
{
    int32_t threadIdx = static_cast<int32_t>(Simt::GetThreadIdx<0>());
    int32_t blockIdx = static_cast<int32_t>(Simt::GetBlockIdx());
    int32_t threadNum = static_cast<int32_t>(Simt::GetThreadNum<0>());
    int32_t blockNum = static_cast<int32_t>(Simt::GetBlockNum());
    int32_t totalThreadNum = blockNum * threadNum;

    int64_t perCoreIndicesNum = indices / blockNum;
    int64_t remainder = indices % blockNum;

    int64_t coreStartIdx = (blockIdx < remainder)
                               ? (blockIdx * (perCoreIndicesNum + 1))
                               : (remainder * (perCoreIndicesNum + 1) + (blockIdx - remainder) * perCoreIndicesNum);
    int64_t coreLen = (blockIdx < remainder) ? (perCoreIndicesNum + 1) : perCoreIndicesNum;

    if (coreLen <= 0) {
        return;
    }

    int64_t elementsPerThread = coreLen / threadNum;
    int64_t remainderPerThread = coreLen % threadNum;

    int64_t start, end;
    if (threadIdx < remainderPerThread) {
        start = coreStartIdx + threadIdx * (elementsPerThread + 1);
        end = start + elementsPerThread + 1;
    } else {
        start = coreStartIdx + threadIdx * elementsPerThread + remainderPerThread;
        end = start + elementsPerThread;
    }

    if (start >= indices) {
        return;
    }
    if (end > indices) {
        end = indices;
    }

    int32_t num_misses = 0;

    for (int64_t n = start; n < end; ++n) {
        int64_t idx = linear_cache_indices[n];

        if (idx == invalid_index) {
            continue;
        }

        int32_t cache_set = CacheSlot(idx, slots);

        if (lxu_cache_state[cache_set] == idx) {
            lxu_cache_locations[n] = cache_set;
        } else {
            lxu_cache_locations[n] = -1;
            num_misses++;
        }
    }

    if (gather_cache_status && (threadIdx == 0)) {
        if (uvm_len > static_cast<int64_t>(UVM_NUM_MISSES)) {
            atomicAdd(&uvm_cache_stats[UVM_NUM_MISSES], num_misses);
        }
    }
}

}  // namespace DirectMappedLxuCacheLookupSimt

#endif
