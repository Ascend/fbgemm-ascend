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

#ifndef LXU_CACHE_LOOKUP_SIMT_KERNEL_H
#define LXU_CACHE_LOOKUP_SIMT_KERNEL_H

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t CACHE_LOCATION_MISSING = -1;

// uvm_cache_stats 索引
constexpr int32_t UVM_NUM_CONFLICT_MISSES = 5;

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

/// Warp 内是否存在任一线程 v 为真（等价 __any_sync 语义）。
__simt_callee__ inline bool WarpAny(bool v)
{
    int32_t b = v ? 1 : 0;
#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1) {
        b |= asc_shfl_xor(b, off);
    }
    return b != 0;
}

/// 获取 warp 内第一个为 true 的 lane ID
__simt_callee__ inline int32_t WarpFindFirst(bool v, int32_t myLaneId)
{
    // 将当前 lane 的 ID（如果 found 为 true）广播给所有 lane
    // 然后取最小的 lane ID
    int32_t laneWithMatch = v ? myLaneId : -1;
    int32_t minLane = laneWithMatch;
#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1) {
        int32_t other = asc_shfl_xor(minLane, off);
        if (other >= 0 && (minLane < 0 || other < minLane)) {
            minLane = other;
        }
    }
    return minLane;
}

/// Warp 内求和（32 个线程）
__simt_callee__ inline int32_t WarpSum(int32_t value)
{
    int32_t sum = value;
#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1) {
        sum += asc_shfl_xor(sum, off);
    }
    return sum;
}

namespace LxuCacheLookupSimt {

template <typename IndexT>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void LxuCacheLookupCompute(
    __gm__ IndexT* linearCacheIndices, __gm__ int64_t* lxuCacheState, int64_t invalidIndex,
    __gm__ int32_t* lxuCacheLocations, bool gatherCacheStats, __gm__ int32_t* uvmCacheStats, __gm__ int32_t* nUnique,
    int32_t C, int32_t W, int32_t totalLength, bool uniqLookup)
{
    // 每个 warp 处理一个索引，warp 内 32 个线程并行检查 32 个 way
    const int32_t warpId = Simt::GetThreadIdx<0>() / WARP_SIZE;
    const int32_t laneId = Simt::GetThreadIdx<0>() % WARP_SIZE;
    const int32_t numWarpsPerBlock = MAX_THREADS_PER_BLOCK / WARP_SIZE;
    const int32_t bid = Simt::GetBlockIdx();
    int32_t grid = Simt::GetBlockNum();
    if (grid <= 0) {
        grid = 1;
    }

    // 计算有效索引数 N, 非 uniq 模式时使用 totalLength
    const int32_t N = uniqLookup ? ((nUnique != nullptr) ? *nUnique : 0) : totalLength;

    if (N <= 0) {
        return;
    }

    // 每个 warp 处理一个索引，warp 内 32 个线程并行检查 32 个 way
    for (int32_t n = bid * numWarpsPerBlock + warpId; n < N; n += grid * numWarpsPerBlock) {
        const int64_t idx = static_cast<int64_t>(linearCacheIndices[n]);

        // Skip invalid index
        if (idx == invalidIndex) {
            if (laneId == 0) {
                lxuCacheLocations[n] = CACHE_LOCATION_MISSING;
            }
            continue;
        }

        const int32_t cacheSet = CacheSlot(idx, C);

        // laneId (0-31) 检查 lxuCacheState[cacheSet][laneId]
        bool found = false;
        if (laneId < W) {
            const int64_t pos = static_cast<int64_t>(cacheSet) * static_cast<int64_t>(W) + laneId;
            found = (lxuCacheState[pos] == idx);
        }

        // 使用 WarpAny 检测
        if (WarpAny(found)) {
            // 获取第一个匹配的 way (lane ID)
            const int32_t way = WarpFindFirst(found, laneId);
            // 只有 lane 0 写入结果
            if (laneId == 0) {
                if (way >= 0 && way < W) {
                    lxuCacheLocations[n] = cacheSet * WARP_SIZE + way;
                } else {
                    lxuCacheLocations[n] = CACHE_LOCATION_MISSING;
                }
            }
        } else {
            // 没有找到匹配的 way
            if (laneId == 0) {
                lxuCacheLocations[n] = CACHE_LOCATION_MISSING;
            }
        }

        // 统计冲突未命中: gather_cache_stats 时统计
        if (gatherCacheStats && uvmCacheStats != nullptr && laneId == 0) {
            int32_t nIndices = 1;
            int32_t nHits = WarpSum(found ? 1 : 0);
            int32_t nConflictMisses = nIndices - nHits;
            if (nConflictMisses > 0) {
                atomicAdd(&uvmCacheStats[UVM_NUM_CONFLICT_MISSES], nConflictMisses);
            }
        }
    }
}

}  // namespace LxuCacheLookupSimt

#endif  // LXU_CACHE_LOOKUP_SIMT_KERNEL_H
