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

#ifndef LRU_CACHE_FIND_UNCACHED_KERNEL_H
#define LRU_CACHE_FIND_UNCACHED_KERNEL_H

#include <cstdint>
#include "kernel_operator.h"
#include "simt_kernel.h"

using namespace AscendC;

namespace LruCacheFindUncached {

struct Args {
    GM_ADDR unique_indices;
    GM_ADDR unique_indices_length;
    GM_ADDR lxu_cache_state;
    GM_ADDR lru_state;
    GM_ADDR uvm_cache_stats;
    GM_ADDR lxu_cache_locking_counter;
    GM_ADDR cache_sets;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

template <typename IndexT>
class LruCacheFindUncachedKernel {
public:
    __aicore__ inline LruCacheFindUncachedKernel(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        N = tilingData.totalLength;
        C = static_cast<int32_t>(tilingData.numCacheSets);
        W = static_cast<int32_t>(tilingData.numWays);
        uvmLen = tilingData.uvmStatsLength;
        lockLen = tilingData.lockCounterLength;
        gatherStats = tilingData.gatherCacheStats != 0;
        maxIdx = tilingData.maxIndices;
        ts = tilingData.timeStamp;
        lockLine = tilingData.lockCacheLine != 0;

        uniqueIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ IndexT*>(args.unique_indices), N);
        uniqueIndicesLengthGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(args.unique_indices_length), 1);
        const int64_t cw = static_cast<int64_t>(C) * static_cast<int64_t>(W);
        lxuCacheStateGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.lxu_cache_state), cw);
        lruStateGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.lru_state), cw);
        uvmCacheStatsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args.uvm_cache_stats), uvmLen);
        lxuCacheLockingCounterGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(args.lxu_cache_locking_counter), lockLen);
        cacheSetsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args.cache_sets), N);
        (void)args.workspace;

        gmUniqueIndices_ = reinterpret_cast<__gm__ IndexT*>(args.unique_indices);
        gmUniqueLen_ = reinterpret_cast<__gm__ int32_t*>(args.unique_indices_length);
        gmLxuCacheState_ = reinterpret_cast<__gm__ int64_t*>(args.lxu_cache_state);
        gmLruState_ = reinterpret_cast<__gm__ int64_t*>(args.lru_state);
        gmUvmStats_ = reinterpret_cast<__gm__ int32_t*>(args.uvm_cache_stats);
        gmLxuCacheLockingCounter_ = reinterpret_cast<__gm__ int32_t*>(args.lxu_cache_locking_counter);
        gmCacheSets_ = reinterpret_cast<__gm__ int32_t*>(args.cache_sets);
    }

    __aicore__ inline void Compute()
    {
        asc_vf_call<LruCacheFindUncachedSimt::FindUncachedSmallDataCompute<IndexT>>(
            dim3{static_cast<uint32_t>(LruCacheFindUncachedSimt::MAX_THREADS_PER_BLOCK), 1, 1},
            gmUniqueIndices_,
            N,
            gmUniqueLen_,
            maxIdx,
            static_cast<int32_t>(N),
            gmLxuCacheState_,
            gmCacheSets_,
            ts,
            gmLruState_,
            gatherStats,
            gmUvmStats_,
            uvmLen,
            lockLine,
            gmLxuCacheLockingCounter_,
            lockLen,
            C,
            W);
    }

private:
    int64_t N;
    int32_t C;
    int32_t W;
    int64_t uvmLen;
    int64_t lockLen;
    bool gatherStats;
    int64_t maxIdx;
    int64_t ts;
    bool lockLine;
    __gm__ IndexT* gmUniqueIndices_;
    __gm__ int32_t* gmUniqueLen_;
    __gm__ int64_t* gmLxuCacheState_;
    __gm__ int64_t* gmLruState_;
    __gm__ int32_t* gmUvmStats_;
    __gm__ int32_t* gmLxuCacheLockingCounter_;
    __gm__ int32_t* gmCacheSets_;
    GlobalTensor<IndexT> uniqueIndicesGm;
    GlobalTensor<int32_t> uniqueIndicesLengthGm;
    GlobalTensor<int64_t> lxuCacheStateGm;
    GlobalTensor<int64_t> lruStateGm;
    GlobalTensor<int32_t> uvmCacheStatsGm;
    GlobalTensor<int32_t> lxuCacheLockingCounterGm;
    GlobalTensor<int32_t> cacheSetsGm;
};

}  // namespace LruCacheFindUncached

#endif  // LRU_CACHE_FIND_UNCACHED_KERNEL_H
