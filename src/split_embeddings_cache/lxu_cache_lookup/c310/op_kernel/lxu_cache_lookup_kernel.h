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

#ifndef LXU_CACHE_LOOKUP_KERNEL_H
#define LXU_CACHE_LOOKUP_KERNEL_H

#include "simt_kernel.h"

struct Args {
    GM_ADDR linear_cache_indices;
    GM_ADDR lxu_cache_state;
    GM_ADDR uvm_cache_stats;
    GM_ADDR num_uniq_cache_indices;
    GM_ADDR lxu_cache_locations;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

namespace LxuCacheLookup {

template <typename IndexT>
class LxuCacheLookupKernel {
public:
    __aicore__ inline LxuCacheLookupKernel(Args& args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        this->N = tilingData.totalLength;
        this->C = tilingData.numCacheSets;
        this->W = tilingData.numWays;
        this->gatherStats = tilingData.gatherCacheStats != 0;
        this->invalidIndex = tilingData.invalidIndex;
        this->uniqLookup = tilingData.uniqLookup != 0;

        linearCacheIndicesGm = reinterpret_cast<__gm__ IndexT*>(args.linear_cache_indices);
        lxuCacheStateGm = reinterpret_cast<__gm__ int64_t*>(args.lxu_cache_state);
        lxuCacheLocationsGm = reinterpret_cast<__gm__ int32_t*>(args.lxu_cache_locations);
        uvmCacheStatsGm = reinterpret_cast<__gm__ int32_t*>(args.uvm_cache_stats);
        numUniqCacheIndicesGm = reinterpret_cast<__gm__ int32_t*>(args.num_uniq_cache_indices);
    }

    __aicore__ inline void Compute()
    {
        // 使用 dim3(1024, 1, 1)，每个 warp 处理一个索引
        // warp 级并行：warpId = threadIdx.x / 32, laneId = threadIdx.x % 32
        asc_vf_call<LxuCacheLookupSimt::LxuCacheLookupCompute<IndexT>>(
            dim3{static_cast<uint32_t>(MAX_THREADS_PER_BLOCK), 1, 1}, this->linearCacheIndicesGm, this->lxuCacheStateGm,
            this->invalidIndex, this->lxuCacheLocationsGm, this->gatherStats, this->uvmCacheStatsGm,
            this->numUniqCacheIndicesGm, this->C, this->W, this->N, this->uniqLookup);
    }

private:
    int32_t N;
    int32_t C;
    int32_t W;
    int64_t invalidIndex;
    bool gatherStats;
    bool uniqLookup;
    __gm__ IndexT* linearCacheIndicesGm;
    __gm__ int64_t* lxuCacheStateGm;
    __gm__ int32_t* lxuCacheLocationsGm;
    __gm__ int32_t* uvmCacheStatsGm;
    __gm__ int32_t* numUniqCacheIndicesGm;
};

}  // namespace LxuCacheLookup

#endif  // LXU_CACHE_LOOKUP_KERNEL_H
