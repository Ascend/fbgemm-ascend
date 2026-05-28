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
#ifndef DIRECT_MAPPED_LRU_CACHE_FIND_UNCACHED_KERNEL_H
#define DIRECT_MAPPED_LRU_CACHE_FIND_UNCACHED_KERNEL_H

#include <cstdint>
#include "kernel_operator.h"
#include "simt_kernel.h"

using namespace AscendC;

namespace DirectMappedLruCacheFindUncached {

// Kernel 参数结构体：存放所有 GM 地址与占位参数
struct Args {
    GM_ADDR linear_cache_indices;
    GM_ADDR lxu_cache_state;
    GM_ADDR lru_state;
    GM_ADDR lxu_cache_miss_timestamp;
    GM_ADDR uvm_cache_stats;
    GM_ADDR cache_sets;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

template <typename IndexT>
// Kernel 类：解析 Tiling 数据、绑定 GM 缓冲区、发起 SIMT 调用
class DirectMappedLruCacheFindUncachedKernel {
public:
    // 构造函数：从 Tiling 数据解析参数并绑定 GM 指针
    __aicore__ inline DirectMappedLruCacheFindUncachedKernel(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        N = tilingData.totalLength;
        C = static_cast<int32_t>(tilingData.numCacheSets);
        uvmLen = tilingData.uvmStatsLength;
        gatherStats = tilingData.gatherCacheStats != 0;
        maxIdx = tilingData.maxIndices;
        ts = tilingData.timeStamp;

        (void)args.workspace;

        gmLinearCacheIndices_ = reinterpret_cast<__gm__ IndexT*>(args.linear_cache_indices);
        gmLxuCacheState_ = reinterpret_cast<__gm__ int64_t*>(args.lxu_cache_state);
        gmLruState_ = reinterpret_cast<__gm__ int64_t*>(args.lru_state);
        gmUvmStats_ = reinterpret_cast<__gm__ int32_t*>(args.uvm_cache_stats);
        gmLxuCacheMissTimestamp_ = reinterpret_cast<__gm__ int64_t*>(args.lxu_cache_miss_timestamp);
        gmCacheSets_ = reinterpret_cast<__gm__ int32_t*>(args.cache_sets);
    }

    // 每个 warp 处理一个 cache line 的查找
    __aicore__ inline void Compute()
    {
        asc_vf_call<DirectMappedLruCacheFindUncachedSimt::FindUncachedSmallDataCompute<IndexT>>(
            dim3{static_cast<uint32_t>(DirectMappedLruCacheFindUncachedSimt::MAX_THREADS_PER_BLOCK), 1, 1},
            gmLinearCacheIndices_, N, maxIdx, gmLxuCacheState_, C, gmCacheSets_, ts, gmLruState_, gatherStats,
            gmUvmStats_, uvmLen, gmLxuCacheMissTimestamp_);
    }

private:
    // Tiling 参数
    int64_t N;
    int32_t C;
    int64_t uvmLen;
    bool gatherStats;
    int64_t maxIdx;
    int64_t ts;

    // Global Memory 缓冲区指针
    __gm__ IndexT* gmLinearCacheIndices_;
    __gm__ int64_t* gmLxuCacheState_;
    __gm__ int64_t* gmLruState_;
    __gm__ int32_t* gmUvmStats_;
    __gm__ int64_t* gmLxuCacheMissTimestamp_;
    __gm__ int32_t* gmCacheSets_;
};

}  // namespace DirectMappedLruCacheFindUncached

#endif  // DIRECT_MAPPED_LRU_CACHE_FIND_UNCACHED_KERNEL_H
