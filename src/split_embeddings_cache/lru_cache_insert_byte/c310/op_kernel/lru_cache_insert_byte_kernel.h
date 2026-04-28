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

#ifndef LRU_CACHE_INSERT_BYTE_KERNEL_H
#define LRU_CACHE_INSERT_BYTE_KERNEL_H

#include <cstdint>
#include "kernel_operator.h"
#include "simt_kernel.h"

using namespace AscendC;

namespace LruCacheInsertByte {

struct Args {
    GM_ADDR weights;
    GM_ADDR cache_hash_size_cumsum;
    GM_ADDR cache_index_table_map;
    GM_ADDR weights_offsets;
    GM_ADDR weights_tys;
    GM_ADDR d_offsets;
    GM_ADDR sorted_cache_sets;
    GM_ADDR cache_set_sorted_unique_indices;
    GM_ADDR unique_indices_length;
    GM_ADDR lxu_cache_state;
    GM_ADDR lxu_cache_weights;
    GM_ADDR lru_state;
    GM_ADDR uvm_cache_stats;
    GM_ADDR reserved_out;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

template <typename IndexT>
class LruCacheInsertByteKernel {
public:
    __aicore__ inline LruCacheInsertByteKernel(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        Nbuf = tilingData.bufferLength;
        C = static_cast<int32_t>(tilingData.numCacheSets);
        W = static_cast<int32_t>(tilingData.numWays);
        rowBytes = tilingData.cacheWeightsRowBytes;
        weightsLen = tilingData.weightsTotalLength;
        uvmLen = tilingData.uvmStatsLength;
        gatherStats = tilingData.gatherCacheStats != 0;
        ts = tilingData.timeStamp;
        rowAlign = static_cast<int32_t>(tilingData.rowAlignment);
        numTables = static_cast<int32_t>(tilingData.numTables);
        hashCumLen = tilingData.hashCumsumLength;
        mapLen = tilingData.cacheIndexMapLength;

        weightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.weights), weightsLen);
        hashCumsumGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.cache_hash_size_cumsum), hashCumLen);
        cacheIndexMapGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args.cache_index_table_map), mapLen);
        weightsOffsetsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.weights_offsets), numTables);
        weightsTysGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.weights_tys), numTables);
        dOffsetsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args.d_offsets), numTables + 1);
        sortedCacheSetsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args.sorted_cache_sets), Nbuf);
        cacheSetSortedIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ IndexT*>(args.cache_set_sorted_unique_indices), Nbuf);
        uniqueLenGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args.unique_indices_length), 1);
        const int64_t cw = static_cast<int64_t>(C) * static_cast<int64_t>(W);
        lxuCacheStateGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.lxu_cache_state), cw);
        lxuCacheWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.lxu_cache_weights), cw * rowBytes);
        lruStateGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.lru_state), cw);
        uvmStatsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args.uvm_cache_stats), uvmLen);
        reservedOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args.reserved_out), 1);
        (void)args.workspace;

        gmWeights_ = reinterpret_cast<__gm__ uint8_t*>(args.weights);
        gmHashCumsum_ = reinterpret_cast<__gm__ int64_t*>(args.cache_hash_size_cumsum);
        gmCacheIndexMap_ = reinterpret_cast<__gm__ int32_t*>(args.cache_index_table_map);
        gmWeightsOffsets_ = reinterpret_cast<__gm__ int64_t*>(args.weights_offsets);
        gmWeightsTys_ = reinterpret_cast<__gm__ uint8_t*>(args.weights_tys);
        gmDOffsets_ = reinterpret_cast<__gm__ int32_t*>(args.d_offsets);
        gmSortedCacheSets_ = reinterpret_cast<__gm__ int32_t*>(args.sorted_cache_sets);
        gmCacheSetSortedIdx_ = reinterpret_cast<__gm__ IndexT*>(args.cache_set_sorted_unique_indices);
        gmUniqueLen_ = reinterpret_cast<__gm__ int32_t*>(args.unique_indices_length);
        gmLxuCacheState_ = reinterpret_cast<__gm__ int64_t*>(args.lxu_cache_state);
        gmLxuCacheWeights_ = reinterpret_cast<__gm__ uint8_t*>(args.lxu_cache_weights);
        gmLruState_ = reinterpret_cast<__gm__ int64_t*>(args.lru_state);
        gmUvmStats_ = reinterpret_cast<__gm__ int32_t*>(args.uvm_cache_stats);
    }

    __aicore__ inline void Compute()
    {
        if (GetBlockIdx() == 0) {
            reservedOutGm.SetValue(0, 0);
        }

        asc_vf_call<CumsumSimt::SmallDataCompute<IndexT>>(
            dim3{static_cast<uint32_t>(MAX_THREADS_PER_BLOCK), 1, 1},
            gmWeights_,
            gmHashCumsum_,
            gmCacheIndexMap_,
            gmWeightsOffsets_,
            gmWeightsTys_,
            gmDOffsets_,
            gmSortedCacheSets_,
            gmCacheSetSortedIdx_,
            gmUniqueLen_,
            gmLxuCacheState_,
            gmLxuCacheWeights_,
            static_cast<int32_t>(rowBytes),
            rowAlign,
            ts,
            gmLruState_,
            gatherStats,
            gmUvmStats_,
            false,
            nullptr,
            C,
            W,
            static_cast<int32_t>(Nbuf));
    }

private:
    int64_t Nbuf;
    // C = numCacheSets, W = numWays
    int32_t C;
    int32_t W;
    int64_t rowBytes;
    int64_t weightsLen;
    int64_t uvmLen;
    bool gatherStats;
    int64_t ts;
    int32_t rowAlign;
    int32_t numTables;
    int64_t hashCumLen;
    int64_t mapLen;
    __gm__ uint8_t* gmWeights_;
    __gm__ int64_t* gmHashCumsum_;
    __gm__ int32_t* gmCacheIndexMap_;
    __gm__ int64_t* gmWeightsOffsets_;
    __gm__ uint8_t* gmWeightsTys_;
    __gm__ int32_t* gmDOffsets_;
    __gm__ int32_t* gmSortedCacheSets_;
    __gm__ IndexT* gmCacheSetSortedIdx_;
    __gm__ int32_t* gmUniqueLen_;
    __gm__ int64_t* gmLxuCacheState_;
    __gm__ uint8_t* gmLxuCacheWeights_;
    __gm__ int64_t* gmLruState_;
    __gm__ int32_t* gmUvmStats_;
    GlobalTensor<uint8_t> weightsGm;
    GlobalTensor<int64_t> hashCumsumGm;
    GlobalTensor<int32_t> cacheIndexMapGm;
    GlobalTensor<int64_t> weightsOffsetsGm;
    GlobalTensor<uint8_t> weightsTysGm;
    GlobalTensor<int32_t> dOffsetsGm;
    GlobalTensor<int32_t> sortedCacheSetsGm;
    GlobalTensor<IndexT> cacheSetSortedIdxGm;
    GlobalTensor<int32_t> uniqueLenGm;
    GlobalTensor<int64_t> lxuCacheStateGm;
    GlobalTensor<uint8_t> lxuCacheWeightsGm;
    GlobalTensor<int64_t> lruStateGm;
    GlobalTensor<int32_t> uvmStatsGm;
    GlobalTensor<int32_t> reservedOutGm;
};

}  // namespace LruCacheInsertByte

#endif  // LRU_CACHE_INSERT_BYTE_KERNEL_H
