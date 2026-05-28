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
#ifndef DIRECT_MAPPED_LRU_CACHE_INSERT_BYTE_KERNEL_H
#define DIRECT_MAPPED_LRU_CACHE_INSERT_BYTE_KERNEL_H

#include <cstdint>
#include "kernel_operator.h"
#include "simt_kernel.h"

using namespace AscendC;

namespace DirectMappedLruCacheInsertByte {

// Kernel 参数结构体：存放所有 GM 地址与占位参数
struct Args {
    GM_ADDR weights;
    GM_ADDR cache_hash_size_cumsum;
    GM_ADDR cache_index_table_map;
    GM_ADDR weights_offsets;
    GM_ADDR weights_tys;
    GM_ADDR d_offsets;
    GM_ADDR lxu_cache_state;
    GM_ADDR lxu_cache_weights;
    GM_ADDR lru_state;
    GM_ADDR linear_cache_indices;
    GM_ADDR lxu_cache_miss_timestamp;
    GM_ADDR cache_sets;
    GM_ADDR uvm_cache_stats;
    GM_ADDR reserved_out;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

template <typename IndexT>
// Kernel 类：解析 Tiling 数据、绑定 GM 缓冲区、发起 SIMT 调用
class DirectMappedLruCacheInsertByteKernel {
public:
    // 构造函数：从 Tiling 数据解析参数并绑定 GM 指针
    __aicore__ inline DirectMappedLruCacheInsertByteKernel(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        // 解析 Tiling 参数
        N = tilingData.totalLength;
        C = static_cast<int32_t>(tilingData.numCacheSets);
        rowBytes = static_cast<int32_t>(tilingData.cacheWeightsRowBytes);
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
        lxuCacheStateGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.lxu_cache_state), C);
        lxuCacheWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.lxu_cache_weights),
                                          static_cast<int64_t>(C) * static_cast<int64_t>(rowBytes));
        lruStateGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.lru_state), C);
        linearCacheIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ IndexT*>(args.linear_cache_indices), N);
        lxuCacheMissTimestampGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.lxu_cache_miss_timestamp), C);
        cacheSetsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args.cache_sets), N);
        uvmStatsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args.uvm_cache_stats), uvmLen);
        reservedOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args.reserved_out), 1);
        (void)args.workspace;

        // 绑定 GlobalTensor 缓冲区
        gmWeights_ = reinterpret_cast<__gm__ uint8_t*>(args.weights);
        gmHashCumsum_ = reinterpret_cast<__gm__ int64_t*>(args.cache_hash_size_cumsum);
        gmCacheIndexMap_ = reinterpret_cast<__gm__ int32_t*>(args.cache_index_table_map);
        gmWeightsOffsets_ = reinterpret_cast<__gm__ int64_t*>(args.weights_offsets);
        gmWeightsTys_ = reinterpret_cast<__gm__ uint8_t*>(args.weights_tys);
        gmDOffsets_ = reinterpret_cast<__gm__ int32_t*>(args.d_offsets);
        gmLxuCacheState_ = reinterpret_cast<__gm__ int64_t*>(args.lxu_cache_state);
        gmLxuCacheWeights_ = reinterpret_cast<__gm__ uint8_t*>(args.lxu_cache_weights);
        gmLruState_ = reinterpret_cast<__gm__ int64_t*>(args.lru_state);
        gmLinearCacheIndices_ = reinterpret_cast<__gm__ IndexT*>(args.linear_cache_indices);
        gmLxuCacheMissTimestamp_ = reinterpret_cast<__gm__ int64_t*>(args.lxu_cache_miss_timestamp);
        gmCacheSets_ = reinterpret_cast<__gm__ int32_t*>(args.cache_sets);
        gmUvmStats_ = reinterpret_cast<__gm__ int32_t*>(args.uvm_cache_stats);
    }

    // 每个 warp 处理一个 cache line 的填充
    __aicore__ inline void Compute()
    {
        // Block 0 初始化保留输出为 0
        if (GetBlockIdx() == 0) {
            reservedOutGm.SetValue(0, 0);
        }

        asc_vf_call<DirectMappedLruCacheInsertByte::InsertByteSmallDataCompute<IndexT>>(
            dim3{static_cast<uint32_t>(MAX_THREADS_PER_BLOCK), 1, 1}, gmWeights_, gmHashCumsum_, gmCacheIndexMap_,
            gmWeightsOffsets_, gmWeightsTys_, gmDOffsets_, gmLxuCacheState_, gmLxuCacheWeights_, ts, gmLruState_,
            gmLinearCacheIndices_, gmLxuCacheMissTimestamp_, gmCacheSets_, gatherStats, gmUvmStats_, C, rowBytes,
            rowAlign, static_cast<int32_t>(N));
    }

private:
    // Tiling 参数
    int64_t N;
    int32_t C;
    int32_t rowBytes;
    int64_t weightsLen;
    int64_t uvmLen;
    bool gatherStats;
    int64_t ts;
    int32_t rowAlign;
    int32_t numTables;
    int64_t hashCumLen;
    int64_t mapLen;

    // Global Memory 原始指针
    __gm__ uint8_t* gmWeights_;
    __gm__ int64_t* gmHashCumsum_;
    __gm__ int32_t* gmCacheIndexMap_;
    __gm__ int64_t* gmWeightsOffsets_;
    __gm__ uint8_t* gmWeightsTys_;
    __gm__ int32_t* gmDOffsets_;
    __gm__ int64_t* gmLxuCacheState_;
    __gm__ uint8_t* gmLxuCacheWeights_;
    __gm__ int64_t* gmLruState_;
    __gm__ IndexT* gmLinearCacheIndices_;
    __gm__ int64_t* gmLxuCacheMissTimestamp_;
    __gm__ int32_t* gmCacheSets_;
    __gm__ int32_t* gmUvmStats_;

    // GlobalTensor 缓冲区（用于向量化访存）
    GlobalTensor<uint8_t> weightsGm;
    GlobalTensor<int64_t> hashCumsumGm;
    GlobalTensor<int32_t> cacheIndexMapGm;
    GlobalTensor<int64_t> weightsOffsetsGm;
    GlobalTensor<uint8_t> weightsTysGm;
    GlobalTensor<int32_t> dOffsetsGm;
    GlobalTensor<int64_t> lxuCacheStateGm;
    GlobalTensor<uint8_t> lxuCacheWeightsGm;
    GlobalTensor<int64_t> lruStateGm;
    GlobalTensor<IndexT> linearCacheIndicesGm;
    GlobalTensor<int64_t> lxuCacheMissTimestampGm;
    GlobalTensor<int32_t> cacheSetsGm;
    GlobalTensor<int32_t> uvmStatsGm;
    GlobalTensor<int32_t> reservedOutGm;
};

}  // namespace DirectMappedLruCacheInsertByte

#endif  // DIRECT_MAPPED_LRU_CACHE_INSERT_BYTE_KERNEL_H
