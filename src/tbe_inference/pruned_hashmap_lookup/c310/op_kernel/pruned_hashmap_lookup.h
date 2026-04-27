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

#ifndef PRUNED_HASHMAP_LOOKUP_H
#define PRUNED_HASHMAP_LOOKUP_H

#include <type_traits>
#include "simt_api/asc_simt.h"

#include "common.h"

namespace PrunedHashmapLookup {

constexpr int32_t WARP_SIZE = 32; // 每个Warp线程数
constexpr int32_t BLOCK_DIM_0 = 256;
constexpr int32_t WARP_NUM = BLOCK_DIM_0 / WARP_SIZE;
constexpr int32_t TABLE_DIM = 2;

__simt_callee__ inline uint32_t PrunedHashFunction(uint32_t h)
{
    // MurmorHash3 32-bit mixing function.
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

__simt_callee__ inline uint64_t PrunedHashFunction(uint64_t k)
{
    // MurmorHash3 64-bit mixing function.
    k ^= k >> 33;
    k *= (0xff51afd7ed558ccd);
    k ^= k >> 33;
    k *= (0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    return k;
}

template <typename DataType>
__simt_vf__ __launch_bounds__(SIMT_LAUNCH_BOUND)
    inline void PrunedHashmapLookupSimt(
        __gm__ DataType* indices,
        __gm__ DataType* hashTable,
        __gm__ DataType* denseIndices,
        DataType batchIndexStart,
        DataType segmentLength,
        int64_t hashTableStart,
        int64_t hashTableEnd)
{
    const int32_t warpId = threadIdx.x / WARP_SIZE;
    const int32_t laneId = threadIdx.x % WARP_SIZE;

    using UIDX_T = std::conditional_t<std::is_same_v<DataType, int64_t>, uint64_t, uint32_t>;
    int64_t capacity = hashTableEnd - hashTableStart;

    // 直接返回原始的index为dense_indices，即为不做index剪枝
    if (capacity == 0) {
        for (auto l = threadIdx.x; l < segmentLength; l += blockDim.x) {
            denseIndices[batchIndexStart + l] = indices[batchIndexStart + l];
        }
        return;
    }

    // 每个Warp处理一个index，外层循环每一次就处理 WARP_NUM 个index
    for (int32_t lStart = 0; lStart + warpId < segmentLength; lStart += WARP_NUM) {
        const DataType idx = indices[batchIndexStart + lStart + warpId];
        auto slotStart = PrunedHashFunction(static_cast<UIDX_T>(idx)) % capacity;  // 找到索引分桶

        while (true) {
            const auto slot = (slotStart + laneId) % capacity;  // 计算当前线程要探测的槽位

            // 获取hash table中对应槽位的值
            DataType slotSparseIdx = hashTable[(hashTableStart + static_cast<int64_t>(slot)) * TABLE_DIM];
            DataType slotDenseIdx = hashTable[(hashTableStart + static_cast<int64_t>(slot)) * TABLE_DIM + 1];

            int32_t found = 0;
            int32_t empty = 0;
            if (slotSparseIdx == -1) {
                empty = 1;
            } else if (slotSparseIdx == idx) {
                found = 1;
                denseIndices[batchIndexStart + lStart + warpId] = slotDenseIdx;
            }

            // 退出条件为找到dense indices或者empty, 每个表至少要有一个空槽位
            if (asc_any(found)) {
                break;
            } else if (asc_any(empty)) {
                denseIndices[batchIndexStart + lStart + warpId] = -1;
                break;
            }
            slotStart += WARP_SIZE; // 一个Warp 32个线程，一次探测32个槽位
        }
    }
}

template <typename T>
class PrunedHashmapLookupKernel {
public:
    __aicore__ inline PrunedHashmapLookupKernel(Args& args)
    {
        GET_TILING_DATA(tilingData, args.tiling);

        InitTilingParams(tilingData);
        InitGmParams(args);
    }

    __aicore__ inline void Compute(Args& args)
    {
        ProcessAllBatch(args);
    }

private:
    __aicore__ inline void InitTilingParams(const PrunedHashmapLookupTilingData& tilingData)
    {
        batchNum = tilingData.batchNum;
        batchPerTable = tilingData.batchPerTable;
        tableNum = tilingData.tableNum;
        bigCore = tilingData.bigCore;
        batchNumPerCore = tilingData.batchNumPerCore;
        indicesLen = tilingData.indicesLen;
        offsetsLen = tilingData.offsetsLen;
        hashTableLen = tilingData.hashTableLen;
        hashTableOffsetsLen = tilingData.hashTableOffsetsLen;
    }

    __aicore__ inline void InitGmParams(const Args& args)
    {
        indicesGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args.indices), indicesLen);
        offsetsGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args.offsets), offsetsLen);
        hashTableGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args.hashTable), hashTableLen);
        hashTableOffsetsGT.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.hashTableOffsets),
                                           hashTableOffsetsLen);
        denseIndicesGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args.denseIndices), indicesLen);
    }

    __aicore__ inline void ProcessAllBatch(Args& args)
    {
        int64_t coreIdx = GetBlockIdx();
        int64_t batchCount;
        int64_t batchStart;
        if (coreIdx < bigCore) {
            batchCount = batchNumPerCore + 1;
            batchStart = coreIdx * batchCount;
        } else {
            batchCount = batchNumPerCore;
            batchStart = bigCore * (batchNumPerCore + 1) + (coreIdx - bigCore) * batchNumPerCore;
        }

        for (int64_t i = 0; i < batchCount; ++i) {
            int64_t batchIdx = batchStart + i;
            if (batchIdx >= batchNum) {
                break;
            }
            ProcessSingleBatch(args, batchIdx);
        }
    }

    __aicore__ inline void ProcessSingleBatch(
        Args& args,
        int64_t batchIdx)
    {
        T batchIndexStart = static_cast<T>(offsetsGT.GetValue(batchIdx));
        T batchIndexEnd = static_cast<T>(offsetsGT.GetValue(batchIdx + 1));
        T segmentLength = batchIndexEnd - batchIndexStart;
        if (segmentLength <= 0) {
            return;
        }

        T tableIndex = batchIdx / batchPerTable;
        int64_t hashTableStart = hashTableOffsetsGT.GetValue(tableIndex);
        int64_t hashTableEnd = hashTableOffsetsGT.GetValue(tableIndex + 1);

        ProcessSimt(args, batchIndexStart, segmentLength, hashTableStart, hashTableEnd);
    }

    __aicore__ inline void ProcessSimt(
        Args& args,
        T batchIndexStart,
        T segmentLength,
        int64_t hashTableStart,
        int64_t hashTableEnd)
    {
        __gm__ T* indices = reinterpret_cast<__gm__ T*>(args.indices);
        __gm__ T* hashTable = reinterpret_cast<__gm__ T*>(args.hashTable);
        __gm__ T* denseIndices = reinterpret_cast<__gm__ T*>(args.denseIndices);

        asc_vf_call<PrunedHashmapLookupSimt<T>>(
            dim3{static_cast<uint32_t>(BLOCK_DIM_0), 1, 1},
            indices,
            hashTable,
            denseIndices,
            batchIndexStart,
            segmentLength,
            hashTableStart,
            hashTableEnd);
    }

private:
    TPipe pipe;

    GlobalTensor<T> indicesGT;
    GlobalTensor<T> offsetsGT;
    GlobalTensor<T> hashTableGT;
    GlobalTensor<int64_t> hashTableOffsetsGT;
    GlobalTensor<T> denseIndicesGT;  // 输出tensor

    int64_t batchNum;
    int64_t batchPerTable;
    int64_t tableNum;
    int64_t bigCore;
    int64_t batchNumPerCore;
    int64_t indicesLen;
    int64_t offsetsLen;
    int64_t hashTableLen;
    int64_t hashTableOffsetsLen;
};

}  // namespace PrunedHashmapLookup

#endif  // PRUNED_HASHMAP_LOOKUP_H
