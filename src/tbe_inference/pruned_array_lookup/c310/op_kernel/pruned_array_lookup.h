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

#ifndef PRUNED_ARRAY_LOOKUP_H
#define PRUNED_ARRAY_LOOKUP_H

#include <type_traits>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

namespace PrunedArrayLookup {

static constexpr int32_t SIMT_THREAD_NUM = 1024;
static constexpr int32_t WARP_SIZE = 32;
static constexpr int32_t WARP_NUM = SIMT_THREAD_NUM / WARP_SIZE;

struct Args {
    GM_ADDR indices;
    GM_ADDR offsets;
    GM_ADDR indexRemappings;
    GM_ADDR indexRemappingsOffsets;
    GM_ADDR denseIndices;  // 输出tensor
    GM_ADDR workspace;
    GM_ADDR tiling;
};

template <typename DataType, typename ArrayType>
__simt_vf__ __launch_bounds__(SIMT_THREAD_NUM) inline void PrunedArrayLookupSimtMultiBatch(
    const __gm__ DataType* indices, const __gm__ DataType* offsets, const __gm__ ArrayType* indexRemappings,
    const __gm__ int64_t* indexRemappingsOffset, __gm__ DataType* denseIndices, int64_t batchStart, int64_t batchCount,
    int64_t batchPerTable)
{
    const int64_t warpId = threadIdx.x / WARP_SIZE;
    const int64_t laneId = threadIdx.x % WARP_SIZE;

    // 每个WARP处理多个batch，使用WARP数量作为步长
    for (int64_t warpBatchIdx = warpId; batchStart + warpBatchIdx < batchStart + batchCount; warpBatchIdx += WARP_NUM) {
        const int64_t batchIdx = batchStart + warpBatchIdx;
        const int64_t batchIndexStart = offsets[batchIdx];
        const int64_t batchIndexEnd = offsets[batchIdx + 1];
        const int64_t segmentLength = batchIndexEnd - batchIndexStart;

        const int64_t tableIdx = batchIdx / batchPerTable;
        const int64_t indexRemappingsStart = indexRemappingsOffset[tableIdx];

        // WARP内的每个线程处理segment的一部分
        if (indexRemappingsOffset[tableIdx + 1] == indexRemappingsStart) {
            // 不做剪枝，直接复制
            for (int64_t l = laneId; l < segmentLength; l += WARP_SIZE) {
                const int64_t currentIndex = batchIndexStart + l;
                denseIndices[currentIndex] = indices[currentIndex];
            }
        } else {
            // 执行索引重映射
            for (int64_t l = laneId; l < segmentLength; l += WARP_SIZE) {
                const int64_t currentIndex = batchIndexStart + l;
                const int64_t idx = indices[currentIndex];
                denseIndices[currentIndex] = static_cast<DataType>(indexRemappings[indexRemappingsStart + idx]);
            }
        }
    }
}

template <typename INDICES_T, typename INDEX_REMAPPINGS_T>
class PrunedArrayLookupKernel {
public:
    __aicore__ inline PrunedArrayLookupKernel(Args& args)
    {
        GET_TILING_DATA(tilingData, args.tiling);

        InitTilingParams(tilingData);
    }

    __aicore__ inline void Compute(Args& args)
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

        __gm__ INDICES_T* indices = reinterpret_cast<__gm__ INDICES_T*>(args.indices);
        __gm__ INDICES_T* offsets = reinterpret_cast<__gm__ INDICES_T*>(args.offsets);
        __gm__ INDEX_REMAPPINGS_T* indexRemappings = reinterpret_cast<__gm__ INDEX_REMAPPINGS_T*>(args.indexRemappings);
        __gm__ int64_t* indexRemappingsOffset = reinterpret_cast<__gm__ int64_t*>(args.indexRemappingsOffsets);
        __gm__ INDICES_T* denseIndices = reinterpret_cast<__gm__ INDICES_T*>(args.denseIndices);

        // 处理多batch
        uint32_t threadNum = tableNum * batchNum * WARP_SIZE;
        threadNum = (threadNum + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
        threadNum = threadNum > SIMT_THREAD_NUM ? SIMT_THREAD_NUM : threadNum;
        AscendC::Simt::Dim3 blockDim(threadNum, 1, 1);
        asc_vf_call<PrunedArrayLookupSimtMultiBatch<INDICES_T, INDEX_REMAPPINGS_T>>(
            blockDim, indices, offsets, indexRemappings, indexRemappingsOffset, denseIndices, batchStart, batchCount,
            batchPerTable);
    }

private:
    __aicore__ inline void InitTilingParams(const PrunedArrayLookupTilingData& tilingData)
    {
        batchNum = tilingData.batchNum;
        batchPerTable = tilingData.batchPerTable;
        tableNum = tilingData.tableNum;
        bigCore = tilingData.bigCore;
        batchNumPerCore = tilingData.batchNumPerCore;
        indicesLen = tilingData.indicesLen;
        offsetsLen = tilingData.offsetsLen;
        indexRemappingsLen = tilingData.indexRemappingsLen;
        indexRemappingsOffsetsLen = tilingData.indexRemappingsOffsetsLen;
    }

    int64_t batchNum;
    int64_t batchPerTable;
    int64_t tableNum;
    int64_t bigCore;
    int64_t batchNumPerCore;
    int64_t indicesLen;
    int64_t offsetsLen;
    int64_t indexRemappingsLen;
    int64_t indexRemappingsOffsetsLen;
};

}  // namespace PrunedArrayLookup

#endif  // PRUNED_ARRAY_LOOKUP_H
