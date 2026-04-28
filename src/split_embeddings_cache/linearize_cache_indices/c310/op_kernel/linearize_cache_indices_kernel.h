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

#ifndef LINEARIZE_CACHE_INDICES_KERNEL_H
#define LINEARIZE_CACHE_INDICES_KERNEL_H

#include "kernel_operator.h"

using namespace AscendC;

constexpr int64_t kMaxThreads = 1024;

template <typename IndiceType, typename TableOffsetsType>
__simt_vf__ __aicore__ LAUNCH_BOUND(kMaxThreads) inline void SimtLinearizeIndicesMultiThread(
    __gm__ int64_t* cacheHashSizeCumsum, __gm__ IndiceType* indices, __gm__ TableOffsetsType* tableOffsets,
    __gm__ int64_t* linearCacheIndices, int64_t indicesNum, int64_t numTables, int64_t indicesBaseOffset,
    int64_t tableOffsetsSize)
{
    int32_t blockIdx = static_cast<int32_t>(AscendC::Simt::GetBlockIdx());
    int32_t threadIdx = static_cast<int32_t>(AscendC::Simt::GetThreadIdx<0>());
    int32_t threadNum = static_cast<int32_t>(AscendC::Simt::GetThreadNum<0>());
    int32_t blockNum = static_cast<int32_t>(AscendC::Simt::GetBlockNum());

    int64_t perCoreIndicesNum = indicesNum / blockNum;
    int64_t remainder = indicesNum % blockNum;

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

    if (start >= indicesNum) {
        return;
    }

    if (end > indicesNum) {
        end = indicesNum;
    }

    // 优化：将不变的值提取到循环外，减少 GM 访存
    const int64_t maxOffset = cacheHashSizeCumsum[numTables];
    const int64_t tableOffsetsSizeLocal = tableOffsetsSize;
    const int64_t indicesBaseOffsetLocal = indicesBaseOffset;
    const int64_t numTablesLocal = numTables;

    for (int64_t idx = start; idx < end; ++idx) {
        int left = 0;
        int right = tableOffsetsSizeLocal;
        const auto indexWithOffset = idx + indicesBaseOffsetLocal;

        while (left != right) {
            const int middle = left + (right - left) / 2;
            if (tableOffsets[middle] <= indexWithOffset) {
                left = middle + 1;
            } else {
                right = middle;
            }
        }
        int tableIdx = left;

        // 使用局部变量缓存，减少 GM 访存
        if (tableIdx >= 0 && tableIdx < numTablesLocal) {
            const auto currOffset = cacheHashSizeCumsum[tableIdx];
            const auto idxValue = indices[idx];
            if (currOffset >= 0 && idxValue >= 0) {
                linearCacheIndices[idx] = idxValue + currOffset;
            } else {
                linearCacheIndices[idx] = maxOffset;
            }
        } else {
            linearCacheIndices[idx] = maxOffset;
        }
    }
}

#endif