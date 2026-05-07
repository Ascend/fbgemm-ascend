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

#ifndef PRUNED_ARRAY_LOOKUP_FROM_ROW_IDX_KERNEL_H
#define PRUNED_ARRAY_LOOKUP_FROM_ROW_IDX_KERNEL_H

#include "kernel_operator.h"

namespace PrunedArrayLookupFromRowIdx {
constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;

template <typename TRow, typename TRemap>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtCompute(
    __gm__ TRow* updateRow,
    __gm__ int32_t* updateTable,
    __gm__ TRemap* indexRemappings,
    __gm__ int64_t* offsets,
    __gm__ TRow* denseOut,
    int64_t numIndices,
    uint32_t threadsPerBlock,
    int32_t elemsPerBlock)
{
    const int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    const int64_t blockOffset = static_cast<int64_t>(block_idx) * static_cast<int64_t>(elemsPerBlock);

    for (int64_t i = threadIdx; i < static_cast<int64_t>(elemsPerBlock); i += static_cast<int64_t>(threadsPerBlock)) {
        const int64_t globalIdx = blockOffset + i;
        if (globalIdx < numIndices) {
            const TRow rowIdx = updateRow[globalIdx];
            const int32_t tableIdx = updateTable[globalIdx];
            const int64_t start = offsets[tableIdx];
            const int64_t end = offsets[tableIdx + 1];
            const int64_t capacity = end - start;
            if (capacity > 0) {
                const int64_t remapIdx = start + static_cast<int64_t>(rowIdx);
                denseOut[globalIdx] = static_cast<TRow>(indexRemappings[remapIdx]);
            } else {
                denseOut[globalIdx] = rowIdx;
            }
        }
    }
}
}  // namespace PrunedArrayLookupFromRowIdx

#endif  // PRUNED_ARRAY_LOOKUP_FROM_ROW_IDX_KERNEL_H
