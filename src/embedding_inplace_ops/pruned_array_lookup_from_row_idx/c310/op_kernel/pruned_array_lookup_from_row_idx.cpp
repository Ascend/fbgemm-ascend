/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

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

#include "pruned_array_lookup_from_row_idx_kernel.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void pruned_array_lookup_from_row_idx(GM_ADDR updateRowIndices,
                                                                       GM_ADDR updateTableIndices,
                                                                       GM_ADDR indexRemappings,
                                                                       GM_ADDR indexRemappingsOffsets,
                                                                       GM_ADDR denseIndices,
                                                                       GM_ADDR workspace,
                                                                       GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    const int64_t numIndices = tilingData.numIndices;
    const int32_t elemsPerBlock = tilingData.elemsPerBlock;
    const uint32_t threadsPerBlock = tilingData.threadsPerBlock;

    if (TILING_KEY_IS(0)) {
        AscendC::Simt::VF_CALL<PrunedArrayLookupFromRowIdx::SimtCompute<int64_t, int64_t>>(
            AscendC::Simt::Dim3{threadsPerBlock, 1, 1},
            reinterpret_cast<__gm__ int64_t*>(updateRowIndices),
            reinterpret_cast<__gm__ int32_t*>(updateTableIndices),
            reinterpret_cast<__gm__ int64_t*>(indexRemappings),
            reinterpret_cast<__gm__ int64_t*>(indexRemappingsOffsets),
            reinterpret_cast<__gm__ int64_t*>(denseIndices),
            numIndices,
            threadsPerBlock,
            elemsPerBlock);
    } else if (TILING_KEY_IS(1)) {
        AscendC::Simt::VF_CALL<PrunedArrayLookupFromRowIdx::SimtCompute<int64_t, int32_t>>(
            AscendC::Simt::Dim3{threadsPerBlock, 1, 1},
            reinterpret_cast<__gm__ int64_t*>(updateRowIndices),
            reinterpret_cast<__gm__ int32_t*>(updateTableIndices),
            reinterpret_cast<__gm__ int32_t*>(indexRemappings),
            reinterpret_cast<__gm__ int64_t*>(indexRemappingsOffsets),
            reinterpret_cast<__gm__ int64_t*>(denseIndices),
            numIndices,
            threadsPerBlock,
            elemsPerBlock);
    } else if (TILING_KEY_IS(2)) {
        AscendC::Simt::VF_CALL<PrunedArrayLookupFromRowIdx::SimtCompute<int32_t, int64_t>>(
            AscendC::Simt::Dim3{threadsPerBlock, 1, 1},
            reinterpret_cast<__gm__ int32_t*>(updateRowIndices),
            reinterpret_cast<__gm__ int32_t*>(updateTableIndices),
            reinterpret_cast<__gm__ int64_t*>(indexRemappings),
            reinterpret_cast<__gm__ int64_t*>(indexRemappingsOffsets),
            reinterpret_cast<__gm__ int32_t*>(denseIndices),
            numIndices,
            threadsPerBlock,
            elemsPerBlock);
    } else if (TILING_KEY_IS(3)) {
        AscendC::Simt::VF_CALL<PrunedArrayLookupFromRowIdx::SimtCompute<int32_t, int32_t>>(
            AscendC::Simt::Dim3{threadsPerBlock, 1, 1},
            reinterpret_cast<__gm__ int32_t*>(updateRowIndices),
            reinterpret_cast<__gm__ int32_t*>(updateTableIndices),
            reinterpret_cast<__gm__ int32_t*>(indexRemappings),
            reinterpret_cast<__gm__ int64_t*>(indexRemappingsOffsets),
            reinterpret_cast<__gm__ int32_t*>(denseIndices),
            numIndices,
            threadsPerBlock,
            elemsPerBlock);
    }
}
