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

extern "C" __global__ __aicore__ void pruned_array_lookup_from_row_idx(GM_ADDR update_row_indices,
                                                                       GM_ADDR update_table_indices,
                                                                       GM_ADDR index_remappings,
                                                                       GM_ADDR index_remappings_offsets,
                                                                       GM_ADDR dense_indices,
                                                                       GM_ADDR workspace,
                                                                       GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    const int64_t numIndices = tilingData.numIndices;
    const int32_t elemsPerBlock = tilingData.elemsPerBlock;
    const uint32_t threadsPerBlock = tilingData.threadsPerBlock;

    AscendC::Simt::VF_CALL<PrunedArrayLookupFromRowIdx::SimtCompute<DTYPE_UPDATE_ROW_INDICES, DTYPE_INDEX_REMAPPINGS>>(
        AscendC::Simt::Dim3{threadsPerBlock, 1, 1},
        reinterpret_cast<__gm__ DTYPE_UPDATE_ROW_INDICES*>(update_row_indices),
        reinterpret_cast<__gm__ DTYPE_UPDATE_TABLE_INDICES*>(update_table_indices),
        reinterpret_cast<__gm__ DTYPE_INDEX_REMAPPINGS*>(index_remappings),
        reinterpret_cast<__gm__ DTYPE_INDEX_REMAPPINGS_OFFSETS*>(index_remappings_offsets),
        reinterpret_cast<__gm__ DTYPE_UPDATE_ROW_INDICES*>(dense_indices),
        numIndices,
        threadsPerBlock,
        elemsPerBlock
    );
}
