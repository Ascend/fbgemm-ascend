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

#include "linearize_cache_indices_kernel.h"
#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void linearize_cache_indices(GM_ADDR cache_hash_size_cumsum, GM_ADDR indices,
                                                              GM_ADDR table_offsets, GM_ADDR linearized_indices,
                                                              GM_ADDR workspace, GM_ADDR tiling_data)
{
    GET_TILING_DATA(tilingDataStruct, tiling_data);

    int64_t indicesNum = tilingDataStruct.numIndices;
    int64_t numTables = tilingDataStruct.numTables;
    int64_t indicesBaseOffset = tilingDataStruct.indicesBaseOffset;
    int64_t tableOffsetsSize = tilingDataStruct.tableOffsetsSize;

    int64_t perCoreIndicesNum = indicesNum / GetBlockNum();
    int64_t blockDim = (perCoreIndicesNum > kMaxThreads) ? kMaxThreads : perCoreIndicesNum;
    AscendC::Simt::Dim3 simtDim{static_cast<uint32_t>(blockDim), 1, 1};
    AscendC::Simt::VF_CALL<SimtLinearizeIndicesMultiThread<DTYPE_INDICES, DTYPE_TABLE_OFFSETS>>(
        simtDim, (__gm__ int64_t*)cache_hash_size_cumsum, (__gm__ DTYPE_INDICES*)indices,
        (__gm__ DTYPE_TABLE_OFFSETS*)table_offsets, (__gm__ int64_t*)linearized_indices, indicesNum, numTables,
        indicesBaseOffset, tableOffsetsSize);
}