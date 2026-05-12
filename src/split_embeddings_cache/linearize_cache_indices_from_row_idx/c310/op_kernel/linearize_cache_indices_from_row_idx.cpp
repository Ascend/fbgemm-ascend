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
#include "simt_kernel.h"
#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void linearize_cache_indices_from_row_idx(
    GM_ADDR cache_hash_size_cumsum,
    GM_ADDR update_table_indices,
    GM_ADDR update_row_indices,
    GM_ADDR linear_cache_indices,
    GM_ADDR workspace,
    GM_ADDR tiling_data)
{
    GET_TILING_DATA(tiling_data_struct, tiling_data);
    int64_t totalLength = tiling_data_struct.totalLength;
    int64_t cumsumLength = tiling_data_struct.cumsumLength;
    
    constexpr int32_t kThreadNum = 1024;
    AscendC::Simt::Dim3 dim(kThreadNum, 1, 1);

    AscendC::Simt::VF_CALL<SimtLinearizeCacheIndicesFromRowIdxMultiThread<DTYPE_UPDATE_TABLE_INDICES, DTYPE_LINEAR_CACHE_INDICES>>(
        dim,
        (__gm__ int64_t*)cache_hash_size_cumsum,
        (__gm__ DTYPE_UPDATE_TABLE_INDICES*)update_table_indices,
        (__gm__ DTYPE_UPDATE_ROW_INDICES*)update_row_indices,
        (__gm__ DTYPE_LINEAR_CACHE_INDICES*)linear_cache_indices,
        totalLength,
        cumsumLength
    );
}