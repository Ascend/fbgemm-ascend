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

#include "kernel_operator.h"
#include "simt_kernel.h"

using namespace AscendC;
using namespace emb_inplace_update_kernel;

extern "C" __global__ __aicore__ void emb_inplace_update(GM_ADDR dev_weights, GM_ADDR uvm_weights,
                                                         GM_ADDR weights_placements, GM_ADDR weights_offsets,
                                                         GM_ADDR weights_tys, GM_ADDR D_offsets, GM_ADDR update_weights,
                                                         GM_ADDR update_table_indices, GM_ADDR update_row_indices,
                                                         GM_ADDR update_offsets, GM_ADDR workspace, GM_ADDR tiling_data)
{
    GET_TILING_DATA(tiling_data_struct, tiling_data);
    int64_t N = tiling_data_struct.totalUpdates;
    int32_t rowAlignment = static_cast<int32_t>(tiling_data_struct.rowAlignment);
    int32_t threadNumPerBlock = static_cast<int32_t>(tiling_data_struct.threadNumPerBlock);
    int32_t rowIdxIsInt64 = static_cast<int32_t>(tiling_data_struct.rowIdxIsInt64);

    AscendC::Simt::Dim3 dim(threadNumPerBlock, 1, 1);

    if (rowIdxIsInt64 != 0) {
        AscendC::Simt::VF_CALL<SimtEmbInplaceUpdateMultiThread<int64_t>>(
            dim, (__gm__ uint8_t*)dev_weights, (__gm__ uint8_t*)uvm_weights, (__gm__ int32_t*)weights_placements,
            (__gm__ int64_t*)weights_offsets, (__gm__ uint8_t*)weights_tys, (__gm__ int32_t*)D_offsets,
            (__gm__ uint8_t*)update_weights, (__gm__ int32_t*)update_table_indices, (__gm__ int64_t*)update_row_indices,
            (__gm__ int64_t*)update_offsets, N, rowAlignment);
    } else {
        AscendC::Simt::VF_CALL<SimtEmbInplaceUpdateMultiThread<int32_t>>(
            dim, (__gm__ uint8_t*)dev_weights, (__gm__ uint8_t*)uvm_weights, (__gm__ int32_t*)weights_placements,
            (__gm__ int64_t*)weights_offsets, (__gm__ uint8_t*)weights_tys, (__gm__ int32_t*)D_offsets,
            (__gm__ uint8_t*)update_weights, (__gm__ int32_t*)update_table_indices, (__gm__ int32_t*)update_row_indices,
            (__gm__ int64_t*)update_offsets, N, rowAlignment);
    }
}
