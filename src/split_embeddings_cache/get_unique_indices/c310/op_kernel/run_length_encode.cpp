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

#include "run_length_encode_single_core_kernel.h"
#include "run_length_encode_multi_core_kernel.h"
#include <type_traits>

#if !defined(DTYPE_SORTED_INDICES)
#error "DTYPE_SORTED_INDICES is undefined. Expect input sorted_indices dtype macro from run_length_encode.json."
#endif

template <template <typename, bool> class KERNEL_CLASS, typename INPUT_TYPE, bool COUNT_OUT>
__aicore__ inline void createAndRunKernel(GM_ADDR x, GM_ADDR y, GM_ADDR count, GM_ADDR length, GM_ADDR shape_out,
                                          GM_ADDR workspace, const RunLengthEncodeTilingData* tilingData, TPipe* pipe)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    static_assert(std::is_same_v<INPUT_TYPE, int32_t> || std::is_same_v<INPUT_TYPE, int64_t>,
                  "run_length_encode(c310) only supports int32/int64 sorted_indices.");
    KERNEL_CLASS<INPUT_TYPE, COUNT_OUT> op(pipe);
    op.Init(x, y, count, length, shape_out, workspace, tilingData);
    op.Process();
}

__aicore__ inline void CopyOutEmptyShapeAndLength(GM_ADDR shape_out, GM_ADDR length, AscendC::TPipe* pipe)
{
    GlobalTensor<uint64_t> shapeGm;
    GlobalTensor<int32_t> lengthGm;
    TBuf<TPosition::VECCALC> shapeBuf;
    TBuf<TPosition::VECCALC> lengthBuf;

    shapeGm.SetGlobalBuffer((__gm__ uint64_t*)shape_out);
    lengthGm.SetGlobalBuffer((__gm__ int32_t*)length);
    pipe->InitBuffer(shapeBuf, SHAPE_LEN * sizeof(uint64_t));
    pipe->InitBuffer(lengthBuf, 32);

    LocalTensor<uint64_t> shapeTensor = shapeBuf.Get<uint64_t>();
    LocalTensor<int32_t> lengthTensor = lengthBuf.Get<int32_t>();
    Duplicate(shapeTensor, (uint64_t)1, SHAPE_LEN);
    lengthTensor.SetValue(0, 0);
    SimpleNativePipeSync<HardEvent::V_S>();

    shapeTensor.SetValue(SHAPE0_SIZE_IDX, UINT64_SHAPE_DIM_ONE);
    shapeTensor.SetValue(SHAPE0_DIM0_IDX, 0);

    shapeTensor.SetValue(SHAPE1_SIZE_IDX, UINT64_SHAPE_DIM_ONE);
    shapeTensor.SetValue(SHAPE1_DIM0_IDX, 0);

    shapeTensor.SetValue(SHAPE2_SIZE_IDX, 1);
    shapeTensor.SetValue(SHAPE2_DIM0_IDX, 1);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = SHAPE_LEN * sizeof(uint64_t);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    SimpleNativePipeSync<HardEvent::S_MTE3>();
    DataCopyPad(shapeGm, shapeTensor, dataCopyParams);
    CpLocal2Gm<int32_t>(lengthGm, lengthTensor, 1);
}

extern "C" __global__ __aicore__ void run_length_encode(GM_ADDR sorted_indices, GM_ADDR unique_indices,
                                                        GM_ADDR unique_indices_count, GM_ADDR unique_indices_length,
                                                        GM_ADDR shape_out, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(RunLengthEncodeTilingData, tilingDataIn, tiling);
    const RunLengthEncodeTilingData* __restrict tilingData = &tilingDataIn;

    // tiling key 分发：
    // 11/21 -> 输出 count；10/20 -> 不输出 count；666 -> 空输入快速回写。
    if (TILING_KEY_IS(11)) {
        createAndRunKernel<RunLengthEncodeSingleCoreKernel, DTYPE_SORTED_INDICES, true>(
            sorted_indices, unique_indices, unique_indices_count, unique_indices_length, shape_out, usrWorkspace,
            tilingData, &pipe);
    } else if (TILING_KEY_IS(21)) {
        createAndRunKernel<RunLengthEncodeMultiCoreKernel, DTYPE_SORTED_INDICES, true>(
            sorted_indices, unique_indices, unique_indices_count, unique_indices_length, shape_out, usrWorkspace,
            tilingData, &pipe);
    } else if (TILING_KEY_IS(10)) {
        createAndRunKernel<RunLengthEncodeSingleCoreKernel, DTYPE_SORTED_INDICES, false>(
            sorted_indices, unique_indices, unique_indices_count, unique_indices_length, shape_out, usrWorkspace,
            tilingData, &pipe);
    } else if (TILING_KEY_IS(20)) {
        createAndRunKernel<RunLengthEncodeMultiCoreKernel, DTYPE_SORTED_INDICES, false>(
            sorted_indices, unique_indices, unique_indices_count, unique_indices_length, shape_out, usrWorkspace,
            tilingData, &pipe);
    } else if (TILING_KEY_IS(666)) {
        CopyOutEmptyShapeAndLength(shape_out, unique_indices_length, &pipe);
    }
}
