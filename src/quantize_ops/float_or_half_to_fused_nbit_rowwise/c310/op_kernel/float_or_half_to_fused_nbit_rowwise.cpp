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
#include "float_or_half_to_fused_nbit_rowwise_simd_kernel.h"
#include "float_or_half_to_fused_nbit_rowwise_simt_kernel.h"

extern "C" __global__ __aicore__ void float_or_half_to_fused_nbit_rowwise(GM_ADDR input, GM_ADDR output,
                                                                          GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (FloatOrHalfToFusedNbitRowwiseCommon::IsSimdMode(tiling)) {
        FloatOrHalfToFusedNbitRowwiseSimd::Args args{input, output, workspace, tiling};
        FloatOrHalfToFusedNbitRowwiseSimd::KernelSimd<DTYPE_INPUT> kernel(args);
        kernel.Process();
    } else {
        FloatOrHalfToFusedNbitRowwiseSimt::Args args{input, output, workspace, tiling};
        FloatOrHalfToFusedNbitRowwiseSimt::KernelSimt<DTYPE_INPUT> kernel(args);
        kernel.Process();
    }
}
