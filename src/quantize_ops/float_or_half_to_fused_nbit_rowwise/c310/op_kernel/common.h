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

#ifndef FLOAT_OR_HALF_TO_FUSED_NBIT_ROWWISE_COMMON_H
#define FLOAT_OR_HALF_TO_FUSED_NBIT_ROWWISE_COMMON_H

#include "kernel_common_utils.h"
#include <cfloat>

namespace FloatOrHalfToFusedNbitRowwiseCommon {

using namespace AscendC;

constexpr int32_t KERNEL_MODE_SIMD = 1;
constexpr int32_t SCALE_BIAS_BYTES = 4;  // sizeof(half) * 2: scale(fp16) + bias(fp16)

__aicore__ inline bool IsSimdMode(GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    return tilingData.kernelMode == KERNEL_MODE_SIMD;
}

struct Args {
    GM_ADDR input;
    GM_ADDR output;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

}  // namespace FloatOrHalfToFusedNbitRowwiseCommon

#endif
