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

#ifndef FLOAT_OR_HALF_TO_FUSED_NBIT_ROWWISE_TILING_H
#define FLOAT_OR_HALF_TO_FUSED_NBIT_ROWWISE_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FloatOrHalfToFusedNbitRowwiseTilingData)
TILING_DATA_FIELD_DEF(int64_t, nrows);
TILING_DATA_FIELD_DEF(int64_t, ncols);
TILING_DATA_FIELD_DEF(int32_t, bitRate);
TILING_DATA_FIELD_DEF(int32_t, kernelMode);
TILING_DATA_FIELD_DEF(int64_t, outputColumns);
TILING_DATA_FIELD_DEF(int32_t, numElemPerByte);

// SIMT fields
TILING_DATA_FIELD_DEF(int32_t, bufferNum);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCycle);

// SIMD fields
TILING_DATA_FIELD_DEF(int64_t, blockLen);
TILING_DATA_FIELD_DEF(int64_t, splitBaseLen);
TILING_DATA_FIELD_DEF(int32_t, tailSplitIndex);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FloatOrHalfToFusedNbitRowwise, FloatOrHalfToFusedNbitRowwiseTilingData)
}  // namespace optiling

#endif
