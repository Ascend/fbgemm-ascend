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

#ifndef FLOAT_OR_HALF_TO_FUSED8BITROWWISE_QUANTIZED_TILING_H
#define FLOAT_OR_HALF_TO_FUSED8BITROWWISE_QUANTIZED_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FloatOrHalfToFused8BitRowwiseQuantizedTilingData)
TILING_DATA_FIELD_DEF(int64_t, rows);
TILING_DATA_FIELD_DEF(int64_t, cols);
TILING_DATA_FIELD_DEF(int64_t, outputCols);

TILING_DATA_FIELD_DEF(int64_t, ncolsAligned);
TILING_DATA_FIELD_DEF(size_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, dtype);
TILING_DATA_FIELD_DEF(int32_t, threadsPerRow);
TILING_DATA_FIELD_DEF(int32_t, rowsPerBlock);
TILING_DATA_FIELD_DEF(int32_t, totalThreads);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FloatOrHalfToFused8BitRowwiseQuantized, FloatOrHalfToFused8BitRowwiseQuantizedTilingData)
}  // namespace optiling

#endif  // FLOAT_OR_HALF_TO_FUSED8BITROWWISE_QUANTIZED_TILING_H
