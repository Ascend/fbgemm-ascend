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

#ifndef FUSED_8BIT_ROWWISE_QUANTIZED_TO_FLOAT_OR_HALF_TILING_H
#define FUSED_8BIT_ROWWISE_QUANTIZED_TO_FLOAT_OR_HALF_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Fused8BitRowwiseQuantizedToFloatOrHalfTilingData)
TILING_DATA_FIELD_DEF(int64_t, rows);
TILING_DATA_FIELD_DEF(int64_t, cols);
TILING_DATA_FIELD_DEF(int64_t, outputCols);
TILING_DATA_FIELD_DEF(size_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, dtype);
TILING_DATA_FIELD_DEF(bool, scaleBiasLast);
TILING_DATA_FIELD_DEF(bool, quantPaddingFloatType);
TILING_DATA_FIELD_DEF(int64_t, quantPaddingSize);
TILING_DATA_FIELD_DEF(int32_t, threadsPerRow);
TILING_DATA_FIELD_DEF(int32_t, threadsPerRowLog2);
TILING_DATA_FIELD_DEF(int32_t, rowsPerBlock);
TILING_DATA_FIELD_DEF(int32_t, totalThreads);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Fused8BitRowwiseQuantizedToFloatOrHalf, Fused8BitRowwiseQuantizedToFloatOrHalfTilingData)
}  // namespace optiling

#endif  // FUSED_8BIT_ROWWISE_QUANTIZED_TO_FLOAT_OR_HALF_TILING_H
