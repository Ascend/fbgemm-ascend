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

#ifndef FLOAT_TO_HFP8_QUANTIZED_TILING_H
#define FLOAT_TO_HFP8_QUANTIZED_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling
{
BEGIN_TILING_DATA_DEF(FloatToHfp8QuantizedTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalElems);
TILING_DATA_FIELD_DEF(int64_t, ebits);
TILING_DATA_FIELD_DEF(int64_t, exponent_bias);
TILING_DATA_FIELD_DEF(float, max_pos);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FloatToHfp8Quantized, FloatToHfp8QuantizedTilingData)
}  // namespace optiling

#endif  // FLOAT_TO_HFP8_QUANTIZED_TILING_H
