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

#ifndef JAGGED_DENSE_DENSE_ELEMENTWISE_ADD_JAGGED_OUTPUT_TILING_H
#define JAGGED_DENSE_DENSE_ELEMENTWISE_ADD_JAGGED_OUTPUT_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
constexpr int32_t MAX_OFFSETS_CNT = 5;

BEGIN_TILING_DATA_DEF(JaggedDenseDenseElementwiseAddJaggedOutputTiling)

TILING_DATA_FIELD_DEF(int32_t, denseDim1);
TILING_DATA_FIELD_DEF(int32_t, denseDim2);
TILING_DATA_FIELD_DEF(int32_t, singleLoopSize);
TILING_DATA_FIELD_DEF(int32_t, denseType);
TILING_DATA_FIELD_DEF(int64_t, denseTotal);
TILING_DATA_FIELD_DEF(int64_t, jaggedTotal);
TILING_DATA_FIELD_DEF(int32_t, offsetCnt);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_OFFSETS_CNT, maxLengths);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_OFFSETS_CNT, offsetsLens);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(JaggedDenseDenseElementwiseAddJaggedOutput, JaggedDenseDenseElementwiseAddJaggedOutputTiling)
}  // namespace optiling
#endif
