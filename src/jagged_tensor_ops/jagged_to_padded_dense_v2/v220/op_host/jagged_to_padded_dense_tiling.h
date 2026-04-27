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

#ifndef JAGGED_TO_PADDED_DENSE_V2_TILING
#define JAGGED_TO_PADDED_DENSE_V2_TILING

#include "register/tilingdata_base.h"
#include "constant.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(JaggedToPaddedDenseV2TilingData)

    TILING_DATA_FIELD_DEF(int64_t, total);
    TILING_DATA_FIELD_DEF(int64_t, innerDenseSize);
    TILING_DATA_FIELD_DEF(int64_t, outerDenseSize);

    TILING_DATA_FIELD_DEF(int64_t, offsetCnt);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_OFFSETS_CNT, maxLengths);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_OFFSETS_CNT, offsetsLens);
    TILING_DATA_FIELD_DEF(float, paddingValue);

    TILING_DATA_FIELD_DEF(int64_t, ubCanUsed);

    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(JaggedToPaddedDenseV2, JaggedToPaddedDenseV2TilingData)
}  // namespace optiling
#endif