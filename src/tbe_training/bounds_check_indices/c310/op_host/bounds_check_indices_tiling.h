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

#ifndef BOUNDS_CHECK_INDICES_TILING_H
#define BOUNDS_CHECK_INDICES_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(BoundsCheckIndicesTilingData)
    TILING_DATA_FIELD_DEF(int64_t, numIndices);
    TILING_DATA_FIELD_DEF(int32_t, numTables);
    TILING_DATA_FIELD_DEF(int32_t, batchSize);
    TILING_DATA_FIELD_DEF(int32_t, totalB);
    TILING_DATA_FIELD_DEF(int32_t, boundsCheckMode);
    TILING_DATA_FIELD_DEF(int32_t, vbe);
    TILING_DATA_FIELD_DEF(int32_t, infoBNumBits);
    TILING_DATA_FIELD_DEF(uint32_t, infoBMask);
    TILING_DATA_FIELD_DEF(uint32_t, batchSizeDivMagic);
    TILING_DATA_FIELD_DEF(uint32_t, batchSizeDivShift);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BoundsCheckIndicesV1, BoundsCheckIndicesTilingData)
REGISTER_TILING_DATA_CLASS(BoundsCheckIndicesV2, BoundsCheckIndicesTilingData)

}  // namespace optiling

#endif  // BOUNDS_CHECK_INDICES_TILING_H