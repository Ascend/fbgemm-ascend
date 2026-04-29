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

#ifndef GROUP_INDEX_SELECT_DIM0_BACKWARD_TILING_H
#define GROUP_INDEX_SELECT_DIM0_BACKWARD_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

constexpr uint32_t MAX_GROUP_NUM = 32;

namespace optiling {
BEGIN_TILING_DATA_DEF(GroupIndexSelectDim0BackwardTilingData)
TILING_DATA_FIELD_DEF(int32_t, groupNum);
TILING_DATA_FIELD_DEF(int32_t, ubCanUsed);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_GROUP_NUM, groupGradRows);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_GROUP_NUM, groupIndicesLen);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_GROUP_NUM, groupInnerDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupIndexSelectDim0Backward, GroupIndexSelectDim0BackwardTilingData)
}  // namespace optiling
#endif
