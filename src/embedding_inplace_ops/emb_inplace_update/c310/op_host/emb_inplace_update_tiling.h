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

#ifndef EMB_INPLACE_UPDATE_TILING_H
#define EMB_INPLACE_UPDATE_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(EmbInplaceUpdateTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalUpdates);
TILING_DATA_FIELD_DEF(int64_t, numTables);
TILING_DATA_FIELD_DEF(int64_t, rowAlignment);
TILING_DATA_FIELD_DEF(int64_t, rowIdxIsInt64);
TILING_DATA_FIELD_DEF(int32_t, threadNumPerBlock);
TILING_DATA_FIELD_DEF(int32_t, gridDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(EmbInplaceUpdate, EmbInplaceUpdateTilingData)
} // namespace optiling
#endif // EMB_INPLACE_UPDATE_TILING_H
