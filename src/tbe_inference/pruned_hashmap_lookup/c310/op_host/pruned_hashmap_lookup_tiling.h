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

#ifndef PRUNED_HASHMAP_LOOKUP_TILING
#define PRUNED_HASHMAP_LOOKUP_TILING
#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(PrunedHashmapLookupTilingData)

    TILING_DATA_FIELD_DEF(int64_t, batchNum); // indices中所有的batch数
    TILING_DATA_FIELD_DEF(int64_t, batchPerTable); // 每个表的batch数
    TILING_DATA_FIELD_DEF(int64_t, tableNum);
    TILING_DATA_FIELD_DEF(int64_t, bigCore);
    TILING_DATA_FIELD_DEF(int64_t, batchNumPerCore);
    TILING_DATA_FIELD_DEF(int64_t, indicesLen);
    TILING_DATA_FIELD_DEF(int64_t, offsetsLen);
    TILING_DATA_FIELD_DEF(int64_t, hashTableLen);
    TILING_DATA_FIELD_DEF(int64_t, hashTableOffsetsLen);

    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(PrunedHashmapLookup, PrunedHashmapLookupTilingData)
}  // namespace optiling
#endif
