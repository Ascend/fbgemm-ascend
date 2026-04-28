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

#ifndef LRU_CACHE_FIND_UNCACHED_TILING_H
#define LRU_CACHE_FIND_UNCACHED_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LruCacheFindUncachedTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalLength);
TILING_DATA_FIELD_DEF(int64_t, numCacheSets);
TILING_DATA_FIELD_DEF(int64_t, numWays);
TILING_DATA_FIELD_DEF(int64_t, uvmStatsLength);
TILING_DATA_FIELD_DEF(int64_t, lockCounterLength);
TILING_DATA_FIELD_DEF(int64_t, gatherCacheStats);
TILING_DATA_FIELD_DEF(int64_t, maxIndices);
TILING_DATA_FIELD_DEF(int64_t, timeStamp);
TILING_DATA_FIELD_DEF(int64_t, lockCacheLine);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LruCacheFindUncached, LruCacheFindUncachedTilingData)

}  // namespace optiling

#endif // LRU_CACHE_FIND_UNCACHED_TILING_H
