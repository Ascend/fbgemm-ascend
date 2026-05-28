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
#ifndef DIRECT_MAPPED_LRU_CACHE_FIND_UNCACHED_TILING_H
#define DIRECT_MAPPED_LRU_CACHE_FIND_UNCACHED_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(DirectMappedLruCacheFindUncachedTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalLength);       // 索引总数 N
TILING_DATA_FIELD_DEF(int64_t, numCacheSets);      // Cache set 数量 C
TILING_DATA_FIELD_DEF(int64_t, uvmStatsLength);    // UVM 统计张量长度
TILING_DATA_FIELD_DEF(int64_t, gatherCacheStats);  // 是否收集统计信息
TILING_DATA_FIELD_DEF(int64_t, maxIndices);        // 哨兵值，标记无效/被删索引
TILING_DATA_FIELD_DEF(int64_t, timeStamp);         // 当前 LRU 时间戳
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DirectMappedLruCacheFindUncached, DirectMappedLruCacheFindUncachedTilingData)

}  // namespace optiling

#endif  // DIRECT_MAPPED_LRU_CACHE_FIND_UNCACHED_TILING_H
