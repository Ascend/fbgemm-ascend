/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
#ifndef LRU_CACHE_INSERT_BYTE_TILING_H
#define LRU_CACHE_INSERT_BYTE_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LruCacheInsertByteTilingData)
TILING_DATA_FIELD_DEF(int64_t, bufferLength);
TILING_DATA_FIELD_DEF(int64_t, numCacheSets);
TILING_DATA_FIELD_DEF(int64_t, numWays);
TILING_DATA_FIELD_DEF(int64_t, cacheWeightsRowBytes);
TILING_DATA_FIELD_DEF(int64_t, weightsTotalLength);
TILING_DATA_FIELD_DEF(int64_t, uvmStatsLength);
TILING_DATA_FIELD_DEF(int64_t, gatherCacheStats);
TILING_DATA_FIELD_DEF(int64_t, timeStamp);
TILING_DATA_FIELD_DEF(int64_t, rowAlignment);
TILING_DATA_FIELD_DEF(int64_t, numTables);
TILING_DATA_FIELD_DEF(int64_t, hashCumsumLength);
TILING_DATA_FIELD_DEF(int64_t, cacheIndexMapLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LruCacheInsertByte, LruCacheInsertByteTilingData)

}  // namespace optiling

#endif  // LRU_CACHE_INSERT_BYTE_TILING_H
