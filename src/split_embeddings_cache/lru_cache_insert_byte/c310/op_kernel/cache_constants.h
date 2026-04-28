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

#ifndef LRU_CACHE_INSERT_BYTE_FBGEMM_CACHE_CONSTANTS_H
#define LRU_CACHE_INSERT_BYTE_FBGEMM_CACHE_CONSTANTS_H

#include <cstdint>

namespace fbgemm_compat {

// common.cuh
constexpr int64_t kCacheStateInvalid = -1;

// cuda_prelude.cuh: int8 row stores float2 qparams (8 bytes) after quantized values
constexpr int32_t kINT8QparamsBytes = 8;

// split_embeddings_cache_cuda.cuh — uvm_cache_stats_index
struct UvmCacheStatsIndex {
    static constexpr int32_t num_calls = 0;
    static constexpr int32_t num_requested_indices = 1;
    static constexpr int32_t num_unique_indices = 2;
    static constexpr int32_t num_unique_misses = 3;
    static constexpr int32_t num_conflict_unique_misses = 4;
};

}  // namespace fbgemm_compat

#endif
