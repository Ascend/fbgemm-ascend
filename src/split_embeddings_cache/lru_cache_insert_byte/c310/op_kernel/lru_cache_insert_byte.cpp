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

#include "lru_cache_insert_byte_kernel.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void lru_cache_insert_byte(
    GM_ADDR weights,
    GM_ADDR cache_hash_size_cumsum,
    GM_ADDR cache_index_table_map,
    GM_ADDR weights_offsets,
    GM_ADDR weights_tys,
    GM_ADDR d_offsets,
    GM_ADDR sorted_cache_sets,
    GM_ADDR cache_set_sorted_unique_indices,
    GM_ADDR unique_indices_length,
    GM_ADDR lxu_cache_state,
    GM_ADDR lxu_cache_weights,
    GM_ADDR lru_state,
    GM_ADDR uvm_cache_stats,
    GM_ADDR reserved_out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    LruCacheInsertByte::Args args{weights,
        cache_hash_size_cumsum,
        cache_index_table_map,
        weights_offsets,
        weights_tys,
        d_offsets,
        sorted_cache_sets,
        cache_set_sorted_unique_indices,
        unique_indices_length,
        lxu_cache_state,
        lxu_cache_weights,
        lru_state,
        uvm_cache_stats,
        reserved_out,
        workspace,
        tiling};
    LruCacheInsertByte::LruCacheInsertByteKernel<DTYPE_CACHE_SET_SORTED_UNIQUE_INDICES> kernel(args);
    kernel.Compute();
}
