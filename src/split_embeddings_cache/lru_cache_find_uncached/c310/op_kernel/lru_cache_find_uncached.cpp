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

#include "lru_cache_find_uncached_kernel.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void lru_cache_find_uncached(
    GM_ADDR unique_indices,
    GM_ADDR unique_indices_length,
    GM_ADDR lxu_cache_state,
    GM_ADDR lru_state,
    GM_ADDR uvm_cache_stats,
    GM_ADDR lxu_cache_locking_counter,
    GM_ADDR cache_sets,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    LruCacheFindUncached::Args args{unique_indices,
        unique_indices_length,
        lxu_cache_state,
        lru_state,
        uvm_cache_stats,
        lxu_cache_locking_counter,
        cache_sets,
        workspace,
        tiling};
    LruCacheFindUncached::LruCacheFindUncachedKernel<DTYPE_UNIQUE> kernel(args);
    kernel.Compute();
}
