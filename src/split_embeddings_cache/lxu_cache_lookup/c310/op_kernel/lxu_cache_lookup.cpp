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

#include "lxu_cache_lookup_kernel.h"

extern "C" __global__ __aicore__ void lxu_cache_lookup(GM_ADDR linear_cache_indices, GM_ADDR lxu_cache_state,
                                                       GM_ADDR uvm_cache_stats, GM_ADDR num_uniq_cache_indices,
                                                       GM_ADDR lxu_cache_locations, GM_ADDR workspace, GM_ADDR tiling)
{
    Args args{linear_cache_indices, lxu_cache_state, uvm_cache_stats, num_uniq_cache_indices,
              lxu_cache_locations,  workspace,       tiling};
    LxuCacheLookup::LxuCacheLookupKernel<DTYPE_LINEAR_CACHE_INDICES> kernel(args);
    kernel.Compute();
}
