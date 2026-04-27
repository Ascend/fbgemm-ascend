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

#include "kernel_operator.h"
#include "pruned_hashmap_lookup.h"

extern "C" __global__ __aicore__ void pruned_hashmap_lookup(
    GM_ADDR indices,
    GM_ADDR offsets,
    GM_ADDR hash_table,
    GM_ADDR hash_table_offsets,
    GM_ADDR dense_indices,  // 输出tensor
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    Args args{
        indices, offsets, hash_table, hash_table_offsets,
        dense_indices, workspace, tiling
    };

    PrunedHashmapLookup::PrunedHashmapLookupKernel<DTYPE_INDICES> kernel(args);
    kernel.Compute(args);
}
