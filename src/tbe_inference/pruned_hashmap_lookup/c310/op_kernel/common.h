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

#ifndef PRUNED_HASHMAP_LOOKUP_COMMON_H
#define PRUNED_HASHMAP_LOOKUP_COMMON_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

// 常量定义
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int SIMT_LAUNCH_BOUND = 32 * 8;
constexpr int THRESHOLD = 512;


// Args结构体
struct Args {
    GM_ADDR indices;
    GM_ADDR offsets;
    GM_ADDR hashTable;
    GM_ADDR hashTableOffsets;
    GM_ADDR denseIndices;  // 输出tensor
    GM_ADDR workspace;
    GM_ADDR tiling;
};

#endif // PRUNED_HASHMAP_LOOKUP_COMMON_H

