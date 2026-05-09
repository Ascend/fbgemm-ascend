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

extern "C" __global__ __aicore__ void bounds_check_indices_v2(
    GM_ADDR rowsPerTable,
    GM_ADDR indices,
    GM_ADDR offsets,
    GM_ADDR warning,
    GM_ADDR bOffsets,
    GM_ADDR bTMap,
    GM_ADDR indicesOut,
    GM_ADDR offsetsOut,
    GM_ADDR warningOut,
    GM_ADDR workspace,
    GM_ADDR tiling) {
}