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
#ifndef JAGGED_CONSTANT_H
#define JAGGED_CONSTANT_H

constexpr int GM_ALIGN = 64;
constexpr int UB_ALIGN = 32;
constexpr int RESERVER_UB_SIZE = 20 * 1024;
constexpr int MIN_UB_USED_SIZE = 12 * 1024;
constexpr int DATA_TYPE_INT64 = 8;
constexpr int DATA_TYPE_INT32 = 4;
constexpr int DATA_TYPE_FLOAT32 = 4;
constexpr int NUM_QUEUE = 4;

constexpr int SUPPORT_EMBEDDING_DIM_NUM = 2;
constexpr int MAX_D = 2048;
constexpr int MIN_OFFSETS_CNT = 1;
constexpr int MAX_OFFSETS_CNT = 5;

#endif  // JAGGED_CONSTANT_H
