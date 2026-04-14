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

#ifndef RUN_LENGTH_ENCODE_CONSTANT_H
#define RUN_LENGTH_ENCODE_CONSTANT_H

// Shape 元信息编码中“1 维”字段的固定值。
constexpr int64_t UINT64_SHAPE_DIM_ONE = 0x80000001;
constexpr int32_t SHAPE_LEN = 27;

// 通用偏移常量。
constexpr uint64_t START_POSITION = 0;
constexpr uint64_t DOUBLE_OFFSET = 2;

// 三个输出张量的 shape 描述在 shape buffer 中的关键索引。
constexpr int32_t SHAPE0_SIZE_IDX = 0;
constexpr int32_t SHAPE0_DIM0_IDX = 1;
constexpr int32_t SHAPE1_SIZE_IDX = 9;
constexpr int32_t SHAPE1_DIM0_IDX = 10;
constexpr int32_t SHAPE2_SIZE_IDX = 18;
constexpr int32_t SHAPE2_DIM0_IDX = 19;

#endif  // RUN_LENGTH_ENCODE_CONSTANT_H
