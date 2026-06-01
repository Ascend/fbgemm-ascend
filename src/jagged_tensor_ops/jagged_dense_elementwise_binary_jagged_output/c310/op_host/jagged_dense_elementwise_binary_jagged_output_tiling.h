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

#ifndef JAGGED_DENSE_ELEMENTWISE_BINARY_JAGGED_OUTPUT_TILING_H
#define JAGGED_DENSE_ELEMENTWISE_BINARY_JAGGED_OUTPUT_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
constexpr int32_t MAX_OFFSETS_CNT = 5;

// Host侧写入、Kernel侧读取的tiling结构。
BEGIN_TILING_DATA_DEF(JaggedDenseElementwiseBinaryJaggedOutputTiling)

// 所有dense tensor中间维度乘积。
TILING_DATA_FIELD_DEF(int32_t, denseDim1);
// dense tensor和jagged tensor的尾维，也就是D。
TILING_DATA_FIELD_DEF(int32_t, denseDim2);
// 单次GM/UB搬运计算使用的UB buffer字节数。
TILING_DATA_FIELD_DEF(int32_t, singleLoopSize);
// dense/x/out的数据类型，当前kernel主要通过模板类型处理。
TILING_DATA_FIELD_DEF(int32_t, denseType);
// offsets的数据类型，当前kernel主要通过模板类型处理。
TILING_DATA_FIELD_DEF(int32_t, offsetType);
// dense所有维度乘积。
TILING_DATA_FIELD_DEF(int64_t, denseTotal);
// jagged所有维度乘积，total_L * D。
TILING_DATA_FIELD_DEF(int64_t, jaggedTotal);
// 计算模式：0表示add，1表示mul。
TILING_DATA_FIELD_DEF(int32_t, elementwiseMode);
// jagged维度，也就是offset_list长度。
TILING_DATA_FIELD_DEF(int32_t, offsetCnt);
// 每个jagged维度在dense中的padded长度。
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_OFFSETS_CNT, maxLengths);
// 每个offset tensor的元素个数。
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_OFFSETS_CNT, offsetsLens);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(JaggedDenseElementwiseBinaryJaggedOutput, JaggedDenseElementwiseBinaryJaggedOutputTiling)
}  // namespace optiling
#endif
