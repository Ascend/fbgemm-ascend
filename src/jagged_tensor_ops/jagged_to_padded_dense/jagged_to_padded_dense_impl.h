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

#ifndef FBGEMM_ASCEND_JAGGED_TO_PADDED_DENSE_IMPL_H
#define FBGEMM_ASCEND_JAGGED_TO_PADDED_DENSE_IMPL_H
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../../common/pytorch_npu_helper.hpp"
#include "../../common/common_utils.h"
#include "fbgemm_ascend/jagged_tensor_ops.h"

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using tensor_list = std::vector<at::Tensor>;
using namespace at;
using fbgemm_npu::EXPECTED_DIM_1D;
using fbgemm_npu::EXPECTED_DIM_2D;
using fbgemm_npu::EXPECTED_DIM_3D;

class JaggedToPaddedDenseV1 : public torch::autograd::Function<JaggedToPaddedDenseV1> {
public:
    static at::Tensor forward(AutogradContext* ctx, const at::Tensor& values, const at::Tensor& offsets,
                              const int64_t max_lengths, const double padding_value);

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs);
};
#endif  // FBGEMM_ASCEND_JAGGED_TO_PADDED_DENSE_IMPL_H
