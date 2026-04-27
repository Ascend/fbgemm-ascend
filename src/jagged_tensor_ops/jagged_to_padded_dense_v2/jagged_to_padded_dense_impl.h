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
#ifndef JAGGED_TO_PADDED_DENSE_IMPL_H
#define JAGGED_TO_PADDED_DENSE_IMPL_H
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../../common/pytorch_npu_helper.hpp"
#include "../../common/common_utils.h"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

constexpr int EXPECTED_DIM_1D = 1;
constexpr int EXPECTED_DIM_2D = 2;
constexpr int MAX_OFFSETS_CNT = 5;

at::Tensor jagged_to_padded_dense_impl_v1(const at::Tensor& values,
                                          const at::Tensor& offsets,
                                          const int64_t max_lengths,
                                          const double padding_value);

at::Tensor jagged_to_padded_dense_impl_v2(const at::Tensor& values,
                                          const tensor_list& offsets,
                                          const at::IntArrayRef& max_lengths,
                                          const double padding_value);

at::Tensor dense_to_jagged_impl(const at::Tensor& dense,
                                const at::Tensor& offsets,
                                const c10::optional<int64_t>& total_L);

#endif  // JAGGED_TO_PADDED_DENSE_IMPL_H
