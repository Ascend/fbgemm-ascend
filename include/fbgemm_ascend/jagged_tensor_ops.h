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

#pragma once

#include <ATen/Tensor.h>
#include <ATen/core/DimVector.h>
#include <c10/core/SymInt.h>
#include <c10/util/Optional.h>

#include <tuple>
#include <vector>
using tensor_list = std::vector<at::Tensor>;

// ----- Select / Permute 算子 -----
std::vector<at::Tensor> keyed_jagged_index_select_dim1_impl_npu(const at::Tensor& values, const at::Tensor& lengths,
                                                                const at::Tensor& offsets, const at::Tensor& indices,
                                                                const int64_t& batch_size,
                                                                const c10::optional<at::Tensor>& weights,
                                                                const c10::optional<int64_t>& selected_lengths_sum);

// ----- Jagged ↔ Dense 互转核心算子 -----
std::tuple<at::Tensor, tensor_list> dense_to_jagged_npu(const at::Tensor& dense, const tensor_list& offsets,
                                                        const c10::optional<int64_t> total_L);

std::tuple<at::Tensor, tensor_list> dense_to_jagged_autograd_npu(const at::Tensor& dense, const tensor_list& offsets,
                                                                 const c10::optional<int64_t> total_L);

at::Tensor jagged_to_padded_dense_npu(const at::Tensor& values, const tensor_list& offsets,
                                      const at::IntArrayRef& max_lengths, const double padding_value);

at::Tensor jagged_to_padded_dense_autograd_npu(const at::Tensor& values, const tensor_list& offsets,
                                               const at::IntArrayRef& max_lengths, const double padding_value);

at::Tensor jagged_1d_to_dense_npu(at::Tensor values, at::Tensor offsets, c10::SymInt max_lengths,
                                  const int64_t padding_value);

at::Tensor jagged_2d_to_dense_npu(const at::Tensor& values, const at::Tensor& offsets, const int64_t max_lengths);

at::Tensor jagged_to_padded_dense_impl_v1(const at::Tensor& values, const at::Tensor& offsets,
                                          const int64_t max_lengths, const double padding_value);

at::Tensor jagged_to_padded_dense_impl_v2(const at::Tensor& values, const std::vector<at::Tensor>& offsets,
                                          const at::IntArrayRef& max_lengths, double padding_value);

at::Tensor dense_to_jagged_impl(const at::Tensor& dense, const at::Tensor& offsets,
                                const c10::optional<int64_t>& total_L);

// ----- fbgemm_npu 内部实现（多版本函数、Autograd 包装） -----
namespace fbgemm_npu {
constexpr int EXPECTED_DIM_1D = 1;
constexpr int EXPECTED_DIM_2D = 2;
constexpr int EXPECTED_DIM_3D = 3;
constexpr int MAX_OFFSETS_CNT = 5;
}  // namespace fbgemm_npu
