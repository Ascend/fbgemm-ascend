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
#include "jagged_to_padded_dense_impl_v2.h"

at::Tensor jagged_to_padded_dense_npu(const at::Tensor& values, const tensor_list& offsets,
                                      const at::IntArrayRef& max_lengths, const double padding_value)
{
    return jagged_to_padded_dense_impl_v2(values, offsets, max_lengths, padding_value);
}

at::Tensor jagged_to_padded_dense_forward(const at::Tensor& values, const tensor_list& offsets,
                                          at::ArrayRef<at::SymInt> max_lengths, const double padding_value)
{
    std::vector<int64_t> max_lengths_int64;
    max_lengths_int64.reserve(max_lengths.size());
    for (const auto& len : max_lengths) {
        max_lengths_int64.push_back(len.as_int_unchecked());
    }
    return jagged_to_padded_dense_impl_v2(values, offsets, max_lengths_int64, padding_value);
}

at::Tensor jagged_to_padded_dense_backward(const at::Tensor& dense, const tensor_list& offsets, const int64_t total_L)
{
    TORCH_CHECK(offsets.size() == 1, "offsets must contain exactly 1 tensor, but got ", offsets.size(), " tensors");
    return dense_to_jagged_impl(dense, offsets[0], total_L);
};

at::Tensor jagged_to_padded_dense_autograd_npu(const at::Tensor& values, const tensor_list& offsets,
                                               const at::IntArrayRef& max_lengths, const double padding_value)
{
    return JaggedToPaddedDenseV2::apply(values, offsets, max_lengths, padding_value);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("jagged_to_padded_dense.v2(Tensor values, "
          "                          Tensor[] offsets, "
          "                          int[] max_lengths, "
          "                          float padding_value=0) -> Tensor");

    m.def("jagged_to_padded_dense_forward.v2(Tensor values, "
          "                                  Tensor[] offsets, "
          "                                  int[] max_lengths, "
          "                                  float padding_value=0) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("jagged_to_padded_dense", TORCH_FN(jagged_to_padded_dense_npu));
    m.impl("jagged_to_padded_dense.v2", TORCH_FN(jagged_to_padded_dense_npu));
    m.impl("jagged_to_padded_dense_forward", TORCH_FN(jagged_to_padded_dense_forward));
    m.impl("jagged_to_padded_dense_forward.v2", TORCH_FN(jagged_to_padded_dense_npu));
    m.impl("jagged_to_padded_dense_backward", TORCH_FN(jagged_to_padded_dense_backward));
    m.impl("jagged_to_padded_dense_backward.v2", TORCH_FN(jagged_to_padded_dense_backward));
}

TORCH_LIBRARY_IMPL(fbgemm, AutogradPrivateUse1, m)
{
    m.impl("jagged_to_padded_dense", TORCH_FN(jagged_to_padded_dense_autograd_npu));
    m.impl("jagged_to_padded_dense.v2", TORCH_FN(jagged_to_padded_dense_autograd_npu));
}
