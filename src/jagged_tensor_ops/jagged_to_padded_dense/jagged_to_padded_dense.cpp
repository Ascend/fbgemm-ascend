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
#include "jagged_to_padded_dense_impl.h"

// clang-format off
at::Tensor jagged_to_padded_dense_forward_v1(const at::Tensor& values,
                                             const tensor_list& offsets,
                                             const int64_t max_lengths,
                                             const double padding_value)
{
    TORCH_CHECK(offsets.size() == 1,
                "offsets must contain exactly 1 tensor, but got ", offsets.size(), " tensors");
    return jagged_to_padded_dense_impl_v1(values, offsets[0], max_lengths, padding_value);
}

at::Tensor jagged_to_padded_dense_backward_v1(const at::Tensor& dense,
                                              const tensor_list& offsets,
                                              const int64_t total_L)
{
    TORCH_CHECK(offsets.size() == 1,
                "offsets must contain exactly 1 tensor, but got ", offsets.size(), " tensors");
    return dense_to_jagged_impl(dense, offsets[0], total_L);
};

at::Tensor jagged_to_padded_dense_autograd_v1(const at::Tensor& values,
                                              const tensor_list& offsets,
                                              const int64_t max_lengths,
                                              const double padding_value)
{
    TORCH_CHECK(offsets.size() == 1,
                "offsets must contain exactly 1 tensor, but got ", offsets.size(), " tensors");
    return JaggedToPaddedDenseV1::apply(values, offsets[0], max_lengths, padding_value);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("jagged_to_padded_dense.v1(Tensor values, "
          "                          Tensor[] offsets, "
          "                          int max_lengths, "
          "                          float padding_value=0) -> Tensor");
    m.def("jagged_to_padded_dense_forward.v1(Tensor values, "
          "                                  Tensor[] offsets, "
          "                                  int max_lengths, "
          "                                  float padding_value=0) -> Tensor");
    m.def("jagged_to_padded_dense_backward.v1(Tensor dense, "
          "                                   Tensor[] offsets, "
          "                                   int total_L) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("jagged_to_padded_dense.v1", TORCH_FN(jagged_to_padded_dense_forward_v1));
    m.impl("jagged_to_padded_dense_forward.v1", TORCH_FN(jagged_to_padded_dense_forward_v1));
    m.impl("jagged_to_padded_dense_backward.v1", TORCH_FN(jagged_to_padded_dense_backward_v1));
}

TORCH_LIBRARY_IMPL(fbgemm, AutogradPrivateUse1, m)
{
    m.impl("jagged_to_padded_dense.v1", TORCH_FN(jagged_to_padded_dense_autograd_v1));
}
// clang-format on
