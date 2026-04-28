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
#ifndef JAGGED_TO_DENSE_V2
#define JAGGED_TO_DENSE_V2
#include "jagged_to_padded_dense_impl.h"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

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
    m.impl("jagged_to_padded_dense", TORCH_FN(jagged_to_padded_dense_impl_v2));
    m.impl("jagged_to_padded_dense.v2", TORCH_FN(jagged_to_padded_dense_impl_v2));
    m.impl("jagged_to_padded_dense_forward.v2", TORCH_FN(jagged_to_padded_dense_impl_v2));
}

#endif  // JAGGED_TO_DENSE_V2
