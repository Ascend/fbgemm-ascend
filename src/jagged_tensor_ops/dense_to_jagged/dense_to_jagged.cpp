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
#include "dense_to_jagged_impl.h"

std::tuple<at::Tensor, tensor_list> dense_to_jagged_npu(const at::Tensor& dense, const tensor_list& offsets,
                                                        const c10::optional<int64_t> total_L)
{
    auto jagged = dense_to_jagged_impl(dense, offsets[0], total_L);
    return {jagged, offsets};
};

std::tuple<at::Tensor, tensor_list> dense_to_jagged_autograd_npu(const at::Tensor& dense, const tensor_list& offsets,
                                                                 const c10::optional<int64_t> total_L)
{
    auto jagged = DenseToJaggedFunction::apply(dense, offsets, total_L);
    return {jagged, offsets};
};

at::Tensor dense_to_jagged_forward_npu(const at::Tensor& dense, const tensor_list& offsets,
                                       const c10::optional<int64_t> total_L)
{
    return dense_to_jagged_impl(dense, offsets[0], total_L);
};

at::Tensor dense_to_jagged_backward_npu(const at::Tensor& values, const tensor_list& offsets, const int64_t max_lengths,
                                        const double padding_value)
{
    return jagged_to_padded_dense_impl_v1(values, offsets[0], max_lengths, padding_value);
};

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("dense_to_jagged", &dense_to_jagged_npu);
    m.impl("dense_to_jagged_forward", &dense_to_jagged_forward_npu);
    m.impl("dense_to_jagged_backward", &dense_to_jagged_backward_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, AutogradPrivateUse1, m)
{
    m.impl("dense_to_jagged", &dense_to_jagged_autograd_npu);
}
