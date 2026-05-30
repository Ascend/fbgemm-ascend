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

at::Tensor jagged_1d_to_dense_npu(at::Tensor values, at::Tensor offsets, c10::SymInt max_lengths,
                                  const int64_t padding_value)
{
    int64_t max_L = max_lengths.as_int_unchecked();
    TORCH_CHECK(max_L >= 0, "max_sequence_length must be non-negative, but got ", max_L);
    return jagged_to_padded_dense_impl_v1(values, offsets, max_L, padding_value);
}

class Jagged1DToDense : public torch::autograd::Function<Jagged1DToDense> {
public:
    static at::Tensor forward(AutogradContext* ctx, const at::Tensor& values, const at::Tensor& offsets,
                              const c10::SymInt max_sequence_length, const int64_t padding_value)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        ctx->save_for_backward({offsets});
        ctx->saved_data["total_L"] = values.size(0);
        return jagged_1d_to_dense_npu(values, offsets, max_sequence_length, padding_value);
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
    {
        auto grad_output = grad_outputs[0];
        auto offsets = ctx->get_saved_variables();
        auto total_L = ctx->saved_data["total_L"].toInt();
        // 这里需要把(B, max_1)，转换成(B,max_l,1)来复用dense_to_jagged_impl
        auto grad_input = dense_to_jagged_impl(grad_output.unsqueeze(-1), offsets[0], total_L);
        return {grad_input.squeeze(-1), Variable(), Variable(), Variable()};
    }
};

at::Tensor jagged_1d_to_dense_autograd_npu(at::Tensor values, at::Tensor offsets, c10::SymInt max_sequence_length,
                                           const int64_t padding_value)
{
    return Jagged1DToDense::apply(values, offsets, max_sequence_length, padding_value);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("jagged_1d_to_dense", TORCH_FN(jagged_1d_to_dense_npu));
}

TORCH_LIBRARY_IMPL(fbgemm, AutogradPrivateUse1, m)
{
    m.impl("jagged_1d_to_dense", TORCH_FN(jagged_1d_to_dense_autograd_npu));
}
