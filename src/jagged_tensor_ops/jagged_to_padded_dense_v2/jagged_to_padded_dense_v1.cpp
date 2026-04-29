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
#ifndef JAGGED_TO_DENSE_V1
#define JAGGED_TO_DENSE_V1
#include "jagged_to_padded_dense_impl.h"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

at::Tensor jagged_to_padded_dense_forward(const at::Tensor& values,
                                             const tensor_list& offsets,
                                             at::ArrayRef<at::SymInt> max_lengths,
                                             const double padding_value)
{
    int64_t max_L = max_lengths[0].as_int_unchecked();
    return jagged_to_padded_dense_impl_v1(values, offsets[0], max_L, padding_value);
}

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

at::Tensor jagged_2d_to_dense(const at::Tensor& values,
                              const at::Tensor& offsets,
                              const int64_t max_lengths)
{
    return jagged_to_padded_dense_impl_v1(values, offsets, max_lengths, .0);
}

at::Tensor jagged_1d_to_dense(at::Tensor values,
                                  at::Tensor offsets,
                                  at::ArrayRef<at::SymInt> max_lengths,
                                  const int64_t padding_value)
{
    int64_t max_L = max_lengths[0].as_int_unchecked();
    TORCH_CHECK(max_L >= 0, "max_sequence_length must be non-negative, but got ", max_L);
    return jagged_to_padded_dense_impl_v1(values, offsets, max_L, padding_value);
}

class Jagged1DToDense : public torch::autograd::Function<Jagged1DToDense> {
public:
    static at::Tensor forward(AutogradContext* ctx,
                            const at::Tensor& values,
                            const at::Tensor& offsets,
                            const c10::SymInt max_sequence_length,
                            const int64_t padding_value)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        ctx->save_for_backward({values, offsets});
        return jagged_1d_to_dense(values, offsets, max_sequence_length, padding_value);
    }
    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
    {

        auto grad_output = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        auto values = saved[0];
        auto offsets = saved[1];
        int64_t totalL = values.size(0);
        // 这里需要把(B, max_1)，转换成(B,max_l,1)来复用jagged_to_padded_dense_backward_npu
        auto grad_input = dense_to_jagged_impl(grad_output.unsqueeze(-1), {offsets}, totalL);
        return {grad_input.squeeze(-1), Variable(), Variable(), Variable()};
    }
};

at::Tensor jagged_1d_to_dense_autograd(at::Tensor values,
                                            at::Tensor offsets,
                                            c10::SymInt max_sequence_length,
                                            const int64_t padding_value)
{
    return Jagged1DToDense::apply(values, offsets, max_sequence_length, padding_value);
}


class JaggedToPaddedDenseV1 : public torch::autograd::Function<JaggedToPaddedDenseV1> {
public:
    static at::Tensor forward(AutogradContext* ctx,
                              const at::Tensor& values,
                              const at::Tensor& offsets,
                              const int64_t max_lengths,
                              const double padding_value)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        ctx->save_for_backward({values, offsets});
        ctx->saved_data["padding_value"] = padding_value;
        return jagged_to_padded_dense_impl_v1(values, offsets, max_lengths, padding_value);
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
    {
        auto grad_output = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        auto values = saved[0];
        auto offsets = saved[1];
        int64_t totalL = values.size(0);
        auto grad_input = dense_to_jagged_impl(grad_output, offsets, totalL);
        return {grad_input, Variable(), Variable(), Variable()};
    }
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

at::Tensor jagged_to_padded_dense_autograd_v1_plus(const at::Tensor& values,
                                                   const tensor_list& offsets,
                                                   const at::IntArrayRef& max_lengths,
                                                   const double padding_value)
{
    TORCH_CHECK(offsets.size() == 1,
                "offsets must contain exactly 1 tensor, but got ", offsets.size(), " tensors");
    return JaggedToPaddedDenseV1::apply(values, offsets[0], max_lengths[0], padding_value);
}

at::Tensor jagged_2d_to_dense_autograd(const at::Tensor& values,
                                       const at::Tensor& offsets,
                                       const int64_t max_lengths)
{
    return JaggedToPaddedDenseV1::apply(values, offsets, max_lengths, .0);
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
    m.impl("jagged_2d_to_dense",
           torch::dispatch(c10::DispatchKey::PrivateUse1, TORCH_FN(jagged_2d_to_dense)));
    m.impl("jagged_to_padded_dense.v1",
           torch::dispatch(c10::DispatchKey::PrivateUse1, TORCH_FN(jagged_to_padded_dense_forward_v1)));
    m.impl("jagged_to_padded_dense_forward",
           torch::dispatch(c10::DispatchKey::PrivateUse1, TORCH_FN(jagged_to_padded_dense_forward)));
    m.impl("jagged_to_padded_dense_forward.v1",
           torch::dispatch(c10::DispatchKey::PrivateUse1, TORCH_FN(jagged_to_padded_dense_forward_v1)));
    m.impl("jagged_to_padded_dense_backward.v1",
           torch::dispatch(c10::DispatchKey::PrivateUse1, TORCH_FN(jagged_to_padded_dense_backward_v1)));
}

TORCH_LIBRARY_IMPL(fbgemm, AutogradPrivateUse1, m)
{
    m.impl("jagged_2d_to_dense", TORCH_FN(jagged_2d_to_dense_autograd));
    m.impl("jagged_to_padded_dense.v1", TORCH_FN(jagged_to_padded_dense_autograd_v1));
    m.impl("jagged_1d_to_dense", TORCH_FN(jagged_1d_to_dense_autograd));
    // 待dense_to_jagged增强后切至v2接口, 暂不支持高维自动求导
    m.impl("jagged_to_padded_dense.v2", TORCH_FN(jagged_to_padded_dense_autograd_v1_plus));
}

#endif  // JAGGED_TO_DENSE_V1
