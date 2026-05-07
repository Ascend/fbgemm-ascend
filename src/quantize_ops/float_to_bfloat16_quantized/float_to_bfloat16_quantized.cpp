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
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using namespace at;

at::Tensor float_to_bfloat16_quantized_impl_npu(const Tensor& input)
{
    check_tensor_non_empty(input, "input");
    TORCH_CHECK(input.dtype() == at::kFloat, "Expected input to be float32, but got ", input.dtype());

    auto input_conti = input.contiguous();
    at::Tensor y = at::empty(input_conti.sizes(), input_conti.options().dtype(at::kBFloat16));
    EXEC_NPU_CMD(aclnnFloatToBfloat16Quantized, input_conti, y);
    return y;
}

// 通过继承torch::autograd::Function类实现前向绑定
class FloatToBfloat16Quantized : public torch::autograd::Function<FloatToBfloat16Quantized> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor input)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        auto y = float_to_bfloat16_quantized_impl_npu(input);
        ctx->save_for_backward({y});
        return y;
    }
};

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("FloatToBfloat16Quantized", &float_to_bfloat16_quantized_impl_npu);
}