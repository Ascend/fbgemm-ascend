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
using tensor_list = std::vector<at::Tensor>;
using namespace at;

at::Tensor hfp8_quantized_to_float_impl_npu(const Tensor& input, int64_t ebits, int64_t exponent_bias)
{
    check_tensor_non_empty(input, "input");
    TORCH_CHECK(input.dtype() == at::kByte, "Expected input to be uint8, but got ", input.dtype());
    TORCH_CHECK(ebits > 0, "ebits must be > 0, got ", ebits);
    TORCH_CHECK(exponent_bias > 0, "exponent_bias must be > 0, got ", exponent_bias);

    auto input_conti = input.contiguous();
    at::Tensor y = at::empty(input_conti.sizes(), input_conti.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnHfp8QuantizedToFloat, input_conti, ebits, exponent_bias, y);
    return y;
}

// 通过继承torch::autograd::Function类实现前向绑定
class HFP8QuantizedToFloat : public torch::autograd::Function<HFP8QuantizedToFloat>
{
   public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor input, int64_t ebits, int64_t exponent_bias)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        auto y = hfp8_quantized_to_float_impl_npu(input, ebits, exponent_bias);
        ctx->save_for_backward({y});
        return y;
    }
};

// 在fbgemm命名空间里注册schema
TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m) { m.impl("HFP8QuantizedToFloat", &hfp8_quantized_to_float_impl_npu); }
