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

#include "../../common/common_utils.h"
#include "../../common/pytorch_npu_helper.hpp"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

at::Tensor float_to_hfp8_quantized_impl_npu(const Tensor& input, int64_t ebits, int64_t exponent_bias, double max_pos)
{
    check_tensor_non_empty(input, "input");
    TORCH_CHECK(input.dtype() == at::kFloat, "Expected input to be float32, but got ", input.dtype());
    TORCH_CHECK(ebits > 0, "ebits must be > 0, got ", ebits);
    TORCH_CHECK(exponent_bias > 0, "exponent_bias must be > 0, got ", exponent_bias);

    auto input_conti = input.contiguous();
    at::Tensor y = at::empty(input_conti.sizes(), input_conti.options().dtype(at::kByte));
    float max_pos_f = static_cast<float>(max_pos);
    EXEC_NPU_CMD(aclnnFloatToHfp8Quantized, input_conti, ebits, exponent_bias, max_pos_f, y);
    return y;
}

// 通过继承torch::autograd::Function类实现前向绑定
class FloatToHFP8Quantized : public torch::autograd::Function<FloatToHFP8Quantized>
{
   public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor input, int64_t ebits, int64_t exponent_bias,
                              double max_pos)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        auto y = float_to_hfp8_quantized_impl_npu(input, ebits, exponent_bias, max_pos);
        ctx->save_for_backward({y});
        return y;
    }
};

// 在fbgemm命名空间里注册schema
TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m) { m.impl("FloatToHFP8Quantized", &float_to_hfp8_quantized_impl_npu); }
