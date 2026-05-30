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
using namespace at;

at::Tensor float_or_half_to_fused8_bit_rowwise_quantized_impl_npu(const Tensor& input_data)
{
    TORCH_CHECK(input_data.dim() >= 2, "Tensor 'input' must have >= 2 dimension(s). Found ", input_data.ndimension());
    TORCH_CHECK(
        input_data.dtype() == at::kFloat || input_data.dtype() == at::kHalf || input_data.dtype() == at::kBFloat16,
        "Expected input_data to be float32, float16 or bfloat16, but got ", input_data.dtype());
    auto input_data_conti = input_data.contiguous();

    int64_t nrows = 1;
    for (int64_t i = 0; i < input_data_conti.dim() - 1; ++i) {
        nrows *= input_data_conti.size(i);
    }
    int64_t ncols = input_data_conti.size(input_data_conti.dim() - 1);
    int64_t ncols_aligned = (ncols + 4 - 1) / 4 * 4;
    int64_t output_cols = ncols_aligned + 2 * sizeof(float);

    std::vector<int64_t> output_sizes(input_data_conti.sizes().begin(), input_data_conti.sizes().end());
    output_sizes.back() = output_cols;

    if (nrows == 0 || ncols == 0) {
        return at::empty(output_sizes, input_data_conti.options().dtype(at::kByte));
    }

    auto input_2d = input_data_conti.view({nrows, ncols});
    at::Tensor y = at::empty({nrows, output_cols}, input_data_conti.options().dtype(at::kByte));
    EXEC_NPU_CMD(aclnnFloatOrHalfToFused8BitRowwiseQuantized, input_2d, y);
    return y.view(output_sizes);
}

class FloatOrHalfToFused8BitRowwiseQuantized
    : public torch::autograd::Function<FloatOrHalfToFused8BitRowwiseQuantized> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor input_data)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        auto y = float_or_half_to_fused8_bit_rowwise_quantized_impl_npu(input_data);
        ctx->save_for_backward({y});
        return y;
    }
};

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("FloatToFused8BitRowwiseQuantized", &float_or_half_to_fused8_bit_rowwise_quantized_impl_npu);
    m.impl("HalfToFused8BitRowwiseQuantized", &float_or_half_to_fused8_bit_rowwise_quantized_impl_npu);
    m.impl("FloatOrHalfToFused8BitRowwiseQuantized", &float_or_half_to_fused8_bit_rowwise_quantized_impl_npu);
}
