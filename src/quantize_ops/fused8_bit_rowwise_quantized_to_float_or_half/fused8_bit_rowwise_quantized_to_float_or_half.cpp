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
using namespace at;

at::Tensor fused8_bit_rowwise_quantized_to_float_or_half_impl_npu(const Tensor& input_data, int64_t output_dtype = 0,
                                                                  const bool scale_bias_last = true,
                                                                  const bool quant_padding_float_type = true)
{
    check_tensor_non_empty(input_data, "input_data");
    TORCH_CHECK(input_data.dtype() == at::kByte, "Expected uint8, got ", input_data.dtype());
    TORCH_CHECK(quant_padding_float_type || !scale_bias_last,
                "scale_bias_last only works with quant_padding_float_type=true");
    TORCH_CHECK(output_dtype == 0 || output_dtype == 1 || output_dtype == 5,
                "Unsupported output_dtype: ", output_dtype);

    auto input_data_conti = input_data.contiguous();
    TORCH_CHECK(input_data_conti.dim() >= 2, "Expected input to have at least 2 dimensions, but got ",
                input_data_conti.dim());

    int64_t rows = 1;
    for (int64_t i = 0; i < input_data_conti.dim() - 1; ++i) {
        rows *= input_data_conti.size(i);
    }
    int64_t cols = input_data_conti.size(-1);

    int64_t quant_padding_size =
        quant_padding_float_type ? static_cast<int64_t>(sizeof(float)) : static_cast<int64_t>(sizeof(at::Half));
    int64_t output_cols = cols - 2 * quant_padding_size;

    std::vector<int64_t> output_shape(input_data_conti.sizes().begin(), input_data_conti.sizes().end() - 1);
    output_shape.push_back(output_cols);

    at::ScalarType out_dtype;
    if (output_dtype == 0) {
        out_dtype = at::kFloat;
    } else if (output_dtype == 1) {
        out_dtype = at::kHalf;
    } else {
        out_dtype = at::kBFloat16;
    }
    at::Tensor y = at::empty(output_shape, input_data_conti.options().dtype(out_dtype));

    if (rows == 0 || output_cols == 0) {
        return y;
    }

    EXEC_NPU_CMD(aclnnFused8BitRowwiseQuantizedToFloatOrHalf, input_data_conti, output_dtype, scale_bias_last,
                 quant_padding_float_type, y);
    return y;
}

at::Tensor fused8_bit_rowwise_quantized_to_float(const Tensor& input_data)
{
    return fused8_bit_rowwise_quantized_to_float_or_half_impl_npu(input_data, 0);
}

at::Tensor fused8_bit_rowwise_quantized_to_half(const Tensor& input_data)
{
    return fused8_bit_rowwise_quantized_to_float_or_half_impl_npu(input_data, 1);
}

at::Tensor fused8_bit_rowwise_quantized_to_float_or_half(const Tensor& input_data, int64_t output_dtype = 0,
                                                         const bool scale_bias_last = true,
                                                         const bool quant_padding_float_type = true)
{
    TORCH_CHECK(output_dtype == 0 || output_dtype == 1 || output_dtype == 5,
                "Unsupported output_dtype: ", output_dtype);
    return fused8_bit_rowwise_quantized_to_float_or_half_impl_npu(input_data, output_dtype, scale_bias_last,
                                                                  quant_padding_float_type);
}

class Fused8BitRowwiseQuantizedToFloat : public torch::autograd::Function<Fused8BitRowwiseQuantizedToFloat> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor input_data)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        auto y = fused8_bit_rowwise_quantized_to_float(input_data);
        ctx->save_for_backward({y});
        return y;
    }
};

class Fused8BitRowwiseQuantizedToHalf : public torch::autograd::Function<Fused8BitRowwiseQuantizedToHalf> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor input_data)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        auto y = fused8_bit_rowwise_quantized_to_half(input_data);
        ctx->save_for_backward({y});
        return y;
    }
};

class Fused8BitRowwiseQuantizedToFloatOrHalf
    : public torch::autograd::Function<Fused8BitRowwiseQuantizedToFloatOrHalf> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor input_data, int64_t output_dtype = 0,
                              bool scale_bias_last = true, bool quant_padding_float_type = true)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        auto y = fused8_bit_rowwise_quantized_to_float_or_half(input_data, output_dtype, scale_bias_last,
                                                               quant_padding_float_type);
        ctx->save_for_backward({y});
        return y;
    }
};

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("Fused8BitRowwiseQuantizedToFloat", &fused8_bit_rowwise_quantized_to_float);
    m.impl("Fused8BitRowwiseQuantizedToHalf", &fused8_bit_rowwise_quantized_to_half);
    m.impl("Fused8BitRowwiseQuantizedToFloatOrHalf", &fused8_bit_rowwise_quantized_to_float_or_half);
}
