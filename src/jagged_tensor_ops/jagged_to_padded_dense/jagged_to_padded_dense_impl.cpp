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

at::Tensor jagged_to_padded_dense_impl_v1(const at::Tensor& values, const at::Tensor& offsets,
                                          const int64_t max_lengths, const double padding_value)
{
    // Support 1D (jagged_to_1d_dense) or 2D (jagged_to_padded_dense) values, aligned with FBGEMM
    check_tensor_dim(values, {EXPECTED_DIM_1D, EXPECTED_DIM_2D}, "values");

    check_tensor_non_empty(offsets, "offsets");
    check_tensor_dim(offsets, EXPECTED_DIM_1D, "offsets");

    // 检查NPU设备且设备ID一致
    std::vector<at::Tensor> tensors = {values, offsets};
    std::vector<std::string> names = {"values", "offsets"};
    check_tensor_npu_device(tensors, names);

    TORCH_CHECK(max_lengths >= 0, "max_lengths must be non-negative, but got ", max_lengths);

    auto B = offsets.size(0) - 1;
    at::Tensor output;
    if (values.dim() == EXPECTED_DIM_1D) {
        // jagged_to_1d_dense: values [total_L] -> out [B, max_lengths]
        output = at::empty({B, max_lengths}, values.options());
    } else {
        // jagged_to_padded_dense: values [total_L, D] -> out [B, max_lengths, D]
        auto D = values.size(-1);
        output = at::empty({B, max_lengths, D}, values.options());
    }

    if (max_lengths == 0) {
        return output;
    }

    auto values_contin = values.contiguous();
    int64_t padding_value_int64 = static_cast<int64_t>(padding_value);
    EXEC_NPU_CMD(aclnnJaggedToPaddedDense, values_contin, offsets, max_lengths, padding_value, padding_value_int64,
                 output);
    return output;
}

at::Tensor JaggedToPaddedDenseV1::forward(AutogradContext* ctx, const at::Tensor& values, const at::Tensor& offsets,
                                          const int64_t max_lengths, const double padding_value)
{
    at::AutoDispatchBelowADInplaceOrView guard;
    ctx->save_for_backward({offsets});
    ctx->saved_data["total_L"] = values.size(0);
    return jagged_to_padded_dense_impl_v1(values, offsets, max_lengths, padding_value);
}

tensor_list JaggedToPaddedDenseV1::backward(AutogradContext* ctx, tensor_list grad_outputs)
{
    auto grad_output = grad_outputs[0];
    auto offsets = ctx->get_saved_variables();
    auto total_L = ctx->saved_data["total_L"].toInt();
    auto grad_input = dense_to_jagged_impl(grad_output, offsets[0], total_L);
    return {grad_input, Variable(), Variable(), Variable()};
}
