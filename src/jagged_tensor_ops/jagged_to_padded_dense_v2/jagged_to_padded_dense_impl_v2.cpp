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
#include "jagged_to_padded_dense_impl_v2.h"

at::Tensor jagged_to_padded_dense_impl_v2(const at::Tensor& values, const tensor_list& offsets,
                                          const at::IntArrayRef& max_lengths, const double padding_value)
{
    if (max_lengths.size() == 1) {
        return jagged_to_padded_dense_impl_v1(values, offsets[0], max_lengths[0], padding_value);
    }
    check_tensor_dim(values, {EXPECTED_DIM_1D, EXPECTED_DIM_2D}, "values");
    TORCH_CHECK(offsets.size() > 0, "offsets must contain at least 1 tensor, but got ", offsets.size(), " tensors");
    TORCH_CHECK(offsets.size() <= MAX_OFFSETS_CNT, "offsets must contain at most ", MAX_OFFSETS_CNT,
                " tensors, but got ", offsets.size(), " tensors");
    TORCH_CHECK(max_lengths.size() == offsets.size(), "length of max_lengths.size() [", max_lengths.size(),
                "] != offsets.size() [", offsets.size(), "]");
    TORCH_CHECK(!max_lengths.empty(), "max_lengths must be non-empty");

    int dim = max_lengths.size();
    std::vector<int64_t> outputShape(dim + 2);
    outputShape[0] = offsets[0].size(0) - 1;
    outputShape[dim + 1] = values.size(-1);
    for (int i = 0; i < dim; i++) {
        outputShape[i + 1] = max_lengths[i];
    }

    at::TensorList offsets_tensor_list = at::TensorList(offsets);
    auto values_contin = values.contiguous();
    at::Tensor output = at::full(outputShape, padding_value, values.options());

    EXEC_NPU_CMD(aclnnJaggedToPaddedDenseV2, values_contin, offsets_tensor_list, max_lengths, padding_value, output);
    return output;
}

at::Tensor JaggedToPaddedDenseV2::forward(AutogradContext* ctx, const at::Tensor& values, const tensor_list& offsets,
                                          const at::IntArrayRef& max_lengths, const double padding_value)
{
    at::AutoDispatchBelowADInplaceOrView guard;
    ctx->save_for_backward(offsets);
    ctx->saved_data["total_L"] = values.size(0);
    return jagged_to_padded_dense_impl_v2(values, offsets, max_lengths, padding_value);
}

tensor_list JaggedToPaddedDenseV2::backward(AutogradContext* ctx, tensor_list grad_outputs)
{
    auto grad_output = grad_outputs[0];
    auto offsets = ctx->get_saved_variables();
    auto total_L = ctx->saved_data["total_L"].toInt();
    TORCH_CHECK(offsets.size() == 1,  // dense_to_jagged算子暂不支持多offsets场景
                "offsets must contain exactly 1 tensor, but got ", offsets.size(), " tensors");
    auto grad_input = dense_to_jagged_impl(grad_output, offsets[0], total_L);
    return {grad_input, Variable(), Variable(), Variable()};
}
