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

#include "jagged_to_padded_dense_impl.h"
#include "../../common/pytorch_npu_helper.hpp"
#include "../../common/common_utils.h"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

at::Tensor jagged_to_padded_dense_impl_v1(const at::Tensor& values,
                                          const at::Tensor& offsets,
                                          const int64_t max_lengths,
                                          const double padding_value)
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
    EXEC_NPU_CMD(aclnnJaggedToPaddedDense, values_contin, offsets, max_lengths,
                 padding_value, padding_value_int64, output);
    return output;
}

at::Tensor jagged_to_padded_dense_impl_v2(const at::Tensor& values,
                                          const tensor_list& offsets,
                                          const at::IntArrayRef& max_lengths,
                                          const double padding_value)
{
    if (max_lengths.size() == 1) {
        return jagged_to_padded_dense_impl_v1(values, offsets[0], max_lengths[0], padding_value);
    }
    check_tensor_dim(values, EXPECTED_DIM_2D, "values");
    TORCH_CHECK(offsets.size() > 0,
                "offsets must contain at least 1 tensor, but got ", offsets.size(), " tensors");
    TORCH_CHECK(offsets.size() <= MAX_OFFSETS_CNT,
                "offsets must contain at most ", MAX_OFFSETS_CNT, " tensors, but got ", offsets.size(), " tensors");
    TORCH_CHECK(max_lengths.size() == offsets.size(),
                "length of max_lengths.size() [", max_lengths.size(), "] != offsets.size() [", offsets.size(), "]");
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

at::Tensor dense_to_jagged_impl(const at::Tensor& dense,
                                const at::Tensor& offsets,
                                const c10::optional<int64_t>& total_L)
{
    check_tensor_non_empty(dense, "dense");
    check_tensor_non_empty(offsets, "offsets");

    // 检查NPU设备且设备ID一致
    std::vector<at::Tensor> tensors = {dense, offsets};
    std::vector<std::string> names = {"dense", "offsets"};
    check_tensor_npu_device(tensors, names);

    const at::OptionalDeviceGuard guard(device_of(dense));

    // 2D [B, max_len]：1D jagged backward，内部当作 [B, max_len, 1] 处理，输出再 squeeze 为 [total_L]
    // 3D [B, max_len, D]：2D jagged backward，输出 [total_L, D]
    at::Tensor dense_contin = dense.contiguous();
    auto D = dense_contin.size(-1);
    bool output_1d = (dense.dim() == EXPECTED_DIM_2D);
    if (output_1d) {
        dense_contin = dense_contin.unsqueeze(-1);
    }

    int64_t totalLComputed;
    if (total_L.has_value()) {
        totalLComputed = total_L.value();
    } else {
        totalLComputed = static_cast<int64_t>(offsets.max().item<int64_t>());
    }

    auto output = at::empty({totalLComputed, D}, dense.options());
    EXEC_NPU_CMD(aclnnDenseToJagged, dense_contin, offsets, totalLComputed, output);
    if (output_1d) {
        return output.squeeze(-1);
    }
    return output;
}
