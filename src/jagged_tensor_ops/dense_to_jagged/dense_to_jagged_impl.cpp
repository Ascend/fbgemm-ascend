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
#include "dense_to_jagged_impl.h"

at::Tensor dense_to_jagged_impl(const at::Tensor& dense, const at::Tensor& offsets,
                                const c10::optional<int64_t>& total_L)
{
    // 目前只支持2、3维的dense
    check_tensor_dim(dense, {EXPECTED_DIM_2D, EXPECTED_DIM_3D}, "dense");
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

    auto output = at::zeros({totalLComputed, D}, dense.options());
    EXEC_NPU_CMD(aclnnDenseToJagged, dense_contin, offsets, totalLComputed, output);
    if (output_1d) {
        return output.squeeze(-1);
    }
    return output;
}

at::Tensor DenseToJaggedFunction::forward(AutogradContext* ctx, const at::Tensor& dense, const tensor_list& offsets,
                                          const c10::optional<int64_t> total_L)
{
    at::AutoDispatchBelowADInplaceOrView guard;
    ctx->save_for_backward(offsets);
    ctx->saved_data["max_len"] = dense.sym_size(1);
    return dense_to_jagged_impl(dense, offsets[0], total_L);
}

tensor_list DenseToJaggedFunction::backward(AutogradContext* ctx, tensor_list grad_outputs)
{
    auto grad_output = grad_outputs[0];
    auto offsets = ctx->get_saved_variables();
    auto max_len = ctx->saved_data["max_len"].toInt();
    auto grad_dense = jagged_to_padded_dense_impl_v1(grad_output, offsets[0], max_len, 0.0);
    return {grad_dense, Variable(), Variable()};
}
