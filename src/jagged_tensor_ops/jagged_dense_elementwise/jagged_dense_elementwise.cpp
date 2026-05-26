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
#include <vector>

#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "fbgemm_ascend/jagged_tensor_ops.h"
#include "../../common/common_utils.h"
#include "../../common/pytorch_npu_helper.hpp"

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using torch::autograd::variable_list;
using tensor_list = std::vector<at::Tensor>;

namespace fbgemm_npu {
namespace {

void check_offsets_list(const tensor_list& offsets, const char* name)
{
    // 校验 forward 和 backward 共用的 jagged offsets 输入约束。
    TORCH_CHECK(!offsets.empty(), name, " must contain at least 1 tensor");
    TORCH_CHECK(offsets.size() <= MAX_OFFSETS_CNT, name, " must contain at most ", MAX_OFFSETS_CNT,
                " tensors, but got ", offsets.size());
    for (size_t i = 0; i < offsets.size(); ++i) {
        check_tensor_non_empty(offsets[i], name);
        check_tensor_dim(offsets[i], EXPECTED_DIM_1D, name);
    }
}

std::vector<int64_t> infer_max_lengths_from_dense(const at::Tensor& dense, size_t offset_count, const char* dense_name,
                                                  bool x_is_1d)
{
    // max_lengths 直接由 dense 的中间维度推导。dense 维度时对应 [B, max_L0, ..., max_Ln, D]，
    const int64_t expected_dim = static_cast<int64_t>(offset_count) + (x_is_1d ? 1 : 2);
    TORCH_CHECK(dense.dim() == expected_dim, dense_name, " dim must be ", expected_dim, " for ", offset_count,
                " offset tensors, but got ", dense.dim());
    std::vector<int64_t> max_lengths(offset_count);
    for (size_t i = 0; i < offset_count; ++i) {
        max_lengths[i] = dense.size(static_cast<int64_t>(i) + 1);
    }
    return max_lengths;
}

void check_jagged_dense_inputs(const at::Tensor& x_values, const tensor_list& x_offsets, const at::Tensor& y,
                               const char* x_name, const char* y_name)
{
    check_offsets_list(x_offsets, "x_offsets");

    std::vector<at::Tensor> tensors = {x_values, y};
    std::vector<std::string> names = {x_name, y_name};
    for (size_t i = 0; i < x_offsets.size(); ++i) {
        tensors.push_back(x_offsets[i]);
        names.push_back("x_offsets");
    }
    check_tensor_npu_device(tensors, names);

    const bool x_is_1d = x_values.dim() == EXPECTED_DIM_1D;
    auto max_lengths = infer_max_lengths_from_dense(y, x_offsets.size(), y_name, x_is_1d);

    const int64_t batch_size = x_offsets[0].size(0) - 1;
    TORCH_CHECK(batch_size == y.size(0), "batch size from x_offsets[0] (", batch_size, ") must match ", y_name,
                ".size(0) (", y.size(0), ")");
    if (!x_is_1d) {
        TORCH_CHECK(x_values.size(-1) == y.size(-1), x_name, " last dim (", x_values.size(-1), ") must match ", y_name,
                    " last dim (", y.size(-1), ")");
    }
    for (int64_t max_length : max_lengths) {
        TORCH_CHECK(max_length >= 0, "max length inferred from ", y_name, " must be non-negative");
    }
}

at::Tensor jagged_to_padded_dense_like(const at::Tensor& values, const tensor_list& offsets,
                                       const at::Tensor& dense_like, double padding_value)
{
    const bool output_without_inner = values.dim() == EXPECTED_DIM_1D;
    at::Tensor values_work = output_without_inner ? values.contiguous().unsqueeze(-1) : values.contiguous();
    auto max_lengths = infer_max_lengths_from_dense(dense_like, offsets.size(), "dense_like", output_without_inner);
    auto out = jagged_to_padded_dense_impl_v2(values_work, offsets, max_lengths, padding_value);
    // 底层转换算子要求存在 inner dense 维度，1D values 临时补 D=1，输出后再压回原形状。
    if (output_without_inner) {
        return out.squeeze(-1);
    }
    return out;
}

at::Tensor dense_to_jagged_like(const at::Tensor& dense, const tensor_list& offsets, int64_t total_L, bool output_1d)
{
    at::Tensor grad = output_1d ? dense.unsqueeze(-1) : dense;
    // dense_to_jagged目前还不支持offset为list的场景，后续支持了再替换成直接调用dense_to_jagged_impl
    for (size_t i = 0; i < offsets.size(); ++i) {
        // 每轮恢复一层 jagged 维度，其余 dense 维度先展平为转换算子需要的尾维。
        std::vector<int64_t> flatten_shape;
        flatten_shape.reserve(3);
        flatten_shape.push_back(grad.size(0));
        flatten_shape.push_back(grad.size(1));
        int64_t flattened_inner = 1;
        for (int64_t dim = 2; dim < grad.dim(); ++dim) {
            flattened_inner *= grad.size(dim);
        }
        flatten_shape.push_back(flattened_inner);

        c10::optional<int64_t> current_total_L = c10::nullopt;
        if (i == offsets.size() - 1) {
            current_total_L = total_L;
        }
        auto flattened_grad = grad.reshape(flatten_shape);
        auto jagged_grad = dense_to_jagged_impl(flattened_grad, offsets[i], current_total_L);

        std::vector<int64_t> unflatten_shape;
        unflatten_shape.reserve(static_cast<size_t>(grad.dim()) - 1);
        unflatten_shape.push_back(jagged_grad.size(0));
        for (int64_t dim = 2; dim < grad.dim(); ++dim) {
            unflatten_shape.push_back(grad.size(dim));
        }
        grad = jagged_grad.reshape(unflatten_shape);
    }
    if (output_1d) {
        return grad.squeeze(-1);
    }
    return grad;
}
}  // namespace

at::Tensor jagged_dense_elementwise_add_npu_forward(const at::Tensor& jagged_values, const tensor_list& offsets,
                                                    const at::Tensor& dense_tensor)
{
    check_jagged_dense_inputs(jagged_values, offsets, dense_tensor, "jagged_values", "dense_tensor");

    const at::OptionalDeviceGuard guard(device_of(jagged_values));
    auto padded_jagged = jagged_to_padded_dense_like(jagged_values, offsets, dense_tensor, 0.0);
    return at::add(padded_jagged, dense_tensor.contiguous());
}

at::Tensor jagged_dense_elementwise_add_npu_backward(const at::Tensor& grad_output, const tensor_list& offsets,
                                                     const int64_t total_L, const bool jagged_values_is_1d)
{
    check_offsets_list(offsets, "offsets");

    std::vector<at::Tensor> tensors = {grad_output};
    std::vector<std::string> names = {"grad_output"};
    for (size_t i = 0; i < offsets.size(); ++i) {
        tensors.push_back(offsets[i]);
        names.push_back("offsets");
    }
    check_tensor_npu_device(tensors, names);

    const at::OptionalDeviceGuard guard(device_of(grad_output));
    return dense_to_jagged_like(grad_output, offsets, total_L, jagged_values_is_1d);
}

class JaggedDenseElementwiseAddFunction : public Function<JaggedDenseElementwiseAddFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, const at::Tensor& jagged_values, const tensor_list& offsets,
                              const at::Tensor& dense_tensor)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        // 保存原始 values 和 offsets，backward 需要用它们把 dense 梯度映射回 jagged values。
        tensor_list saved = {jagged_values};
        saved.insert(saved.end(), offsets.begin(), offsets.end());
        ctx->save_for_backward(saved);
        ctx->saved_data["offsets_size"] = static_cast<int64_t>(offsets.size());
        ctx->saved_data["jagged_values_is_1d"] = jagged_values.dim() == EXPECTED_DIM_1D;
        return jagged_dense_elementwise_add_npu_forward(jagged_values, offsets, dense_tensor);
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs)
    {
        auto grad_output = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        auto jagged_values = saved[0];
        auto offsets_size = ctx->saved_data["offsets_size"].toInt();
        auto jagged_values_is_1d = ctx->saved_data["jagged_values_is_1d"].toBool();

        tensor_list offsets;
        offsets.reserve(static_cast<size_t>(offsets_size));
        for (int64_t i = 0; i < offsets_size; ++i) {
            offsets.push_back(saved[static_cast<size_t>(i + 1)]);
        }

        auto grad_jagged_values =
            jagged_dense_elementwise_add_npu_backward(grad_output, offsets, jagged_values.size(0), jagged_values_is_1d);
        return {grad_jagged_values, Variable(), grad_output};
    }
};

at::Tensor jagged_dense_elementwise_add_autograd(const at::Tensor& jagged_values, const tensor_list& offsets,
                                                 const at::Tensor& dense_tensor)
{
    return JaggedDenseElementwiseAddFunction::apply(jagged_values, offsets, dense_tensor);
}

}  // namespace fbgemm_npu

TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("jagged_dense_elementwise_add_backward(Tensor grad_output, "
          "                                      Tensor[] offsets, "
          "                                      int total_L, "
          "                                      bool jagged_values_is_1d) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl(
        "jagged_dense_elementwise_add",
        torch::dispatch(c10::DispatchKey::PrivateUse1, TORCH_FN(fbgemm_npu::jagged_dense_elementwise_add_npu_forward)));
    m.impl("jagged_dense_elementwise_add_backward",
           torch::dispatch(c10::DispatchKey::PrivateUse1,
                           TORCH_FN(fbgemm_npu::jagged_dense_elementwise_add_npu_backward)));
}

TORCH_LIBRARY_IMPL(fbgemm, AutogradPrivateUse1, m)
{
    m.impl("jagged_dense_elementwise_add", TORCH_FN(fbgemm_npu::jagged_dense_elementwise_add_autograd));
}
