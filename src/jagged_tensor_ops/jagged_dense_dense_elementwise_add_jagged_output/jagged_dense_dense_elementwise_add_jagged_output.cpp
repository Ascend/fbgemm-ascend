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

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "fbgemm_ascend/jagged_tensor_ops.h"
#include "../../common/common_utils.h"
#include "../../common/pytorch_npu_helper.hpp"

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;
using tensor_list = std::vector<at::Tensor>;

namespace fbgemm_npu {
namespace {

void check_offsets_list(const tensor_list& offsets, const char* name)
{
    TORCH_CHECK(!offsets.empty(), name, " must contain at least 1 tensor");
    TORCH_CHECK(offsets.size() <= MAX_OFFSETS_CNT, name, " must contain at most ", MAX_OFFSETS_CNT,
                " tensors, but got ", offsets.size());
    for (size_t i = 0; i < offsets.size(); ++i) {
        check_tensor_non_empty(offsets[i], name);
        check_tensor_dim(offsets[i], EXPECTED_DIM_1D, name);
    }
}

// 根据 dense 输入形状推导每个 jagged 维度对应的 max_length，并校验 dense 维度数与 values 形态匹配。
std::vector<int64_t> infer_max_lengths_from_dense(const at::Tensor& dense, size_t offset_count, const char* dense_name,
                                                  bool x_is_1d)
{
    const int64_t expected_dim = static_cast<int64_t>(offset_count) + (x_is_1d ? 1 : 2);
    TORCH_CHECK(dense.dim() == expected_dim, dense_name, " dim must be ", expected_dim, " for ", offset_count,
                " offset tensors, but got ", dense.dim());
    std::vector<int64_t> max_lengths(offset_count);
    for (size_t i = 0; i < offset_count; ++i) {
        max_lengths[i] = dense.size(static_cast<int64_t>(i) + 1);
    }
    return max_lengths;
}

void check_jagged_dense_inputs(const at::Tensor& xValues, const tensor_list& offsets, const at::Tensor& y,
                               const char* x_name, const char* y_name)
{
    check_offsets_list(offsets, "offsets");

    std::vector<at::Tensor> tensors = {xValues, y};
    std::vector<std::string> names = {x_name, y_name};
    for (size_t i = 0; i < offsets.size(); ++i) {
        tensors.push_back(offsets[i]);
        names.push_back("offsets [" + std::to_string(i) + "]");
    }
    check_tensor_npu_device(tensors, names);

    const bool x_is_1d = xValues.dim() == EXPECTED_DIM_1D;
    auto max_lengths = infer_max_lengths_from_dense(y, offsets.size(), y_name, x_is_1d);

    const int64_t batch_size = offsets[0].size(0) - 1;
    TORCH_CHECK(batch_size == y.size(0), "batch size from offsets[0] (", batch_size, ") must match ", y_name,
                ".size(0) (", y.size(0), ")");
    if (!x_is_1d) {
        TORCH_CHECK(xValues.size(-1) == y.size(-1), x_name, " last dim (", xValues.size(-1), ") must match ", y_name,
                    " last dim (", y.size(-1), ")");
    }
    for (int64_t max_length : max_lengths) {
        TORCH_CHECK(max_length >= 0, "max length inferred from ", y_name, " must be non-negative");
    }
}

at::Tensor canonicalize_dense_for_fused(const at::Tensor& dense, bool x_was_1d)
{
    return x_was_1d ? dense.contiguous().unsqueeze(-1) : dense.contiguous();
}

// 将 jagged 梯度按 dense_like 的形状回填到 padded dense，用于生成 y0/y1 的梯度。
at::Tensor jagged_to_padded_dense_v2_with_shape(const at::Tensor& values, const tensor_list& offsets,
                                                const at::Tensor& dense_like, double padding_value)
{
    const bool output_without_inner = values.dim() == EXPECTED_DIM_1D;
    at::Tensor values_work = output_without_inner ? values.contiguous().unsqueeze(-1) : values.contiguous();
    auto max_lengths = infer_max_lengths_from_dense(dense_like, offsets.size(), "dense_like", output_without_inner);
    auto out = jagged_to_padded_dense_impl_v2(values_work, offsets, max_lengths, padding_value);
    if (output_without_inner) {
        return out.squeeze(-1);
    }
    return out;
}
}  // namespace

// NPU 前向入口：将 jagged values 与两个 dense tensor 在有效 jagged 位置逐元素相加，输出仍保持 jagged values 形状。
at::Tensor jagged_dense_dense_elementwise_add_jagged_output_forward_npu(const at::Tensor& xValues,
                                                                        const tensor_list& offsets,
                                                                        const at::Tensor& y0, const at::Tensor& y1)
{
    check_jagged_dense_inputs(xValues, offsets, y0, "xValues", "y0");
    TORCH_CHECK(xValues.dim() > 0, "xValues must have non-zero dimensions");
    if (xValues.size(0) == 0) {
        return xValues.contiguous();
    }
    TORCH_CHECK(y0.sizes() == y1.sizes(), "y0 and y1 must have the same shape, got y0 ", y0.sizes(), " vs y1 ",
                y1.sizes());
    std::vector<at::Tensor> tensors = {xValues, y0, y1};
    std::vector<std::string> names = {"xValues", "y0", "y1"};
    for (const auto& offset : offsets) {
        tensors.push_back(offset);
        names.push_back("offsets");
    }
    check_tensor_npu_device(tensors, names);

    const at::OptionalDeviceGuard guard(device_of(xValues));
    const bool x_was_1d = (xValues.dim() == EXPECTED_DIM_1D);
    // ACLNN 自定义算子按二维 values 处理，1D 输入在 host 侧补齐最后一维，返回前再 squeeze 回原形状。
    at::Tensor x_work = x_was_1d ? xValues.contiguous().unsqueeze(-1) : xValues.contiguous();
    at::Tensor y0_work = canonicalize_dense_for_fused(y0, x_was_1d);
    at::Tensor y1_work = canonicalize_dense_for_fused(y1, x_was_1d);
    const int64_t jagged_dim0 = x_work.size(0);
    auto out = at::empty_like(x_work);
    at::TensorList offsets_tensor_list(offsets);

    EXEC_NPU_CMD(aclnnJaggedDenseDenseElementwiseAddJaggedOutput, x_work, y0_work, y1_work, offsets_tensor_list,
                 jagged_dim0, out);
    if (x_was_1d) {
        return out.squeeze(-1);
    }
    return out;
}

// 对外主接口返回 values 与原 offsets，保持 FBGEMM jagged_output 算子接口约定。
std::tuple<at::Tensor, std::vector<at::Tensor>> jagged_dense_dense_elementwise_add_jagged_output_npu(
    const at::Tensor& xValues, const tensor_list& offsets, const at::Tensor& y0, const at::Tensor& y1)
{
    auto sum_values = jagged_dense_dense_elementwise_add_jagged_output_forward_npu(xValues, offsets, y0, y1);
    return std::make_tuple(sum_values, offsets);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> jagged_dense_dense_elementwise_add_jagged_output_backward_npu(
    const at::Tensor& grad_output, const at::Tensor& xValues, const tensor_list& offsets, const at::Tensor& y0,
    const at::Tensor& y1)
{
    auto grad_x = grad_output.contiguous();
    if (grad_output.size(0) == 0) {
        auto grad_y0 = at::zeros_like(y0);
        auto grad_y1 = at::zeros_like(y1);
        return std::make_tuple(grad_x, grad_y0, grad_y1);
    }
    auto grad_y0 = jagged_to_padded_dense_v2_with_shape(grad_x, offsets, y0, 0.0);
    auto grad_y1 = jagged_to_padded_dense_v2_with_shape(grad_x, offsets, y1, 0.0);
    return std::make_tuple(grad_x, grad_y0, grad_y1);
}

std::tuple<at::Tensor, std::vector<at::Tensor>> jagged_dense_dense_elementwise_add_jagged_output_meta(
    const at::Tensor& xValues, const tensor_list& offsets, const at::Tensor& y0, const at::Tensor& y1)
{
    TORCH_CHECK(y0.sym_sizes() == y1.sym_sizes(), "y0 and y1 must have the same shape, got y0 ", y0.sym_sizes(),
                " vs y1 ", y1.sym_sizes());
    auto out = at::empty_symint(xValues.sym_sizes(), xValues.options().device(c10::kMeta));
    std::vector<at::Tensor> output_offsets;
    output_offsets.reserve(offsets.size());
    for (const auto& offset : offsets) {
        output_offsets.emplace_back(at::empty_symint(offset.sym_sizes(), offset.options().device(c10::kMeta)));
    }
    return std::make_tuple(out, output_offsets);
}

at::Tensor jagged_dense_dense_elementwise_add_jagged_output_forward_meta(const at::Tensor& xValues,
                                                                         const tensor_list& offsets,
                                                                         const at::Tensor& y0, const at::Tensor& y1)
{
    return std::get<0>(jagged_dense_dense_elementwise_add_jagged_output_meta(xValues, offsets, y0, y1));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> jagged_dense_dense_elementwise_add_jagged_output_backward_meta(
    const at::Tensor& grad_output, const at::Tensor& xValues, const tensor_list& offsets, const at::Tensor& y0,
    const at::Tensor& y1)
{
    auto grad_x = at::empty_symint(xValues.sym_sizes(), xValues.options().device(c10::kMeta));
    auto grad_y0 = at::empty_symint(y0.sym_sizes(), y0.options().device(c10::kMeta));
    auto grad_y1 = at::empty_symint(y1.sym_sizes(), y1.options().device(c10::kMeta));
    return std::make_tuple(grad_x, grad_y0, grad_y1);
}
}  // namespace fbgemm_npu

TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("jagged_dense_dense_elementwise_add_jagged_output_backward(Tensor grad_output, Tensor xValues, Tensor[] "
          "offsets, Tensor y0, Tensor y1) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("jagged_dense_dense_elementwise_add_jagged_output",
           torch::dispatch(c10::DispatchKey::PrivateUse1,
                           TORCH_FN(fbgemm_npu::jagged_dense_dense_elementwise_add_jagged_output_npu)));
    m.impl("jagged_dense_dense_elementwise_add_jagged_output_forward",
           torch::dispatch(c10::DispatchKey::PrivateUse1,
                           TORCH_FN(fbgemm_npu::jagged_dense_dense_elementwise_add_jagged_output_forward_npu)));
    m.impl("jagged_dense_dense_elementwise_add_jagged_output_backward",
           torch::dispatch(c10::DispatchKey::PrivateUse1,
                           TORCH_FN(fbgemm_npu::jagged_dense_dense_elementwise_add_jagged_output_backward_npu)));
}

TORCH_LIBRARY_IMPL(fbgemm, Meta, m)
{
    m.impl("jagged_dense_dense_elementwise_add_jagged_output",
           TORCH_FN(fbgemm_npu::jagged_dense_dense_elementwise_add_jagged_output_meta));
    m.impl("jagged_dense_dense_elementwise_add_jagged_output_forward",
           TORCH_FN(fbgemm_npu::jagged_dense_dense_elementwise_add_jagged_output_forward_meta));
    m.impl("jagged_dense_dense_elementwise_add_jagged_output_backward",
           TORCH_FN(fbgemm_npu::jagged_dense_dense_elementwise_add_jagged_output_backward_meta));
}
