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
#include <array>
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

// 与底层自定义算子的elementwise_mode保持一致。
constexpr int64_t JAGGED_EW_BINARY_MODE_ADD = 0;
constexpr int64_t JAGGED_EW_BINARY_MODE_MUL = 1;

// 检查offsets list的数量和每个offset tensor的基本维度。
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

// 根据dense shape推导每个jagged维度对应的max length。
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

// 检查jagged-dense elementwise前向输入的设备、shape和尾维匹配关系。
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

// dense_to_jagged与add/mul二元计算。
at::Tensor run_jagged_dense_elementwise_binary_jagged_output_fused(const at::Tensor& x_values,
                                                                   const tensor_list& x_offsets, const at::Tensor& y,
                                                                   int64_t elementwise_mode)
{
    TORCH_CHECK(x_values.dim() > 0, "x_values must have non-zero dimensions");
    if (x_values.numel() == 0) {
        return x_values.contiguous();
    }
    const bool x_was_1d = (x_values.dim() == EXPECTED_DIM_1D);
    at::Tensor x_work = x_was_1d ? x_values.contiguous().unsqueeze(-1) : x_values.contiguous();

    at::Tensor y_work = x_was_1d ? y.contiguous().unsqueeze(-1) : y.contiguous();
    const int64_t jagged_dim0 = x_work.size(0);
    auto out = at::empty_like(x_work);
    at::TensorList offsets_tensor_list(x_offsets);
    EXEC_NPU_CMD(aclnnJaggedDenseElementwiseBinaryJaggedOutput, x_work, y_work, offsets_tensor_list, jagged_dim0,
                 elementwise_mode, out);
    if (x_was_1d) {
        return out.squeeze(-1);
    }
    return out;
}

// 将jagged梯度还原成与dense_like形状一致的padded dense梯度。
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

// 按多层offsets将dense抽取为与x_values对齐的jagged values。
at::Tensor dense_to_jagged_like(const at::Tensor& dense, const tensor_list& offsets, int64_t total_L, bool output_1d)
{
    at::Tensor grad = output_1d ? dense.unsqueeze(-1) : dense;
    for (size_t i = 0; i < offsets.size(); ++i) {
        // 每轮恢复一层jagged维度，其余维度先展平到dense_to_jagged_impl需要的尾维。
        std::array<int64_t, 3> flatten_shape = {grad.size(0), grad.size(1), 1};
        for (int64_t dim = 2; dim < grad.dim(); ++dim) {
            flatten_shape[2] *= grad.size(dim);
        }

        c10::optional<int64_t> current_total_L = c10::nullopt;
        if (i == offsets.size() - 1) {
            current_total_L = total_L;
        }
        auto flattened_grad = grad.reshape(c10::IntArrayRef(flatten_shape.data(), flatten_shape.size()));
        auto jagged_grad = dense_to_jagged_impl(flattened_grad, offsets[i], current_total_L);

        std::array<int64_t, MAX_OFFSETS_CNT + 2> unflatten_shape = {jagged_grad.size(0)};
        const auto unflatten_dim = static_cast<size_t>(grad.dim() - 1);
        for (int64_t dim = 2; dim < grad.dim(); ++dim) {
            unflatten_shape[static_cast<size_t>(dim - 1)] = grad.size(dim);
        }
        grad = jagged_grad.reshape(c10::IntArrayRef(unflatten_shape.data(), unflatten_dim));
    }
    if (output_1d) {
        return grad.squeeze(-1);
    }
    return grad;
}
}  // namespace

// add_jagged_output前向：输出x_values + dense_to_jagged(y)。
std::tuple<at::Tensor, std::vector<at::Tensor>> jagged_dense_elementwise_add_jagged_output_npu(
    const at::Tensor& x_values, const tensor_list& x_offsets, const at::Tensor& y)
{
    check_jagged_dense_inputs(x_values, x_offsets, y, "x_values", "y");
    const at::OptionalDeviceGuard guard(device_of(x_values));
    auto sum_values =
        run_jagged_dense_elementwise_binary_jagged_output_fused(x_values, x_offsets, y, JAGGED_EW_BINARY_MODE_ADD);
    return std::make_tuple(sum_values, x_offsets);
}

// add_jagged_output反向：grad_x为grad_output，grad_y由jagged梯度还原到dense形状。
std::tuple<at::Tensor, at::Tensor> jagged_dense_elementwise_add_jagged_output_backward_npu(
    const at::Tensor& grad_output, const at::Tensor& x_values, const tensor_list& x_offsets, const at::Tensor& y)
{
    auto grad_x = grad_output.contiguous();
    if (grad_output.numel() == 0 || x_values.numel() == 0) {
        return std::make_tuple(grad_x, at::zeros_like(y));
    }
    auto grad_y = jagged_to_padded_dense_v2_with_shape(grad_x, x_offsets, y, 0.0);
    return std::make_tuple(grad_x, grad_y);
}

// mul前向：输出x_values * dense_to_jagged(y)。
at::Tensor jagged_dense_elementwise_mul_forward_npu(const at::Tensor& x_values, const tensor_list& x_offsets,
                                                    const at::Tensor& y)
{
    check_jagged_dense_inputs(x_values, x_offsets, y, "x_values", "y");
    const at::OptionalDeviceGuard guard(device_of(x_values));
    return run_jagged_dense_elementwise_binary_jagged_output_fused(x_values, x_offsets, y, JAGGED_EW_BINARY_MODE_MUL);
}

// mul对外接口：返回乘法结果和原始offsets。
std::tuple<at::Tensor, std::vector<at::Tensor>> jagged_dense_elementwise_mul_npu(const at::Tensor& x_values,
                                                                                 const tensor_list& x_offsets,
                                                                                 const at::Tensor& y)
{
    auto prod = jagged_dense_elementwise_mul_forward_npu(x_values, x_offsets, y);
    return std::make_tuple(prod, x_offsets);
}

// mul反向：grad_x乘以y_jagged，grad_y由grad_output * x_values还原到dense形状。
std::tuple<at::Tensor, at::Tensor> jagged_dense_elementwise_mul_backward_npu(const at::Tensor& grad_output,
                                                                             const tensor_list& x_offsets,
                                                                             const at::Tensor& y,
                                                                             const at::Tensor& x_values)
{
    auto grad_x = at::empty_like(x_values);
    if (grad_output.numel() == 0 || x_values.numel() == 0) {
        grad_x.zero_();
        return std::make_tuple(grad_x, at::zeros_like(y));
    }
    auto y_jagged = dense_to_jagged_like(y, x_offsets, x_values.size(0), x_values.dim() == EXPECTED_DIM_1D);
    grad_x = at::mul(grad_output.contiguous(), y_jagged);
    auto grad_chain = at::mul(grad_output.contiguous(), x_values.contiguous());
    auto grad_y = jagged_to_padded_dense_v2_with_shape(grad_chain, x_offsets, y, 0.0);
    return std::make_tuple(grad_x, grad_y);
}

// add/mul tuple输出接口的Meta实现，仅推导输出shape。
std::tuple<at::Tensor, std::vector<at::Tensor>> jagged_dense_elementwise_binary_jagged_output_meta(
    const at::Tensor& x_values, const tensor_list& x_offsets, const at::Tensor& y)
{
    return std::make_tuple(at::empty_like(x_values), x_offsets);
}

// mul_forward的Meta实现，仅返回与x_values同形状的占位tensor。
at::Tensor jagged_dense_elementwise_mul_forward_meta(const at::Tensor& x_values, const tensor_list& x_offsets,
                                                     const at::Tensor& y)
{
    return at::empty_like(x_values);
}

// mul_backward的Meta实现，返回grad_x和grad_y的占位tensor。
std::tuple<at::Tensor, at::Tensor> jagged_dense_elementwise_mul_backward_meta(const at::Tensor& grad_output,
                                                                              const tensor_list& x_offsets,
                                                                              const at::Tensor& y,
                                                                              const at::Tensor& x_values)
{
    return std::make_tuple(at::empty_like(grad_output), at::empty_like(y));
}
}  // namespace fbgemm_npu

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("jagged_dense_elementwise_add_jagged_output",
           torch::dispatch(c10::DispatchKey::PrivateUse1,
                           TORCH_FN(fbgemm_npu::jagged_dense_elementwise_add_jagged_output_npu)));
    m.impl("jagged_dense_elementwise_mul",
           torch::dispatch(c10::DispatchKey::PrivateUse1, TORCH_FN(fbgemm_npu::jagged_dense_elementwise_mul_npu)));
    m.impl(
        "jagged_dense_elementwise_mul_forward",
        torch::dispatch(c10::DispatchKey::PrivateUse1, TORCH_FN(fbgemm_npu::jagged_dense_elementwise_mul_forward_npu)));
    m.impl("jagged_dense_elementwise_mul_backward",
           torch::dispatch(c10::DispatchKey::PrivateUse1,
                           TORCH_FN(fbgemm_npu::jagged_dense_elementwise_mul_backward_npu)));
}

TORCH_LIBRARY_IMPL(fbgemm, Meta, m)
{
    m.impl("jagged_dense_elementwise_add_jagged_output",
           TORCH_FN(fbgemm_npu::jagged_dense_elementwise_binary_jagged_output_meta));
    m.impl("jagged_dense_elementwise_mul", TORCH_FN(fbgemm_npu::jagged_dense_elementwise_binary_jagged_output_meta));
    m.impl("jagged_dense_elementwise_mul_forward", TORCH_FN(fbgemm_npu::jagged_dense_elementwise_mul_forward_meta));
    m.impl("jagged_dense_elementwise_mul_backward", TORCH_FN(fbgemm_npu::jagged_dense_elementwise_mul_backward_meta));
}
