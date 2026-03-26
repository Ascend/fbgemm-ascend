/**
 * 声明 Jagged Tensor 相关算子（与 fbgemm_gpu 风格保持一致）。
 */
#pragma once

#include <ATen/Tensor.h>
#include <ATen/core/DimVector.h>
#include <c10/core/SymInt.h>
#include <c10/util/Optional.h>

#include <tuple>
#include <vector>

// ----- Select / Permute 算子 -----
std::vector<at::Tensor> keyed_jagged_index_select_dim1_impl_npu(
    const at::Tensor& values,
    const at::Tensor& lengths,
    const at::Tensor& offsets,
    const at::Tensor& indices,
    const int64_t& batch_size,
    const c10::optional<at::Tensor>& weights,
    const c10::optional<int64_t>& selected_lengths_sum);

// ----- Jagged ↔ Dense 互转核心算子 -----
at::Tensor jagged_to_padded_dense_forward_npu(
    const at::Tensor& values,
    const std::vector<at::Tensor>& offsets,
    int64_t max_lengths,
    double padding_value);

at::Tensor dense_to_jagged_forward_npu(
    const at::Tensor& dense,
    const std::vector<at::Tensor>& offsets,
    const c10::optional<int64_t> total_L);

std::tuple<at::Tensor, std::vector<at::Tensor>> dense_to_jagged_npu(
    const at::Tensor& dense,
    const std::vector<at::Tensor>& offsets,
    const c10::optional<int64_t> total_L);

at::Tensor dense_to_jagged_backward_npu(
    const at::Tensor& values,
    const std::vector<at::Tensor>& offsets,
    int64_t max_lengths,
    double padding_value);

at::Tensor dense_to_jagged_autograd(
    const at::Tensor& dense,
    const std::vector<at::Tensor>& offsets,
    const c10::optional<int64_t> total_L);

std::tuple<at::Tensor, std::vector<at::Tensor>> dense_to_jagged_npu_autograd(
    const at::Tensor& dense,
    const std::vector<at::Tensor>& offsets,
    const c10::optional<int64_t> total_L);

// ----- fbgemm_npu 内部实现（多版本函数、Autograd 包装） -----
namespace fbgemm_npu {

at::Tensor dense_to_jagged_forward_npu(
    const at::Tensor& dense,
    const std::vector<at::Tensor>& offsets,
    const c10::optional<int64_t>& total_L);

at::Tensor jagged_to_padded_dense_forward_npu_v1(
    const at::Tensor& values,
    const std::vector<at::Tensor>& offsets,
    int64_t max_lengths,
    double padding_value);

at::Tensor jagged_to_padded_dense_npu_v1(
    const at::Tensor& values,
    const std::vector<at::Tensor>& offsets,
    int64_t max_lengths,
    double padding_value);

at::Tensor jagged_to_padded_dense_forward_npu_v2(
    const at::Tensor& values,
    const std::vector<at::Tensor>& offsets,
    const at::IntArrayRef max_lengths,
    double padding_value);

at::Tensor jagged_to_padded_dense_npu_v2(
    const at::Tensor& values,
    const std::vector<at::Tensor>& offsets,
    const at::IntArrayRef max_lengths,
    double padding_value);

at::Tensor jagged_to_padded_dense_backward_npu(
    const at::Tensor& grad_output,
    const std::vector<at::Tensor>& offsets,
    int64_t total_L);

at::Tensor jagged_2d_to_dense_npu(
    at::Tensor values,
    at::Tensor offsets,
    c10::SymInt max_sequence_length);

at::Tensor jagged_to_padded_dense_npu_v1_autograd(
    const at::Tensor& values,
    const std::vector<at::Tensor>& offsets,
    int64_t max_lengths,
    double padding_value);

at::Tensor jagged_to_padded_dense_npu_v2_autograd(
    const at::Tensor& values,
    const std::vector<at::Tensor>& offsets,
    const at::IntArrayRef max_lengths,
    double padding_value);

at::Tensor jagged_2d_to_dense_npu_autograd(
    at::Tensor values,
    at::Tensor offsets,
    c10::SymInt max_sequence_length);

}
