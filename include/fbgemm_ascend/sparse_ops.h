/**
 * Copyright (C) 2025-2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#pragma once

#include <ATen/Tensor.h>
#include <c10/core/SymInt.h>
#include <c10/util/Optional.h>
#include <tuple>
#include <vector>

/// @defgroup sparse-data-npu Sparse Data NPU Operators (Ascend)
/// NPU 实现，注册到 torch.ops.fbgemm (PrivateUse1)

/// 异步完整累积和：output = [0, t[0], t[0]+t[1], ...]
at::Tensor asynchronous_complete_cumsum_npu(const at::Tensor& offset);
/// 异步包含式累积和：output = [t[0], t[0]+t[1], ...]
at::Tensor asynchronous_inclusive_cumsum_npu(const at::Tensor& offset);
/// 异步排除式累积和：output = [0, t[0], t[0]+t[1], ...] 去掉最后一项
at::Tensor asynchronous_exclusive_cumsum_npu(const at::Tensor& offset);

/// block_bucketize_sparse_features 的返回类型
using BlockBucketizeResult = std::tuple<
    at::Tensor,
    at::Tensor,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>>;

BlockBucketizeResult block_bucketize_sparse_features_npu(
    const at::Tensor& lengths,
    const at::Tensor& indices,
    bool bucketize_pos,
    bool sequence,
    const at::Tensor& block_sizes,
    int64_t my_size,
    const c10::optional<at::Tensor>& weights,
    const c10::optional<at::Tensor>& batch_size_per_feature,
    int64_t max_B,
    const c10::optional<std::vector<at::Tensor>>& block_bucketize_pos,
    bool keep_orig_idx,
    const c10::optional<at::Tensor>& total_num_blocks);

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> permute1d_sparse_data_impl_npu(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& values,
    const c10::optional<at::Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum);

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> permute2d_sparse_data_impl_npu(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& values,
    const c10::optional<at::Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum);

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> permute2d_sparse_data_input1D_impl_npu(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& values,
    const int64_t& stride,
    const c10::optional<at::Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum);

at::Tensor expand_into_jagged_permute_impl_npu(
    const at::Tensor& permute,
    const at::Tensor& input_offsets,
    const at::Tensor& output_offsets,
    c10::SymInt output_size);

at::Tensor invert_permute_impl_npu(const at::Tensor& permute);

at::Tensor segment_sum_csr_impl_npu(
    const at::Tensor& csr_seg,
    const at::Tensor& values,
    int64_t batch_size);

at::Tensor offsets_range_impl_npu(const at::Tensor& offsets, int64_t range_size);

void init_address_lookup_impl_npu(
    at::Tensor& address_lookups,
    at::Tensor& buffer_offsets,
    at::Tensor& emb_sizes);
