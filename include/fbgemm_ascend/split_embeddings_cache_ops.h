/**
 * Copyright (C) 2025-2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> get_unique_indices_impl_npu(
    const at::Tensor &linear_indices,
    const int64_t max_indices,
    const bool compute_count);

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>> get_unique_indices_with_inverse_impl_npu(
    const at::Tensor &linear_indices,
    const int64_t max_indices,
    const bool compute_count,
    const bool compute_inverse_indices);

/// 线性化缓存索引：将多维缓存索引映射到线性索引空间
at::Tensor linearize_cache_indices_npu(
    const at::Tensor& cache_hash_size_cumsum,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    const c10::optional<at::Tensor>& B_offsets,
    const int64_t max_B,
    const int64_t indices_base_offset
);