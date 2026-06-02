/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <iostream>

#include "../../common/pytorch_npu_helper.hpp"
#include "../../common/common_utils.h"

at::Tensor lxu_cache_lookup_npu(const at::Tensor linear_cache_indices, const at::Tensor lxu_cache_state,
                                const int64_t invalid_index, const bool gather_cache_stats,
                                const std::optional<at::Tensor> uvm_cache_stats,
                                const std::optional<at::Tensor> num_uniq_cache_indices,
                                const std::optional<at::Tensor> lxu_cache_locations_output)
{
    const bool uniq_lookup = num_uniq_cache_indices.has_value();
    TORCH_CHECK(!uniq_lookup || !gather_cache_stats,
                "Unique lxu_cache_locations generation does not support gather_cache_stats=true");

    check_tensor_non_empty(lxu_cache_state, "lxu_cache_state");

    std::vector<at::Tensor> tensors_to_check;
    std::vector<at::string> tensor_names = {"linear_cache_indices", "lxu_cache_state"};
    tensors_to_check.push_back(linear_cache_indices);
    tensors_to_check.push_back(lxu_cache_state);

    if (num_uniq_cache_indices.has_value()) {
        check_tensor_dim(num_uniq_cache_indices.value(), 1, "num_uniq_cache_indices");
        tensors_to_check.push_back(num_uniq_cache_indices.value());
        tensor_names.push_back("num_uniq_cache_indices");
    }

    if (uvm_cache_stats.has_value()) {
        check_tensor_dim(uvm_cache_stats.value(), 1, "uvm_cache_stats");
        tensors_to_check.push_back(uvm_cache_stats.value());
        tensor_names.push_back("uvm_cache_stats");
    }

    check_tensor_npu_device(tensors_to_check, tensor_names);
    check_tensor_dim(linear_cache_indices, 1, "linear_cache_indices");
    check_tensor_dim(lxu_cache_state, 2, "lxu_cache_state");

    if (gather_cache_stats && !uniq_lookup) {
        TORCH_CHECK(uvm_cache_stats.has_value(), "gather_cache_stats is true, but uvm_cache_stats is null");
    }

    at::Tensor lxu_cache_locations = lxu_cache_locations_output.value_or(
        empty_like(linear_cache_indices, linear_cache_indices.options().dtype(at::kInt)));

    const auto N = linear_cache_indices.numel();
    if (N == 0) {
        return lxu_cache_locations;
    }

    at::Tensor linear_cache_indices_ref(linear_cache_indices);
    at::Tensor lxu_cache_state_ref(lxu_cache_state);
    at::Tensor lxu_cache_locations_ref(lxu_cache_locations);
    at::Tensor uvm_cache_stats_ref;
    at::Tensor num_uniq_cache_indices_ref;

    if (uvm_cache_stats.has_value()) {
        uvm_cache_stats_ref = uvm_cache_stats.value();
    }
    if (num_uniq_cache_indices.has_value()) {
        num_uniq_cache_indices_ref = num_uniq_cache_indices.value();
    }

    EXEC_NPU_CMD(aclnnLxuCacheLookup, linear_cache_indices_ref, lxu_cache_state_ref, uvm_cache_stats_ref,
                 num_uniq_cache_indices_ref, invalid_index, gather_cache_stats, uniq_lookup, lxu_cache_locations_ref);

    return lxu_cache_locations;
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("lxu_cache_lookup", &lxu_cache_lookup_npu);
}
