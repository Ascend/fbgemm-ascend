/**
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../../common/pytorch_npu_helper.hpp"
#include "../../common/common_utils.h"

at::Tensor direct_mapped_lxu_cache_lookup_npu(at::Tensor linear_cache_indices, at::Tensor lxu_cache_state,
                                              int64_t invalid_index, bool gather_cache_status,
                                              std::optional<at::Tensor> uvm_cache_stats)
{
    check_tensor_non_empty(lxu_cache_state, "lxu_cache_state");
    if (gather_cache_status) {
        TORCH_CHECK(uvm_cache_stats.has_value(), "gather_cache_status is true, but uvm_cache_stats is null");
    }

    std::vector<at::Tensor> tensors_to_check;
    std::vector<at::string> tensor_names = {"linear_cache_indices", "lxu_cache_state"};
    tensors_to_check.push_back(linear_cache_indices);
    tensors_to_check.push_back(lxu_cache_state);

    if (uvm_cache_stats.has_value()) {
        check_tensor_dim(uvm_cache_stats.value(), 1, "uvm_cache_stats");
        tensors_to_check.push_back(uvm_cache_stats.value());
        tensor_names.push_back("uvm_cache_stats");
    }

    check_tensor_npu_device(tensors_to_check, tensor_names);
    check_tensor_dim(linear_cache_indices, 1, "linear_cache_indices");
    check_tensor_dim(lxu_cache_state, 2, "lxu_cache_state");

    at::Tensor lxu_cache_locations = at::empty({linear_cache_indices.numel()}, linear_cache_indices.options());
    if (linear_cache_indices.numel() == 0) {
        return lxu_cache_locations;
    }

    at::Tensor linear_cache_indices_ref(linear_cache_indices);
    at::Tensor lxu_cache_state_ref(lxu_cache_state);
    at::Tensor lxu_cache_locations_ref(lxu_cache_locations);
    at::Tensor uvm_cache_stats_ref;
    if (uvm_cache_stats.has_value()) {
        uvm_cache_stats_ref = uvm_cache_stats.value();
    }

    EXEC_NPU_CMD(aclnnDirectMappedLxuCacheLookup, linear_cache_indices_ref, lxu_cache_state_ref, uvm_cache_stats_ref,
                 invalid_index, gather_cache_status, lxu_cache_locations_ref);
    return lxu_cache_locations;
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("direct_mapped_lxu_cache_lookup", &direct_mapped_lxu_cache_lookup_npu);
}
