/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

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

#include <ATen/DeviceGuard.h>
#include <torch/library.h>

#include "../../common/common_utils.h"
#include "../../common/pytorch_npu_helper.hpp"

namespace fbgemm_ascend {

at::Tensor linearize_cache_indices_npu(const at::Tensor& cache_hash_size_cumsum, const at::Tensor& indices,
                                       const at::Tensor& offsets, const c10::optional<at::Tensor>& B_offsets,
                                       const int64_t max_B, const int64_t indices_base_offset)
{
    const at::OptionalDeviceGuard guard(device_of(cache_hash_size_cumsum));

    std::vector<at::Tensor> tensors = {cache_hash_size_cumsum, indices, offsets};
    std::vector<std::string> names = {"cache_hash_size_cumsum", "indices", "offsets"};
    check_tensor_npu_device(tensors, names);

    TORCH_CHECK(cache_hash_size_cumsum.dtype() == at::kLong, "cache_hash_size_cumsum must be int64 tensor");
    TORCH_CHECK(indices.dtype() == at::kLong || indices.dtype() == at::kInt, "indices must be int64 or int32 tensor");
    TORCH_CHECK(offsets.dtype() == at::kLong || offsets.dtype() == at::kInt, "offsets must be int64 or int32 tensor");

    const auto T = cache_hash_size_cumsum.size(0) - 1;
    TORCH_CHECK(T > 0, "T must be greater than 0");
    const int32_t total_B = offsets.size(0) - 1;

    auto cache_hash_size_cumsum_contig = cache_hash_size_cumsum.contiguous();
    auto indices_contig = indices.contiguous();
    auto offsets_contig = offsets.contiguous();

    auto linear_cache_indices = at::empty(indices.sizes(), indices.options().dtype(at::kLong));

    if (total_B == 0 || indices.numel() == 0) {
        return linear_cache_indices;
    }

    const bool use_vbe = B_offsets.has_value();
    at::Tensor table_offsets;

    if (use_vbe) {
        TORCH_CHECK(max_B >= 0, "Invalid max_B ", max_B, ". max_B must be >= 0 in VBE mode");
        auto b_offsets_slice = B_offsets.value().slice(0, 1, T + 1);
        table_offsets = at::index_select(offsets_contig, 0, b_offsets_slice).contiguous();
    } else {
        const auto B = total_B / T;
        TORCH_CHECK(B >= 0, "Invalid B ", B, ". Please check the size of offsets and cache_hash_size_cumsum.");
        table_offsets = offsets_contig.slice(0, B, B * T, B).contiguous();
    }

    EXEC_NPU_CMD(aclnnLinearizeCacheIndices, cache_hash_size_cumsum_contig, indices_contig, table_offsets,
                 indices_base_offset, linear_cache_indices);

    return linear_cache_indices;
}

at::Tensor linearize_cache_indices_ascendc(const at::Tensor& cache_hash_size_cumsum, const at::Tensor& indices,
                                           const at::Tensor& offsets, const c10::optional<at::Tensor>& B_offsets,
                                           const int64_t max_B, const int64_t indices_base_offset)
{
    return linearize_cache_indices_npu(cache_hash_size_cumsum, indices, offsets, B_offsets, max_B, indices_base_offset);
}

}  // namespace fbgemm_ascend

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("linearize_cache_indices", &fbgemm_ascend::linearize_cache_indices_npu);
}