/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../../common/pytorch_npu_helper.hpp"
#include "../../common/common_utils.h"
using namespace at;

namespace npu_pruned_hashmap_lookup {

constexpr int64_t INDICES_DIM = 1;
constexpr int64_t OFFSETS_DIM = 1;
constexpr int64_t HASH_TABLE_DIM = 2;
constexpr int64_t HASH_TABLE_OFFSETS_DIM = 1;

bool is_target_dtype(const at::Tensor& tensor, at::ScalarType targetDtype) {
    return tensor.scalar_type() == targetDtype;
}

Tensor npu_pruned_hashmap_lookup_impl(at::Tensor indices, at::Tensor offsets,
                                      at::Tensor hash_table, at::Tensor hash_table_offsets)
{
    if (indices.numel() == 0) {
        return at::zeros_like(indices);
    }

    check_tensor_dim(indices, INDICES_DIM, "pruned_hashmap_lookup indices");
    check_tensor_dim(offsets, OFFSETS_DIM, "pruned_hashmap_lookup offsets");
    check_tensor_dim(hash_table, HASH_TABLE_DIM, "pruned_hashmap_lookup hash_table");
    check_tensor_dim(hash_table_offsets, HASH_TABLE_OFFSETS_DIM, "pruned_hashmap_lookup hash_table_offsets");

    check_tensor_non_empty(indices, "indices");
    check_tensor_non_empty(offsets, "offsets");
    TORCH_CHECK(offsets.numel() > 1, "offsets tensor numel must be greater than 1");

    TORCH_CHECK(is_target_dtype(indices, at::kInt) || is_target_dtype(indices, at::kLong),
                "indices tensor dtype must be int32 or int64");
    TORCH_CHECK(is_target_dtype(indices, offsets.scalar_type()), "indices and offsets dtype must be same");
    TORCH_CHECK(is_target_dtype(hash_table, at::kInt) || is_target_dtype(hash_table, at::kLong),
                "hash_table tensor dtype must be int32 or int64");
    TORCH_CHECK(is_target_dtype(hash_table_offsets, at::kLong), "hash_table_offsets dtype must be int64");

    auto batch_count = offsets.numel() - 1;
    auto table_count = hash_table_offsets.numel() - 1;
    TORCH_CHECK(batch_count % table_count == 0, "the batch count needs to be an integer multiple of the table count");

    auto indices_new = indices.contiguous();
    auto offsets_new = offsets.contiguous();
    auto hash_table_new = hash_table.contiguous();
    auto hash_table_offsets_new = hash_table_offsets.contiguous();
    
    at::Tensor dense_indices = at::empty_like(indices);
    EXEC_NPU_CMD(aclnnPrunedHashmapLookup, indices_new, offsets_new, hash_table_new, hash_table_offsets_new,
                 dense_indices);
    return dense_indices;
}
}  // namespace npu_pruned_hashmap_lookup

TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.impl("pruned_hashmap_lookup",
           torch::dispatch(c10::DispatchKey::PrivateUse1,
                           TORCH_FN(npu_pruned_hashmap_lookup::npu_pruned_hashmap_lookup_impl)));
}
