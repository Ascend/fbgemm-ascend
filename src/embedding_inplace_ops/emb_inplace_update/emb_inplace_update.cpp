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

#include "../../common/common_utils.h"
#include "../../common/pytorch_npu_helper.hpp"

using namespace at;

namespace {
constexpr int64_t EXPECTED_DIM_1D = 1;
}  // namespace

/**
 * 注意：昇腾 SIMT 暂不支持 UVM 特性。kernel 仅处理 placement == DEVICE(0)
 * 的记录，非 DEVICE 的记录会被静默跳过（不报错、不写入）。
 * uvm_weights / lxu_cache_* 参数仅为对齐 fbgemm schema 而保留。
 */
void emb_inplace_update_impl_npu(at::Tensor dev_weights, at::Tensor uvm_weights, at::Tensor weights_placements,
                                 at::Tensor weights_offsets, at::Tensor weights_tys, at::Tensor D_offsets,
                                 at::Tensor update_weights, at::Tensor update_table_indices,
                                 at::Tensor update_row_indices, at::Tensor update_offsets, int64_t row_alignment,
                                 c10::optional<at::Tensor> /*lxu_cache_weights*/,
                                 c10::optional<at::Tensor> /*lxu_cache_locations*/)
{
    int64_t N = update_row_indices.numel();
    if (N == 0) {
        return;
    }

    std::vector<at::Tensor> tensors = {dev_weights,        uvm_weights,   weights_placements, weights_offsets,
                                       weights_tys,        D_offsets,     update_weights,     update_table_indices,
                                       update_row_indices, update_offsets};
    std::vector<std::string> names = {
        "dev_weights", "uvm_weights",    "weights_placements",   "weights_offsets",    "weights_tys",
        "D_offsets",   "update_weights", "update_table_indices", "update_row_indices", "update_offsets"};
    check_tensor_npu_device(tensors, names);

    TORCH_CHECK(dev_weights.scalar_type() == at::kByte, "dev_weights must be uint8");
    TORCH_CHECK(update_weights.scalar_type() == at::kByte, "update_weights must be uint8");
    TORCH_CHECK(weights_tys.scalar_type() == at::kByte, "weights_tys must be uint8");
    TORCH_CHECK(weights_placements.scalar_type() == at::kInt, "weights_placements must be int32");
    TORCH_CHECK(D_offsets.scalar_type() == at::kInt, "D_offsets must be int32");
    TORCH_CHECK(weights_offsets.scalar_type() == at::kLong, "weights_offsets must be int64");
    TORCH_CHECK(update_offsets.scalar_type() == at::kLong, "update_offsets must be int64");
    TORCH_CHECK(update_table_indices.scalar_type() == at::kInt, "update_table_indices must be int32");
    TORCH_CHECK(update_row_indices.scalar_type() == at::kInt || update_row_indices.scalar_type() == at::kLong,
                "update_row_indices must be int32 or int64");

    TORCH_CHECK(update_row_indices.dim() == EXPECTED_DIM_1D, "update_row_indices must be 1D");
    TORCH_CHECK(update_table_indices.dim() == EXPECTED_DIM_1D, "update_table_indices must be 1D");
    TORCH_CHECK(update_table_indices.numel() == N,
                "update_table_indices and update_row_indices must have the same "
                "length, got ",
                update_table_indices.numel(), " vs ", N);
    TORCH_CHECK(update_offsets.numel() == N + 1 || update_offsets.numel() == N,
                "update_offsets length must be N or N+1, got ", update_offsets.numel());

    auto dev_w = dev_weights.contiguous();
    auto uvm_w = uvm_weights.contiguous();
    auto placements = weights_placements.contiguous();
    auto w_offsets = weights_offsets.contiguous();
    auto w_tys = weights_tys.contiguous();
    auto d_offsets = D_offsets.contiguous();
    auto upd_w = update_weights.contiguous();
    auto upd_t_idx = update_table_indices.contiguous();
    auto upd_r_idx = update_row_indices.contiguous();
    auto upd_offsets = update_offsets.contiguous();

    EXEC_NPU_CMD(aclnnEmbInplaceUpdate, dev_w, uvm_w, placements, w_offsets, w_tys, d_offsets, upd_w, upd_t_idx,
                 upd_r_idx, upd_offsets, row_alignment);

    // 若 contiguous() 产生了新 tensor，需写回原 tensor 以保持 in-place 语义。
    if (!dev_w.is_same(dev_weights)) {
        dev_weights.copy_(dev_w);
    }
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("emb_inplace_update", &emb_inplace_update_impl_npu);
}
