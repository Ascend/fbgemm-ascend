/**
 * @file lru_cache_populate_byte.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * NPU 适配层：将 torch.ops.fbgemm.lru_cache_populate_byte 派发到 PrivateUse1，
 * 语义对齐 FBGEMM CUDA 路径 lru_cache_populate_byte_cuda（见
 * fbgemm_gpu/src/split_embeddings_cache/lru_cache_populate_byte.cu）。
 *
 * 流程：unique → aclnnLruCacheFindUncached + sort/index_select → aclnnLruCacheInsertByte（AscendC）。
 */
#include <limits>
#include <optional>
#include <tuple>

#include <torch/library.h>

#include "../../common/pytorch_npu_helper.hpp"
#include "../../common/common_utils.h"

using namespace at;

namespace fbgemm_npu {

// NPU entry matching fbgemm_gpu::lru_cache_find_uncached_cuda (lru_cache_find.cu): same parameters
// and return tuple. Find step: aclnnLruCacheFindUncached. Sort step: same key/value pairing as
// INVOKE_CUB_SORT_PAIRS(cache_sets, unique_indices) via stable sort + index_select.
std::tuple<Tensor, Tensor, std::optional<Tensor>> lru_cache_find_uncached_npu(
    Tensor unique_indices,
    Tensor unique_indices_length,
    int64_t max_indices,
    Tensor lxu_cache_state,
    int64_t time_stamp,
    Tensor lru_state,
    bool gather_cache_stats,
    Tensor uvm_cache_stats,
    bool lock_cache_line,
    Tensor lxu_cache_locking_counter,
    const bool compute_inverse_indices)
{
    std::vector<at::Tensor> same_dev = {unique_indices,
        unique_indices_length,
        lxu_cache_state,
        lru_state,
        uvm_cache_stats,
        lxu_cache_locking_counter};
    std::vector<std::string> same_names = {"unique_indices",
        "unique_indices_length",
        "lxu_cache_state",
        "lru_state",
        "uvm_cache_stats",
        "lxu_cache_locking_counter"};
    check_tensor_npu_device(same_dev, same_names);

    TORCH_CHECK(
        lxu_cache_state.is_contiguous(),
        "lxu_cache_state must be contiguous (match CUDA kernel expectation)");
    TORCH_CHECK(lru_state.is_contiguous(), "lru_state must be contiguous");
    TORCH_CHECK(
        uvm_cache_stats.is_contiguous(),
        "uvm_cache_stats must be contiguous");
    TORCH_CHECK(
        lxu_cache_locking_counter.is_contiguous(),
        "lxu_cache_locking_counter must be contiguous");
    TORCH_CHECK(unique_indices.is_contiguous(), "unique_indices must be contiguous");
    TORCH_CHECK(
        unique_indices_length.is_contiguous(),
        "unique_indices_length must be contiguous");
    TORCH_CHECK(
        unique_indices_length.scalar_type() == at::kInt,
        "unique_indices_length must be int32");

    const at::OptionalDeviceGuard device_guard(device_of(unique_indices));

    auto cache_sets = at::full_like(
        unique_indices,
        lxu_cache_state.size(0),
        unique_indices.options().dtype(at::kInt));

    Tensor cache_sets_positions;
    std::optional<Tensor> cache_set_inverse_indices = std::nullopt;
    if (compute_inverse_indices) {
        TORCH_CHECK(
            cache_sets.numel() <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
            "Number of elements in cache_sets is larger than int32_t max");
        cache_sets_positions =
            at::arange({cache_sets.numel()}, cache_sets.options().dtype(at::kInt));
    }

    EXEC_NPU_CMD(aclnnLruCacheFindUncached,
        unique_indices,
        unique_indices_length,
        lxu_cache_state,
        lru_state,
        uvm_cache_stats,
        lxu_cache_locking_counter,
        gather_cache_stats,
        max_indices,
        time_stamp,
        lock_cache_line,
        cache_sets);

    auto sort_keys_vals = at::sort(cache_sets, /*dim=*/0, /*descending=*/false, /*stable=*/true);
    Tensor sorted_cache_sets = std::get<0>(sort_keys_vals);
    Tensor perm = std::get<1>(sort_keys_vals);
    Tensor cache_set_sorted_unique_indices = unique_indices.index_select(0, perm);

    if (compute_inverse_indices) {
        cache_set_inverse_indices = cache_sets_positions.index_select(0, perm);
    }

    return {
        sorted_cache_sets,
        cache_set_sorted_unique_indices,
        cache_set_inverse_indices};
}

// 与 fbgemm_gpu::lru_cache_insert_byte_cuda 参数一致；插入步通过 aclnnLruCacheInsertByte（自定义算子）在设备上完成。
void lru_cache_insert_byte_npu(
    at::Tensor weights,
    at::Tensor cache_hash_size_cumsum,
    at::Tensor cache_index_table_map,
    at::Tensor weights_offsets,
    at::Tensor weights_tys,
    at::Tensor d_offsets,
    at::Tensor sorted_cache_sets,
    at::Tensor cache_set_sorted_unique_indices,
    at::Tensor unique_indices_length,
    at::Tensor lxu_cache_state,
    at::Tensor lxu_cache_weights,
    int64_t time_stamp,
    at::Tensor lru_state,
    bool gather_cache_stats,
    at::Tensor uvm_cache_stats,
    int64_t row_alignment)
{
    std::vector<at::Tensor> same = {weights,
        cache_hash_size_cumsum,
        cache_index_table_map,
        weights_offsets,
        weights_tys,
        d_offsets,
        sorted_cache_sets,
        cache_set_sorted_unique_indices,
        unique_indices_length,
        lxu_cache_state,
        lxu_cache_weights,
        lru_state,
        uvm_cache_stats};
    std::vector<std::string> names = {"weights",
        "cache_hash_size_cumsum",
        "cache_index_table_map",
        "weights_offsets",
        "weights_tys",
        "d_offsets",
        "sorted_cache_sets",
        "cache_set_sorted_unique_indices",
        "unique_indices_length",
        "lxu_cache_state",
        "lxu_cache_weights",
        "lru_state",
        "uvm_cache_stats"};
    check_tensor_npu_device(same, names);

    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(cache_hash_size_cumsum.is_contiguous(), "cache_hash_size_cumsum must be contiguous");
    TORCH_CHECK(cache_index_table_map.is_contiguous(), "cache_index_table_map must be contiguous");
    TORCH_CHECK(weights_offsets.is_contiguous(), "weights_offsets must be contiguous");
    TORCH_CHECK(weights_tys.is_contiguous(), "weights_tys must be contiguous");
    TORCH_CHECK(d_offsets.is_contiguous(), "d_offsets must be contiguous");
    TORCH_CHECK(sorted_cache_sets.is_contiguous(), "sorted_cache_sets must be contiguous");
    TORCH_CHECK(
        cache_set_sorted_unique_indices.is_contiguous(),
        "cache_set_sorted_unique_indices must be contiguous");
    TORCH_CHECK(unique_indices_length.is_contiguous(), "unique_indices_length must be contiguous");
    TORCH_CHECK(lxu_cache_state.is_contiguous(), "lxu_cache_state must be contiguous");
    TORCH_CHECK(lxu_cache_weights.is_contiguous(), "lxu_cache_weights must be contiguous");
    TORCH_CHECK(lru_state.is_contiguous(), "lru_state must be contiguous");
    TORCH_CHECK(uvm_cache_stats.is_contiguous(), "uvm_cache_stats must be contiguous");
    TORCH_CHECK(
        unique_indices_length.scalar_type() == at::kInt,
        "unique_indices_length must be int32");

    const at::OptionalDeviceGuard device_guard(device_of(weights));
    auto reserved_out = at::zeros({1}, weights.options().dtype(at::kInt));

    EXEC_NPU_CMD(aclnnLruCacheInsertByte,
        weights,
        cache_hash_size_cumsum,
        cache_index_table_map,
        weights_offsets,
        weights_tys,
        d_offsets,
        sorted_cache_sets,
        cache_set_sorted_unique_indices,
        unique_indices_length,
        lxu_cache_state,
        lxu_cache_weights,
        lru_state,
        uvm_cache_stats,
        gather_cache_stats,
        time_stamp,
        row_alignment,
        reserved_out);
}

// 签名须与 fbgemm_gpu::lru_cache_populate_byte_cpu 一致（按值 Tensor + std::optional），
// 否则 TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1) 会与 CPU 核 C++ 签名不匹配而注册失败。
void lru_cache_populate_byte_impl_npu(
    at::Tensor weights,
    at::Tensor hash_size_cumsum,
    int64_t total_cache_hash_size,
    at::Tensor cache_index_table_map,
    at::Tensor weights_offsets,
    at::Tensor weights_tys,
    at::Tensor d_offsets,
    at::Tensor linear_cache_indices,
    at::Tensor lxu_cache_state,
    at::Tensor lxu_cache_weights,
    int64_t time_stamp,
    at::Tensor lru_state,
    int64_t row_alignment,
    bool gather_cache_stats,
    std::optional<at::Tensor> uvm_cache_stats)
{
    TORCH_CHECK(
        linear_cache_indices.numel() < std::numeric_limits<int32_t>::max(),
        "linear_cache_indices.numel() must fit int32");
    if (linear_cache_indices.numel() == 0) {
        return;
    }

    if (gather_cache_stats) {
        TORCH_CHECK(
            uvm_cache_stats.has_value() && uvm_cache_stats.value().defined(),
            "gather_cache_stats=True requires uvm_cache_stats tensor");
    }

    at::Tensor uvm_stats_tensor =
        (uvm_cache_stats.has_value() && uvm_cache_stats.value().defined())
            ? uvm_cache_stats.value()
            : at::empty({0}, weights.options().dtype(at::kInt));

    std::vector<at::Tensor> tensors = {weights,
        hash_size_cumsum,
        cache_index_table_map,
        weights_offsets,
        weights_tys,
        d_offsets,
        linear_cache_indices,
        lxu_cache_state,
        lxu_cache_weights,
        lru_state};
    std::vector<std::string> names = {"weights",
        "hash_size_cumsum",
        "cache_index_table_map",
        "weights_offsets",
        "weights_tys",
        "D_offsets",
        "linear_cache_indices",
        "lxu_cache_state",
        "lxu_cache_weights",
        "lru_state"};
    if (gather_cache_stats) {
        tensors.push_back(uvm_stats_tensor);
        names.push_back("uvm_cache_stats");
    }
    check_tensor_npu_device(tensors, names);

    TORCH_CHECK(
        lxu_cache_state.is_contiguous(),
        "lxu_cache_state must be contiguous");
    TORCH_CHECK(lru_state.is_contiguous(), "lru_state must be contiguous");
    TORCH_CHECK(
        uvm_stats_tensor.is_contiguous(),
        "uvm_cache_stats must be contiguous");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(hash_size_cumsum.is_contiguous(), "hash_size_cumsum must be contiguous");
    TORCH_CHECK(cache_index_table_map.is_contiguous(), "cache_index_table_map must be contiguous");
    TORCH_CHECK(weights_offsets.is_contiguous(), "weights_offsets must be contiguous");
    TORCH_CHECK(weights_tys.is_contiguous(), "weights_tys must be contiguous");
    TORCH_CHECK(d_offsets.is_contiguous(), "D_offsets must be contiguous");
    TORCH_CHECK(lxu_cache_weights.is_contiguous(), "lxu_cache_weights must be contiguous");

    const at::OptionalDeviceGuard guard(device_of(weights));

    auto lin_c = linear_cache_indices.contiguous();
    at::Tensor unique_indices = std::get<0>(at::_unique(lin_c));
    const int32_t unique_len_i32 = static_cast<int32_t>(unique_indices.size(0));
    at::Tensor unique_indices_length_tensor = at::tensor(
        {unique_len_i32},
        at::TensorOptions(lin_c.device()).dtype(at::kInt));

    Tensor lxu_cache_locking_counter =
        at::empty({0, 0}, lxu_cache_state.options().dtype(at::kInt));
    const auto find_uncached_out = lru_cache_find_uncached_npu(
        unique_indices,
        unique_indices_length_tensor,
        total_cache_hash_size,
        lxu_cache_state,
        time_stamp,
        lru_state,
        gather_cache_stats,
        uvm_stats_tensor,
        /*lock_cache_line=*/false,
        lxu_cache_locking_counter,
        /*compute_inverse_indices=*/false);

    lru_cache_insert_byte_npu(
        weights,
        hash_size_cumsum,
        cache_index_table_map,
        weights_offsets,
        weights_tys,
        d_offsets,
        std::get<0>(find_uncached_out),
        std::get<1>(find_uncached_out),
        unique_indices_length_tensor,
        lxu_cache_state,
        lxu_cache_weights,
        time_stamp,
        lru_state,
        gather_cache_stats,
        uvm_stats_tensor,
        row_alignment);
}

} // namespace fbgemm_npu

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def(
        "lru_cache_populate_byte(Tensor weights, Tensor hash_size_cumsum, int total_cache_hash_size, "
        "Tensor cache_index_table_map, Tensor weights_offsets, Tensor weights_tys, Tensor D_offsets, "
        "Tensor linear_cache_indices, Tensor(a!) lxu_cache_state, Tensor(b!) lxu_cache_weights, "
        "int time_stamp, Tensor(c!) lru_state, int row_alignment=16, bool gather_cache_stats=False, "
        "Tensor(d!)? uvm_cache_stats=None) -> ()");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("lru_cache_populate_byte", TORCH_FN(fbgemm_npu::lru_cache_populate_byte_impl_npu));
}

// 复用 fbgemm 已有 schema（fbgemm_gpu split_embeddings_cache_ops），仅注册 NPU 实现
TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("lru_cache_populate_byte", TORCH_FN(fbgemm_npu::lru_cache_populate_byte_impl_npu));
}
