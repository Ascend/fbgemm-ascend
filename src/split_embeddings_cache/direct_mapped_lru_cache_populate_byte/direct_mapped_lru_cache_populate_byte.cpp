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
#include <limits>
#include <optional>
#include <tuple>

#include <torch/library.h>

#include "../../common/pytorch_npu_helper.hpp"
#include "../../common/common_utils.h"

using namespace at;

// find_uncached NPU 适配层：调用 aclnnDirectMappedLruCacheFindUncached 查找未命中项
Tensor direct_mapped_lru_cache_find_uncached_npu(Tensor linear_cache_indices, Tensor lxu_cache_state, Tensor lru_state,
                                                 Tensor lxu_cache_miss_timestamp, Tensor uvm_cache_stats,
                                                 int64_t max_indices, int64_t time_stamp, bool gather_cache_stats)
{
    // 设备一致性校验
    std::vector<Tensor> same_dev = {linear_cache_indices, lxu_cache_state, lru_state, lxu_cache_miss_timestamp,
                                    uvm_cache_stats};
    std::vector<std::string> same_names = {"linear_cache_indices", "lxu_cache_state", "lru_state",
                                           "lxu_cache_miss_timestamp", "uvm_cache_stats"};
    check_tensor_npu_device(same_dev, same_names);

    // 连续性校验
    TORCH_CHECK(linear_cache_indices.is_contiguous(), "linear_cache_indices must be contiguous");
    TORCH_CHECK(lxu_cache_state.is_contiguous(), "lxu_cache_state must be contiguous (match CUDA kernel expectation)");
    TORCH_CHECK(lru_state.is_contiguous(), "lru_state must be contiguous");
    TORCH_CHECK(lxu_cache_miss_timestamp.is_contiguous(), "lxu_cache_miss_timestamp must be contiguous");
    TORCH_CHECK(uvm_cache_stats.is_contiguous(), "uvm_cache_stats must be contiguous");

    // 设备保护
    const OptionalDeviceGuard device_guard(device_of(linear_cache_indices));

    // 预分配输出张量，默认值 -1（对应 CUDA 的 -1 哨兵）
    auto cache_sets = full_like(linear_cache_indices, -1, linear_cache_indices.options().dtype(kInt));

    // 调用 aclnn 算子
    EXEC_NPU_CMD(aclnnDirectMappedLruCacheFindUncached, linear_cache_indices, lxu_cache_state, lru_state,
                 lxu_cache_miss_timestamp, uvm_cache_stats, max_indices, time_stamp, gather_cache_stats, cache_sets);

    return cache_sets;
}

// insert_byte NPU 适配层：调用 aclnnDirectMappedLruCacheInsertByte 将 UVM 权重写入 Cache
void direct_mapped_lru_cache_insert_byte_npu(Tensor weights, Tensor cache_hash_size_cumsum,
                                             Tensor cache_index_table_map, Tensor weights_offsets, Tensor weights_tys,
                                             Tensor d_offsets, Tensor lxu_cache_state, Tensor lxu_cache_weights,
                                             Tensor lru_state, Tensor linear_cache_indices,
                                             Tensor lxu_cache_miss_timestamp, Tensor cache_sets,
                                             bool gather_cache_stats, Tensor uvm_cache_stats, int64_t time_stamp,
                                             int64_t row_alignment)
{
    std::vector<Tensor> same = {weights,
                                cache_hash_size_cumsum,
                                cache_index_table_map,
                                weights_offsets,
                                weights_tys,
                                d_offsets,
                                lxu_cache_state,
                                lxu_cache_weights,
                                lru_state,
                                linear_cache_indices,
                                lxu_cache_miss_timestamp,
                                cache_sets,
                                uvm_cache_stats};
    std::vector<std::string> names = {"weights",
                                      "cache_hash_size_cumsum",
                                      "cache_index_table_map",
                                      "weights_offsets",
                                      "weights_tys",
                                      "d_offsets",
                                      "lxu_cache_state",
                                      "lxu_cache_weights",
                                      "lru_state",
                                      "linear_cache_indices",
                                      "lxu_cache_miss_timestamp",
                                      "cache_sets",
                                      "uvm_cache_stats"};
    check_tensor_npu_device(same, names);

    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(cache_hash_size_cumsum.is_contiguous(), "cache_hash_size_cumsum must be contiguous");
    TORCH_CHECK(cache_index_table_map.is_contiguous(), "cache_index_table_map must be contiguous");
    TORCH_CHECK(weights_offsets.is_contiguous(), "weights_offsets must be contiguous");
    TORCH_CHECK(weights_tys.is_contiguous(), "weights_tys must be contiguous");
    TORCH_CHECK(d_offsets.is_contiguous(), "d_offsets must be contiguous");
    TORCH_CHECK(lxu_cache_state.is_contiguous(), "lxu_cache_state must be contiguous");
    TORCH_CHECK(lxu_cache_weights.is_contiguous(), "lxu_cache_weights must be contiguous");
    TORCH_CHECK(lru_state.is_contiguous(), "lru_state must be contiguous");
    TORCH_CHECK(linear_cache_indices.is_contiguous(), "linear_cache_indices must be contiguous");
    TORCH_CHECK(lxu_cache_miss_timestamp.is_contiguous(), "lxu_cache_miss_timestamp must be contiguous");
    TORCH_CHECK(cache_sets.is_contiguous(), "cache_sets must be contiguous");
    TORCH_CHECK(uvm_cache_stats.is_contiguous(), "uvm_cache_stats must be contiguous");

    const OptionalDeviceGuard device_guard(device_of(weights));
    auto reserved_out = zeros({1}, weights.options().dtype(kInt));

    EXEC_NPU_CMD(aclnnDirectMappedLruCacheInsertByte, weights, cache_hash_size_cumsum, cache_index_table_map,
                 weights_offsets, weights_tys, d_offsets, lxu_cache_state, lxu_cache_weights, lru_state,
                 linear_cache_indices, lxu_cache_miss_timestamp, cache_sets, uvm_cache_stats, gather_cache_stats,
                 time_stamp, row_alignment, reserved_out);
}

// populate_byte 主入口：组合 find_uncached + insert_byte 完成 direct‑mapped LRU 缓存填充
void direct_mapped_lru_cache_populate_byte_impl_npu(Tensor weights, Tensor hash_size_cumsum,
                                                    int64_t total_cache_hash_size, Tensor cache_index_table_map,
                                                    Tensor weights_offsets, Tensor weights_tys, Tensor d_offsets,
                                                    Tensor linear_cache_indices, Tensor lxu_cache_state,
                                                    Tensor lxu_cache_weights, int64_t time_stamp, Tensor lru_state,
                                                    Tensor lxu_cache_miss_timestamp, int64_t row_alignment,
                                                    bool gather_cache_stats, std::optional<Tensor> uvm_cache_stats)
{
    // N 值边界保护：索引数须在 int32 范围内
    TORCH_CHECK(linear_cache_indices.numel() < std::numeric_limits<int32_t>::max(),
                "linear_cache_indices.numel() must fit int32");
    if (linear_cache_indices.numel() == 0) {
        return;
    }

    // 统计模式校验：gather_cache_stats 为 true 时必须提供 uvm_cache_stats
    if (gather_cache_stats) {
        TORCH_CHECK(uvm_cache_stats.has_value() && uvm_cache_stats.value().defined(),
                    "gather_cache_stats=True requires uvm_cache_stats tensor");
    }

    // UVM 统计张量：有值则使用，无值则创建空张量占位
    Tensor uvm_stats_tensor = (uvm_cache_stats.has_value() && uvm_cache_stats.value().defined())
                                  ? uvm_cache_stats.value()
                                  : empty({0}, weights.options().dtype(kInt));

    // 设备一致性校验：确保所有输入张量在同一 NPU 设备上
    std::vector<Tensor> tensors = {weights,           hash_size_cumsum, cache_index_table_map,   weights_offsets,
                                   weights_tys,       d_offsets,        linear_cache_indices,    lxu_cache_state,
                                   lxu_cache_weights, lru_state,        lxu_cache_miss_timestamp};
    std::vector<std::string> names = {
        "weights",           "hash_size_cumsum", "cache_index_table_map",   "weights_offsets",
        "weights_tys",       "d_offsets",        "linear_cache_indices",    "lxu_cache_state",
        "lxu_cache_weights", "lru_state",        "lxu_cache_miss_timestamp"};
    if (gather_cache_stats) {
        tensors.push_back(uvm_stats_tensor);
        names.push_back("uvm_cache_stats");
    }
    check_tensor_npu_device(tensors, names);

    // 连续性校验
    TORCH_CHECK(lxu_cache_state.is_contiguous(), "lxu_cache_state must be contiguous");
    TORCH_CHECK(lru_state.is_contiguous(), "lru_state must be contiguous");
    TORCH_CHECK(uvm_stats_tensor.is_contiguous(), "uvm_cache_stats must be contiguous");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(hash_size_cumsum.is_contiguous(), "hash_size_cumsum must be contiguous");
    TORCH_CHECK(cache_index_table_map.is_contiguous(), "cache_index_table_map must be contiguous");
    TORCH_CHECK(weights_offsets.is_contiguous(), "weights_offsets must be contiguous");
    TORCH_CHECK(weights_tys.is_contiguous(), "weights_tys must be contiguous");
    TORCH_CHECK(d_offsets.is_contiguous(), "d_offsets must be contiguous");
    TORCH_CHECK(lxu_cache_weights.is_contiguous(), "lxu_cache_weights must be contiguous");
    TORCH_CHECK(lxu_cache_miss_timestamp.is_contiguous(), "lxu_cache_miss_timestamp must be contiguous");

    // 设备保护
    const OptionalDeviceGuard guard(device_of(weights));

    // Step 1：查找未命中项，获得 cache_sets 分配
    Tensor cache_sets = direct_mapped_lru_cache_find_uncached_npu(
        linear_cache_indices, lxu_cache_state, lru_state, lxu_cache_miss_timestamp, uvm_stats_tensor,
        total_cache_hash_size, time_stamp, gather_cache_stats);

    // Step 2：将 UVM 中 miss 的权重数据写入 Cache
    direct_mapped_lru_cache_insert_byte_npu(weights, hash_size_cumsum, cache_index_table_map, weights_offsets,
                                            weights_tys, d_offsets, lxu_cache_state, lxu_cache_weights, lru_state,
                                            linear_cache_indices, lxu_cache_miss_timestamp, cache_sets,
                                            gather_cache_stats, uvm_stats_tensor, time_stamp, row_alignment);
}

// 注册 fbgemm 算子 NPU 后端实现
TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("direct_mapped_lru_cache_populate_byte", &direct_mapped_lru_cache_populate_byte_impl_npu);
}
