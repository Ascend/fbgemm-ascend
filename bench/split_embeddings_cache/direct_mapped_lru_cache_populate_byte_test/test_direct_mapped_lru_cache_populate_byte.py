#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for direct_mapped_lru_cache_populate_byte operator (NPU).
Parametrized version for clear pass/fail visualization.
"""

import random
import pytest
import torch
import torch_npu
import fbgemm_ascend  # noqa: F401
import fbgemm_gpu  # noqa: F401

# ---------- 环境配置 ----------
DEVICE = "npu:0"
torch.npu.config.allow_internal_format = False
if torch.npu.is_available():
    torch_npu.npu.set_device(DEVICE)
else:
    raise RuntimeError("NPU is not available")

random.seed(10000)
torch.manual_seed(10000)


# ---------- 辅助函数 ----------
def cache_slot(idx, C):
    """MurmurHash3 64‑bit 哈希函数，与 C++ 实现一致"""
    h = idx & 0xFFFFFFFFFFFFFFFF
    h ^= (h >> 33) & 0xFFFFFFFFFFFFFFFF
    h = (h * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    h ^= (h >> 33) & 0xFFFFFFFFFFFFFFFF
    h = (h * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
    h ^= (h >> 33) & 0xFFFFFFFFFFFFFFFF
    return h % C


def padded_row_size_in_bytes(dim, weight_ty, row_alignment, scale_bias_bytes=4):
    """计算对齐后的行大小（字节）"""
    if weight_ty == 0:  # FP32
        raw_size = dim * 4
    elif weight_ty in (1, 5):  # FP16, BF16
        raw_size = dim * 2
    elif weight_ty == 6:  # FP8
        raw_size = dim
    elif weight_ty == 2:  # INT8
        raw_size = dim + scale_bias_bytes
    elif weight_ty == 3:  # INT4
        raw_size = dim // 2 + scale_bias_bytes
    elif weight_ty == 4:  # INT2
        raw_size = dim // 4 + scale_bias_bytes
    else:
        raise ValueError(f"Unknown weight_ty: {weight_ty}")
    return ((raw_size + row_alignment - 1) // row_alignment) * row_alignment


def call_operator(
    weights,
    hash_size_cumsum,
    total_cache_hash_size,
    cache_index_table_map,
    weights_offsets,
    weights_tys,
    d_offsets,
    linear_cache_indices,
    lxu_cache_state,
    lxu_cache_weights,
    time_stamp,
    lru_state,
    lxu_cache_miss_timestamp,
    row_alignment=16,
    gather_cache_stats=False,
    uvm_cache_stats=None,
):
    """封装算子调用，同步后返回 CPU 端结果字典"""
    if uvm_cache_stats is None:
        uvm_cache_stats = torch.empty(0, dtype=torch.int32, device=DEVICE)
    torch.ops.fbgemm.direct_mapped_lru_cache_populate_byte(
        weights,
        hash_size_cumsum,
        total_cache_hash_size,
        cache_index_table_map,
        weights_offsets,
        weights_tys,
        d_offsets,
        linear_cache_indices,
        lxu_cache_state,
        lxu_cache_weights,
        time_stamp,
        lru_state,
        lxu_cache_miss_timestamp,
        row_alignment,
        gather_cache_stats,
        uvm_cache_stats,
    )
    torch.npu.synchronize()
    return {
        "lxu_cache_state": lxu_cache_state.cpu(),
        "lxu_cache_weights": lxu_cache_weights.cpu(),
        "lru_state": lru_state.cpu(),
        "lxu_cache_miss_timestamp": lxu_cache_miss_timestamp.cpu(),
        "uvm_cache_stats": uvm_cache_stats.cpu() if gather_cache_stats else None,
    }


def _make_cache_state(C, fill=-1):
    return torch.full((C, 1), fill, dtype=torch.int64, device=DEVICE)


def _make_lru_state(C, fill=0):
    return torch.full((C, 1), fill, dtype=torch.int64, device=DEVICE)


def _make_miss_timestamp(C, fill=-1):
    return torch.full((C, 1), fill, dtype=torch.int64, device=DEVICE)


def _generate_cache_params(num_tables, D, row_bytes, total_hash_size):
    """根据总哈希空间生成所有辅助张量"""
    rows_per_table = total_hash_size // num_tables
    total_actual = rows_per_table * num_tables
    hash_cumsum = torch.zeros(num_tables, dtype=torch.int64)
    for t in range(num_tables):
        hash_cumsum[t] = t * rows_per_table
    index_map = torch.zeros(total_actual, dtype=torch.int32)
    for t in range(num_tables):
        index_map[t * rows_per_table : (t + 1) * rows_per_table] = t
    w_offsets = torch.zeros(num_tables, dtype=torch.int64)
    for t in range(num_tables):
        w_offsets[t] = t * rows_per_table * row_bytes
    w_tys = torch.zeros(num_tables, dtype=torch.uint8)
    d_offs = torch.zeros(num_tables + 1, dtype=torch.int32)
    for t in range(num_tables + 1):
        d_offs[t] = t * D
    return hash_cumsum, index_map, w_offsets, w_tys, d_offs, total_actual, rows_per_table


def _build_weights(num_tables, rows_per_table, row_bytes):
    """构造确定性的权重数据，用于验证"""
    total_rows = num_tables * rows_per_table
    weights = torch.zeros(total_rows * row_bytes, dtype=torch.uint8)
    for t in range(num_tables):
        for r in range(rows_per_table):
            for b in range(row_bytes):
                idx = t * rows_per_table * row_bytes + r * row_bytes + b
                weights[idx] = (t * 10000 + r * 100 + b) % 256
    return weights


def _get_weights_offset_and_row(weights, table, idx_within_table, row_bytes, w_offsets):
    base = w_offsets[table].item() + idx_within_table * row_bytes
    return weights[base : base + row_bytes]


def _generate_non_colliding_indices(C, n_indices, start_idx=1):
    """
    生成 n_indices 个映射到不同缓存槽的索引。
    不限制搜索上限，保证一定成功（不会失败）。
    """
    indices, seen = [], set()
    idx = start_idx
    while len(indices) < n_indices:
        s = cache_slot(idx, C)
        if s not in seen:
            indices.append(idx)
            seen.add(s)
        idx += 1
    return torch.tensor(indices, dtype=torch.int64)


# ---------- 测试参数域 ----------
C_list = [4, 8, 16, 32, 64]
D_list = [8, 16, 32]
num_tables_list = [1, 2]
weight_ty_list = [0, 1, 2, 3, 4, 5, 6]
indices_dtype_list = [torch.int32, torch.int64]


def generate_param_list(num_samples=10):
    """生成 num_samples 组随机参数 (N, C, D, num_tables, weight_ty, indices_dtype)"""
    param_list = []
    for _ in range(num_samples):
        C = random.choice(C_list)
        D = random.choice(D_list)
        num_tables = random.choice(num_tables_list)
        weight_ty = random.choice(weight_ty_list)
        indices_dtype = random.choice(indices_dtype_list)
        max_N = min(10, C)
        N = random.randint(1, max_N) if max_N > 0 else 1
        param_list.append((N, C, D, num_tables, weight_ty, indices_dtype))
    return param_list


# 冲突测试要求 C >= 4
conflict_C = [c for c in C_list if c >= 4]


def generate_conflict_param_list(num_samples=10):
    param_list = []
    for _ in range(num_samples):
        C = random.choice(conflict_C)
        D = random.choice(D_list)
        num_tables = random.choice(num_tables_list)
        weight_ty = random.choice(weight_ty_list)
        indices_dtype = random.choice(indices_dtype_list)
        N = random.randint(1, 5)
        param_list.append((N, C, D, num_tables, weight_ty, indices_dtype))
    return param_list


# 统计测试要求 C >= 16
stats_C = [c for c in C_list if c >= 16]


def generate_stats_param_list(num_samples=8):
    param_list = []
    for _ in range(num_samples):
        C = random.choice(stats_C)
        D = random.choice(D_list)
        num_tables = random.choice(num_tables_list)
        weight_ty = random.choice(weight_ty_list)
        indices_dtype = random.choice(indices_dtype_list)
        param_list.append((C, D, num_tables, weight_ty, indices_dtype))
    return param_list


# 预生成参数列表
BASIC_PARAMS = generate_param_list(10)
ALL_HIT_PARAMS = generate_param_list(10)
CONFLICT_PARAMS = generate_conflict_param_list(10)
EMPTY_PARAMS = generate_param_list(8)
REPEATED_PARAMS = generate_param_list(8)
STATS_PARAMS = generate_stats_param_list(8)


# ---------- 测试类 ----------
class TestDirectMappedLRUCache:
    @pytest.mark.parametrize("N, C, D, num_tables, weight_ty, indices_dtype", BASIC_PARAMS)
    def test_basic_populate(self, N, C, D, num_tables, weight_ty, indices_dtype):
        """基本插入：空缓存插入不冲突的索引，验证状态和权重数据"""
        n_indices = min(N, C)
        indices = _generate_non_colliding_indices(C, n_indices)
        if indices_dtype == torch.int32:
            indices = indices.to(torch.int32)

        min_hash_size = num_tables * C
        total_hash_size = max(int(indices.max().item()) + 1, min_hash_size)

        if total_hash_size % num_tables != 0:
            total_hash_size = ((total_hash_size // num_tables) + 1) * num_tables

        row_bytes = padded_row_size_in_bytes(D, weight_ty, 16)
        hash_cumsum_cpu, index_map_cpu, w_offsets_cpu, w_tys_cpu, d_offs_cpu, total_hash, rows_per_table = (
            _generate_cache_params(num_tables, D, row_bytes, total_hash_size)
        )
        w_tys_cpu[:] = weight_ty
        weights_cpu = _build_weights(num_tables, rows_per_table, row_bytes)

        cache_state = _make_cache_state(C, -1)
        cache_weights = torch.zeros(C, row_bytes, dtype=torch.uint8, device=DEVICE)
        lru_state = _make_lru_state(C, 0)
        miss_ts = _make_miss_timestamp(C, -1)

        result = call_operator(
            weights_cpu.to(DEVICE),
            hash_cumsum_cpu.to(DEVICE),
            total_hash,
            index_map_cpu.to(DEVICE),
            w_offsets_cpu.to(DEVICE),
            w_tys_cpu.to(DEVICE),
            d_offs_cpu.to(DEVICE),
            indices.to(DEVICE),
            cache_state,
            cache_weights,
            100,
            lru_state,
            miss_ts,
            row_alignment=16,
            gather_cache_stats=False,
        )

        cache_state_cpu = result["lxu_cache_state"]
        cache_weights_cpu = result["lxu_cache_weights"]
        for idx in indices:
            s = cache_slot(idx.item(), C)
            assert cache_state_cpu[s].item() == idx.item()
            table = index_map_cpu[idx.item()].item()
            local_idx = idx.item() - hash_cumsum_cpu[table].item()
            expected_row = _get_weights_offset_and_row(weights_cpu, table, local_idx, row_bytes, w_offsets_cpu)
            assert torch.equal(cache_weights_cpu[s], expected_row)

    @pytest.mark.parametrize("N, C, D, num_tables, weight_ty, indices_dtype", ALL_HIT_PARAMS)
    def test_all_hit_no_modification(self, N, C, D, num_tables, weight_ty, indices_dtype):
        """全部命中：两次相同索引，第二次应命中且只更新时间戳"""
        n_indices = min(N, C)
        indices = _generate_non_colliding_indices(C, n_indices)
        if indices_dtype == torch.int32:
            indices = indices.to(torch.int32)

        # 动态设定总哈希空间，但确保足够大避免多表时越界
        min_hash_size = num_tables * C
        total_hash_size = max(int(indices.max().item()) + 1, min_hash_size)
        if total_hash_size % num_tables != 0:
            total_hash_size = ((total_hash_size // num_tables) + 1) * num_tables

        row_bytes = padded_row_size_in_bytes(D, weight_ty, 16)
        hash_cumsum_cpu, index_map_cpu, w_offsets_cpu, w_tys_cpu, d_offs_cpu, total_hash, rows_per_table = (
            _generate_cache_params(num_tables, D, row_bytes, total_hash_size)
        )
        w_tys_cpu[:] = weight_ty
        weights_cpu = _build_weights(num_tables, rows_per_table, row_bytes)

        cache_state = _make_cache_state(C, -1)
        cache_weights = torch.zeros(C, row_bytes, dtype=torch.uint8, device=DEVICE)
        lru_state = _make_lru_state(C, 0)
        miss_ts = _make_miss_timestamp(C, -1)

        result1 = call_operator(
            weights_cpu.to(DEVICE),
            hash_cumsum_cpu.to(DEVICE),
            total_hash,
            index_map_cpu.to(DEVICE),
            w_offsets_cpu.to(DEVICE),
            w_tys_cpu.to(DEVICE),
            d_offs_cpu.to(DEVICE),
            indices.to(DEVICE),
            cache_state,
            cache_weights,
            100,
            lru_state,
            miss_ts,
            row_alignment=16,
            gather_cache_stats=False,
        )
        result2 = call_operator(
            weights_cpu.to(DEVICE),
            hash_cumsum_cpu.to(DEVICE),
            total_hash,
            index_map_cpu.to(DEVICE),
            w_offsets_cpu.to(DEVICE),
            w_tys_cpu.to(DEVICE),
            d_offs_cpu.to(DEVICE),
            indices.to(DEVICE),
            cache_state,
            cache_weights,
            200,
            lru_state,
            miss_ts,
            row_alignment=16,
            gather_cache_stats=False,
        )

        assert torch.equal(result1["lxu_cache_state"], result2["lxu_cache_state"])
        assert torch.equal(result1["lxu_cache_weights"], result2["lxu_cache_weights"])
        for idx in indices:
            s = cache_slot(idx.item(), C)
            assert result2["lru_state"][s].item() == 200

    @pytest.mark.parametrize("N, C, D, num_tables, weight_ty, indices_dtype", CONFLICT_PARAMS)
    def test_conflict_resolution(self, N, C, D, num_tables, weight_ty, indices_dtype):
        """冲突解决：多个索引映射到同一槽，原子操作应保证恰好一个被插入"""
        target_set = 0
        indices = []
        idx = 1

        while len(indices) < min(N, 5):
            if cache_slot(idx, C) == target_set:
                indices.append(idx)
            idx += 1
        indices_t = torch.tensor(indices, dtype=torch.int64)
        if indices_dtype == torch.int32:
            indices_t = indices_t.to(torch.int32)

        min_hash_size = num_tables * C
        total_hash_size = max(int(indices_t.max().item()) + 1, min_hash_size)
        if total_hash_size % num_tables != 0:
            total_hash_size = ((total_hash_size // num_tables) + 1) * num_tables

        row_bytes = padded_row_size_in_bytes(D, weight_ty, 16)
        hash_cumsum_cpu, index_map_cpu, w_offsets_cpu, w_tys_cpu, d_offs_cpu, total_hash, rows_per_table = (
            _generate_cache_params(num_tables, D, row_bytes, total_hash_size)
        )
        w_tys_cpu[:] = weight_ty
        weights_cpu = _build_weights(num_tables, rows_per_table, row_bytes)

        cache_state = _make_cache_state(C, -1)
        cache_weights = torch.zeros(C, row_bytes, dtype=torch.uint8, device=DEVICE)
        lru_state = _make_lru_state(C, 0)
        miss_ts = _make_miss_timestamp(C, -1)

        result = call_operator(
            weights_cpu.to(DEVICE),
            hash_cumsum_cpu.to(DEVICE),
            total_hash,
            index_map_cpu.to(DEVICE),
            w_offsets_cpu.to(DEVICE),
            w_tys_cpu.to(DEVICE),
            d_offs_cpu.to(DEVICE),
            indices_t.to(DEVICE),
            cache_state,
            cache_weights,
            100,
            lru_state,
            miss_ts,
            row_alignment=16,
            gather_cache_stats=False,
        )

        cached_idx = result["lxu_cache_state"][target_set].item()
        assert cached_idx != -1
        assert cached_idx in indices
        table = index_map_cpu[cached_idx].item()
        local_idx = cached_idx - hash_cumsum_cpu[table].item()
        expected_row = _get_weights_offset_and_row(weights_cpu, table, local_idx, row_bytes, w_offsets_cpu)
        assert torch.equal(result["lxu_cache_weights"][target_set], expected_row)
        assert result["lru_state"][target_set].item() == 100

    @pytest.mark.parametrize("N, C, D, num_tables, weight_ty, indices_dtype", EMPTY_PARAMS)
    def test_empty_input(self, N, C, D, num_tables, weight_ty, indices_dtype):
        """空输入：缓存状态不应改变"""

        total_hash_size = 256  # 任意小值即可

        row_bytes = padded_row_size_in_bytes(D, weight_ty, 16)
        hash_cumsum_cpu, index_map_cpu, w_offsets_cpu, w_tys_cpu, d_offs_cpu, total_hash, rows_per_table = (
            _generate_cache_params(num_tables, D, row_bytes, total_hash_size)
        )
        w_tys_cpu[:] = weight_ty
        weights_cpu = _build_weights(num_tables, rows_per_table, row_bytes)

        cache_state = _make_cache_state(C, -1)
        cache_weights = torch.zeros(C, row_bytes, dtype=torch.uint8, device=DEVICE)
        lru_state = _make_lru_state(C, 0)
        miss_ts = _make_miss_timestamp(C, -1)

        orig_state = cache_state.cpu().clone()
        orig_weights = cache_weights.cpu().clone()

        empty_indices = torch.tensor([], dtype=torch.int64, device=DEVICE)
        if indices_dtype == torch.int32:
            empty_indices = empty_indices.to(torch.int32)

        result = call_operator(
            weights_cpu.to(DEVICE),
            hash_cumsum_cpu.to(DEVICE),
            total_hash,
            index_map_cpu.to(DEVICE),
            w_offsets_cpu.to(DEVICE),
            w_tys_cpu.to(DEVICE),
            d_offs_cpu.to(DEVICE),
            empty_indices,
            cache_state,
            cache_weights,
            100,
            lru_state,
            miss_ts,
            row_alignment=16,
            gather_cache_stats=False,
        )
        assert torch.equal(result["lxu_cache_state"], orig_state)
        assert torch.equal(result["lxu_cache_weights"], orig_weights)

    @pytest.mark.parametrize("N, C, D, num_tables, weight_ty, indices_dtype", REPEATED_PARAMS)
    def test_repeated_indices(self, N, C, D, num_tables, weight_ty, indices_dtype):
        """重复索引：相同索引重复出现，只应插入一次"""
        idx_val = 42
        indices = torch.tensor([idx_val, idx_val, idx_val], dtype=torch.int64)
        if indices_dtype == torch.int32:
            indices = indices.to(torch.int32)

        min_hash_size = num_tables * C
        total_hash_size = max(idx_val + 1, min_hash_size)
        if total_hash_size % num_tables != 0:
            total_hash_size = ((total_hash_size // num_tables) + 1) * num_tables

        row_bytes = padded_row_size_in_bytes(D, weight_ty, 16)
        hash_cumsum_cpu, index_map_cpu, w_offsets_cpu, w_tys_cpu, d_offs_cpu, total_hash, rows_per_table = (
            _generate_cache_params(num_tables, D, row_bytes, total_hash_size)
        )
        w_tys_cpu[:] = weight_ty
        weights_cpu = _build_weights(num_tables, rows_per_table, row_bytes)

        cache_state = _make_cache_state(C, -1)
        cache_weights = torch.zeros(C, row_bytes, dtype=torch.uint8, device=DEVICE)
        lru_state = _make_lru_state(C, 0)
        miss_ts = _make_miss_timestamp(C, -1)

        result = call_operator(
            weights_cpu.to(DEVICE),
            hash_cumsum_cpu.to(DEVICE),
            total_hash,
            index_map_cpu.to(DEVICE),
            w_offsets_cpu.to(DEVICE),
            w_tys_cpu.to(DEVICE),
            d_offs_cpu.to(DEVICE),
            indices.to(DEVICE),
            cache_state,
            cache_weights,
            1000,
            lru_state,
            miss_ts,
            row_alignment=16,
            gather_cache_stats=False,
        )

        s = cache_slot(idx_val, C)
        assert result["lxu_cache_state"][s].item() == idx_val
        table = index_map_cpu[idx_val].item()
        local_idx = idx_val - hash_cumsum_cpu[table].item()
        expected = _get_weights_offset_and_row(weights_cpu, table, local_idx, row_bytes, w_offsets_cpu)
        assert torch.equal(result["lxu_cache_weights"][s], expected)
        assert result["lru_state"][s].item() == 1000

    @pytest.mark.parametrize("C, D, num_tables, weight_ty, indices_dtype", STATS_PARAMS)
    def test_stats_gathering(self, C, D, num_tables, weight_ty, indices_dtype):
        """统计收集：验证 uvm_cache_stats 中的调用次数和冲突未命中数"""
        n_indices = min(8, C)
        indices = torch.arange(0, n_indices, dtype=torch.int64)
        if indices_dtype == torch.int32:
            indices = indices.to(torch.int32)

        min_hash_size = num_tables * C
        total_hash_size = max(int(indices.max().item()) + 1, min_hash_size)
        if total_hash_size % num_tables != 0:
            total_hash_size = ((total_hash_size // num_tables) + 1) * num_tables

        row_bytes = padded_row_size_in_bytes(D, weight_ty, 16)
        hash_cumsum_cpu, index_map_cpu, w_offsets_cpu, w_tys_cpu, d_offs_cpu, total_hash, rows_per_table = (
            _generate_cache_params(num_tables, D, row_bytes, total_hash_size)
        )
        w_tys_cpu[:] = weight_ty
        weights_cpu = _build_weights(num_tables, rows_per_table, row_bytes)

        cache_state = _make_cache_state(C, -1)
        cache_weights = torch.zeros(C, row_bytes, dtype=torch.uint8, device=DEVICE)
        lru_state = _make_lru_state(C, 0)
        miss_ts = _make_miss_timestamp(C, -1)
        uvm_stats = torch.zeros(5, dtype=torch.int32, device=DEVICE)

        # 第一次：全部未命中
        result = call_operator(
            weights_cpu.to(DEVICE),
            hash_cumsum_cpu.to(DEVICE),
            total_hash,
            index_map_cpu.to(DEVICE),
            w_offsets_cpu.to(DEVICE),
            w_tys_cpu.to(DEVICE),
            d_offs_cpu.to(DEVICE),
            indices.to(DEVICE),
            cache_state,
            cache_weights,
            100,
            lru_state,
            miss_ts,
            row_alignment=16,
            gather_cache_stats=True,
            uvm_cache_stats=uvm_stats,
        )
        stats_cpu = result["uvm_cache_stats"]
        assert stats_cpu[0].item() == 1  # num_calls
        assert stats_cpu[1].item() == n_indices  # num_requested_indices

        # 第二次：全部命中
        uvm_stats.zero_()
        result2 = call_operator(
            weights_cpu.to(DEVICE),
            hash_cumsum_cpu.to(DEVICE),
            total_hash,
            index_map_cpu.to(DEVICE),
            w_offsets_cpu.to(DEVICE),
            w_tys_cpu.to(DEVICE),
            d_offs_cpu.to(DEVICE),
            indices.to(DEVICE),
            cache_state,
            cache_weights,
            200,
            lru_state,
            miss_ts,
            row_alignment=16,
            gather_cache_stats=True,
            uvm_cache_stats=uvm_stats,
        )
        stats_cpu2 = result2["uvm_cache_stats"]
        assert stats_cpu2[0].item() == 1
        assert stats_cpu2[1].item() == n_indices
        assert stats_cpu2[4].item() == 0  # 命中后冲突未命中为0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
