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
# ==============================================================================

import pytest
import torch
import torch_npu
import fbgemm_gpu  # noqa: F401
import fbgemm_ascend  # noqa: F401

torch.npu.config.allow_internal_format = False

DEVICE = "npu:0"
# 检查 NPU 是否可用
npu_available = torch.npu.is_available()
if npu_available:
    torch_npu.npu.set_device(DEVICE)


def test_all_miss():
    """测试全部未命中：查询的索引都不在缓存中"""
    cache_state = torch.tensor([[0], [1], [2], [3]], device=DEVICE)
    indices = torch.tensor([10, 11, 12, 13], device=DEVICE)

    result = torch.ops.fbgemm.direct_mapped_lxu_cache_lookup(indices, cache_state, -1, False, None)

    expected = torch.tensor([-1, -1, -1, -1], device=DEVICE)
    assert torch.all(result == expected), f"Expected {expected}, got {result}"


def test_mixed_hit_miss():
    """测试混合命中/未命中"""
    cache_state = torch.tensor([[20], [10], [15], [20]], device=DEVICE)
    indices = torch.tensor([5, 10, 99, 15, 88, 20], device=DEVICE)

    result = torch.ops.fbgemm.direct_mapped_lxu_cache_lookup(indices, cache_state, -1, False, None)

    expected = torch.tensor([-1, 1, -1, -1, -1, -1], device=DEVICE)
    assert torch.all(result == expected), f"Expected {expected}, got {result}"


def test_duplicate_indices():
    """测试重复索引查询"""
    cache_state = torch.tensor([[42], [100]], device=DEVICE)
    indices = torch.tensor([42, 42, 42, 100, 42], device=DEVICE)

    result = torch.ops.fbgemm.direct_mapped_lxu_cache_lookup(indices, cache_state, -1, False, None)

    expected = torch.tensor([0, 0, 0, -1, 0], device=DEVICE)
    assert torch.all(result == expected), f"Expected {expected}, got {result}"


def test_single_element():
    """测试单个元素"""
    cache_state = torch.tensor([[123]], device=DEVICE)
    indices = torch.tensor([123], device=DEVICE)

    result = torch.ops.fbgemm.direct_mapped_lxu_cache_lookup(indices, cache_state, -1, False, None)

    expected = torch.tensor([0], device=DEVICE)
    assert torch.all(result == expected), f"Expected {expected}, got {result}"


def test_single_slot_cache():
    """测试单个缓存槽位（C=1）"""
    cache_state = torch.tensor([[999]], device=DEVICE)
    indices = torch.tensor([999, 888, 999, 777], device=DEVICE)

    result = torch.ops.fbgemm.direct_mapped_lxu_cache_lookup(indices, cache_state, -1, False, None)

    expected = torch.tensor([0, -1, 0, -1], device=DEVICE)
    assert torch.all(result == expected), f"Expected {expected}, got {result}"


def test_large_index_values():
    """测试大索引值"""
    cache_state = torch.tensor([[2**50], [2**55]], device=DEVICE)
    indices = torch.tensor([2**50, 2**55, 2**60], device=DEVICE)

    result = torch.ops.fbgemm.direct_mapped_lxu_cache_lookup(indices, cache_state, -1, False, None)

    expected = torch.tensor([0, 1, -1], device=DEVICE)
    assert torch.all(result == expected), f"Expected {expected}, got {result}"


def test_empty_input():
    """测试空输入"""
    cache_state = torch.tensor([[1]], device=DEVICE)
    indices = torch.tensor([], device=DEVICE)

    result = torch.ops.fbgemm.direct_mapped_lxu_cache_lookup(indices, cache_state, -1, False, None)

    assert result.numel() == 0, f"Expected empty tensor, got shape {result.shape}"


def test_custom_invalid_index():
    """测试自定义 invalid_index 值"""
    cache_state = torch.tensor([[0], [1], [2], [3]], device=DEVICE)
    indices = torch.tensor([0, -999, 1, -999, 3, 0, -999, 2, -999, 3], device=DEVICE)

    result = torch.ops.fbgemm.direct_mapped_lxu_cache_lookup(indices, cache_state, -999, False, None)

    assert result[0] == 0, f"Expected indices[0]=0, got {result[0]}"
    assert result[2] == -1, f"Expected indices[2]=1, got {result[2]}"
    assert result[4] == -1, f"Expected indices[4]=3, got {result[4]}"
    assert result[5] == 0, f"Expected indices[5]=0, got {result[5]}"
    assert result[7] == -1, f"Expected indices[7]=2, got {result[7]}"
    assert result[9] == -1, f"Expected indices[9]=3, got {result[9]}"


def test_stats_gathering():
    """测试统计信息收集"""
    cache_state = torch.tensor([[0], [1], [2]], device=DEVICE)
    indices = torch.tensor([0, 1, 99, 2, 0, 99], device=DEVICE)
    stats = torch.zeros(6, dtype=torch.int32, device=DEVICE)

    result = torch.ops.fbgemm.direct_mapped_lxu_cache_lookup(indices, cache_state, -1, True, stats)

    expected_result = torch.tensor([0, -1, -1, -1, 0, -1], device=DEVICE)
    expected_stats = torch.tensor([0, 0, 0, 0, 0, 4], device=DEVICE)
    assert torch.all(result == expected_result), f"Expected {expected_result}, got {result}"
    assert torch.all(stats == expected_stats), f"Expected {expected_stats}, got {stats}"


def test_invalid_index_skip():
    """测试 invalid_index 跳过时输出未定义的行为"""
    cache_state = torch.tensor([[0], [1], [2], [3]], device=DEVICE)

    indices = torch.tensor([0, -1, 2, -1, 4], device=DEVICE)
    stats = torch.zeros(6, dtype=torch.int32, device=DEVICE)

    result = torch.ops.fbgemm.direct_mapped_lxu_cache_lookup(indices, cache_state, -1, True, stats)

    assert result[0].item() == 0, f"Expected 0, got {result[0]}"
    assert result[2].item() == -1, f"Expected -1, got {result[2]}"
    assert result[4].item() == -1, f"Expected -1, got {result[4]}"
    expected_stats = torch.tensor([0, 0, 0, 0, 0, 2], device=DEVICE)
    assert torch.all(stats == expected_stats), f"Expected {expected_stats}, got {stats}"


def test_gather_cache_stats_error_without_uvm_cache_stats():
    """测试 gather_cache_stats 为True时uvm_cache_stats为空的行为"""
    cache_state = torch.tensor([[0], [1], [2], [3]], device=DEVICE)

    indices = torch.tensor([0, 1, 2, 3], device=DEVICE)
    with pytest.raises(Exception):
        torch.ops.fbgemm.direct_mapped_lxu_cache_lookup(indices, cache_state, -1, True, None)
        torch.synchronize()


def test_with_empty_cache():
    """测试 cache_state 为空时的行为"""
    cache_state = torch.tensor([[]], device=DEVICE)
    indices = torch.tensor([0, 1, 2, 3], device=DEVICE)

    with pytest.raises(Exception):
        torch.ops.fbgemm.direct_mapped_lxu_cache_lookup(indices, cache_state, -1, True, None)
        torch.synchronize()
