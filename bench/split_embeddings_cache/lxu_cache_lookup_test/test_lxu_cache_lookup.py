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
import fbgemm_ascend  # noqa: F401

torch.npu.config.allow_internal_format = False

DEVICE = "npu:0"
# 检查 NPU 是否可用
npu_available = torch.npu.is_available()
if npu_available:
    torch_npu.npu.set_device(DEVICE)


DEFAULT_ASSOC = 32


def test_all_miss_single_set():
    """测试全部未命中：单个cache set，查询的索引都不在缓存中"""
    # cache_state: (1, 32) - 1个cache set，32个slots
    cache_state = torch.arange(DEFAULT_ASSOC, dtype=torch.int64, device=DEVICE).unsqueeze(0)
    # 索引32-39不在cache中
    indices = torch.tensor([32, 33, 34, 35, 36, 100, 1000, 1725], dtype=torch.int64, device=DEVICE)
    max_index = 8000

    result = torch.ops.fbgemm.lxu_cache_lookup(indices, cache_state, max_index)

    expected = torch.full((indices.numel(),), -1, dtype=torch.int32)
    torch.testing.assert_close(result.cpu(), expected)


def test_all_hit_single_set():
    """测试全部命中：单个cache set，所有索引都在缓存中"""
    cache_state = torch.arange(DEFAULT_ASSOC, dtype=torch.int64, device=DEVICE).unsqueeze(0)
    # 索引0-31都在cache中
    indices = torch.arange(DEFAULT_ASSOC, dtype=torch.int64, device=DEVICE)
    max_index = 8000

    result = torch.ops.fbgemm.lxu_cache_lookup(indices, cache_state, max_index)

    expected = indices.cpu().int()
    torch.testing.assert_close(result.cpu(), expected)


def test_mixed_hit_miss_single_set():
    """测试混合命中/未命中：单个cache set"""
    cache_state = torch.arange(DEFAULT_ASSOC, dtype=torch.int64, device=DEVICE).unsqueeze(0)
    # 0-7命中，8-9未命中
    indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 100, 200], dtype=torch.int64, device=DEVICE)
    max_index = 8000

    result = torch.ops.fbgemm.lxu_cache_lookup(indices, cache_state, max_index)

    expected = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, -1, -1], dtype=torch.int32)
    torch.testing.assert_close(result.cpu(), expected)


def test_empty_input():
    """测试空输入"""
    cache_state = torch.arange(DEFAULT_ASSOC, dtype=torch.int64, device=DEVICE).unsqueeze(0)
    indices = torch.tensor([], dtype=torch.int64, device=DEVICE)
    max_index = 8000

    result = torch.ops.fbgemm.lxu_cache_lookup(indices, cache_state, max_index)

    assert result.numel() == 0


def test_single_element():
    """测试单个元素"""
    cache_state = torch.arange(DEFAULT_ASSOC, dtype=torch.int64, device=DEVICE).unsqueeze(0)
    indices = torch.tensor([0], dtype=torch.int64, device=DEVICE)
    max_index = 8000

    result = torch.ops.fbgemm.lxu_cache_lookup(indices, cache_state, max_index)

    expected = torch.tensor([0], dtype=torch.int32)
    torch.testing.assert_close(result.cpu(), expected)


def test_large_index_values():
    """测试大索引值"""
    cache_state = torch.zeros(1, DEFAULT_ASSOC, dtype=torch.int64, device=DEVICE)
    cache_state[0][0] = 2**50
    cache_state[0][1] = 2**55
    indices = torch.tensor([2**50, 2**55, 2**60], dtype=torch.int64, device=DEVICE)
    max_index = 8000

    result = torch.ops.fbgemm.lxu_cache_lookup(indices, cache_state, max_index)

    expected = torch.tensor([0, 1, -1], dtype=torch.int32)
    torch.testing.assert_close(result.cpu(), expected)


def test_custom_invalid_index():
    """测试自定义 invalid_index 值"""
    cache_state = torch.arange(DEFAULT_ASSOC, dtype=torch.int64, device=DEVICE).unsqueeze(0)
    # -999 作为 invalid_index
    indices = torch.tensor([0, -999, 1, -999, 2, 0, -999, 1, -999, 2], dtype=torch.int64, device=DEVICE)
    max_index = 8000

    result = torch.ops.fbgemm.lxu_cache_lookup(indices, cache_state, max_index, False, None, None, None)

    # 0->0, -999跳过返回-1, 1->1, -999跳过, 2->2, 0->0, -999跳过, 1->1, -999跳过, 2->2
    expected = torch.tensor([0, -1, 1, -1, 2, 0, -1, 1, -1, 2], dtype=torch.int32)
    torch.testing.assert_close(result.cpu(), expected)


def test_duplicate_indices():
    """测试重复索引查询"""
    cache_state = torch.arange(DEFAULT_ASSOC, dtype=torch.int64, device=DEVICE).unsqueeze(0)
    indices = torch.tensor([0, 0, 0, 1, 0, 2, 2, 2], dtype=torch.int64, device=DEVICE)
    max_index = 8000

    result = torch.ops.fbgemm.lxu_cache_lookup(indices, cache_state, max_index)

    expected = torch.tensor([0, 0, 0, 1, 0, 2, 2, 2], dtype=torch.int32)
    torch.testing.assert_close(result.cpu(), expected)


def test_without_optional_params():
    """测试只使用必需参数的调用"""
    cache_state = torch.arange(DEFAULT_ASSOC, dtype=torch.int64, device=DEVICE).unsqueeze(0)
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=DEVICE)
    max_index = 8000

    result = torch.ops.fbgemm.lxu_cache_lookup(indices, cache_state, max_index)

    expected = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    torch.testing.assert_close(result.cpu(), expected)


def test_indices_at_associativity_boundary():
    """测试 associativity 边界值"""
    cache_state = torch.arange(32, dtype=torch.int64, device=DEVICE).unsqueeze(0)
    # 索引32正好是associativity，应该未命中
    indices = torch.tensor([0, 31, 32, 33, 63, 64], dtype=torch.int64, device=DEVICE)
    max_index = 8000

    result = torch.ops.fbgemm.lxu_cache_lookup(indices, cache_state, max_index)

    # 0->0, 31->31, 32->未命中, 33->未命中, 63->未命中, 64->未命中
    expected = torch.tensor([0, 31, -1, -1, -1, -1], dtype=torch.int32)
    torch.testing.assert_close(result.cpu(), expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
