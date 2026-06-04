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

import itertools

import pytest
import torch

import fbgemm_ascend  # noqa: F401

NPU_ENABLE = not torch.cuda.is_available()

if NPU_ENABLE:
    import torch_npu  # noqa: F401

    DEVICE = "npu:0"
else:
    DEVICE = "cuda:0"

VALUES_DATA_TYPES = [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64]
LENGTHS_DATA_TYPES = [torch.int32, torch.int64]
NAMESPACES = ["fbgemm"]

FORWARD_CASES = [
    {
        "id": "single_key_min_len",
        "num_keys": 1,
        "batch_size": 1,
        "max_lengths_per_key": [1],
        "length_mode": "full",
        "padding_value": 0,
    },
    {
        "id": "two_keys_mixed_zero",
        "num_keys": 2,
        "batch_size": 5,
        "max_lengths_per_key": [4, 6],
        "length_mode": "mixed_zero",
        "padding_value": -1,
    },
    {
        "id": "three_keys_staircase",
        "num_keys": 3,
        "batch_size": 4,
        "max_lengths_per_key": [3, 5, 7],
        "length_mode": "staircase",
        "padding_value": 2,
    },
    {
        "id": "all_zero_lengths",
        "num_keys": 3,
        "batch_size": 6,
        "max_lengths_per_key": [2, 4, 8],
        "length_mode": "all_zero",
        "padding_value": 9,
    },
    {
        "id": "full_lengths_varied_keys",
        "num_keys": 4,
        "batch_size": 3,
        "max_lengths_per_key": [1, 2, 4, 8],
        "length_mode": "full",
        "padding_value": 0,
    },
    {
        "id": "zero_max_length_key",
        "num_keys": 3,
        "batch_size": 4,
        "max_lengths_per_key": [0, 3, 5],
        "length_mode": "mixed_zero",
        "padding_value": -3,
    },
]

SHAPE_STRESS_CASES = [
    {
        "id": "many_keys_small_batch",
        "num_keys": 16,
        "batch_size": 4,
        "max_lengths_per_key": [8] * 16,
        "length_mode": "staircase",
        "padding_value": 0,
    },
    {
        "id": "large_batch_medium_sequence",
        "num_keys": 4,
        "batch_size": 128,
        "max_lengths_per_key": [64, 32, 16, 8],
        "length_mode": "random",
        "padding_value": 1,
    },
    {
        "id": "long_sequence",
        "num_keys": 2,
        "batch_size": 32,
        "max_lengths_per_key": [512, 1024],
        "length_mode": "random",
        "padding_value": 0,
    },
]


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(2026)


def op_namespace(namespace):
    return getattr(torch.ops, namespace)


def make_lengths(batch_size, max_lengths_per_key, mode, dtype):
    rows = []
    for key_idx, max_len in enumerate(max_lengths_per_key):
        if mode == "all_zero" or max_len == 0:
            row = torch.zeros((batch_size,), dtype=torch.int64)
        elif mode == "full":
            row = torch.full((batch_size,), max_len, dtype=torch.int64)
        elif mode == "mixed_zero":
            base = torch.arange(batch_size, dtype=torch.int64)
            row = (base * (key_idx + 1)) % (max_len + 1)
        elif mode == "staircase":
            base = torch.arange(batch_size, dtype=torch.int64)
            row = (base + key_idx) % (max_len + 1)
        else:
            row = torch.randint(0, max_len + 1, (batch_size,), dtype=torch.int64)
        rows.append(row)
    return torch.stack(rows, dim=0).to(dtype)


def make_values(total_l, dtype):
    if dtype in [torch.int32, torch.int64]:
        return torch.randint(-1000000, 1000000, (total_l,), dtype=dtype)
    return torch.empty((total_l,), dtype=dtype).uniform_(-1.0, 1.0)


def make_case_tensors(case, values_dtype, lengths_dtype):
    lengths = make_lengths(
        case["batch_size"],
        case["max_lengths_per_key"],
        case["length_mode"],
        lengths_dtype,
    )
    per_key_totals = lengths.to(torch.int64).sum(dim=1).tolist()
    offset_per_key = [0] + list(itertools.accumulate(per_key_totals))
    values = make_values(offset_per_key[-1], values_dtype)
    return values, lengths, offset_per_key


def stacked_jagged_1d_to_dense_reference(values, lengths, offset_per_key, max_lengths_per_key, padding_value):
    return torch.ops.fbgemm.stacked_jagged_1d_to_dense(
        values,
        lengths,
        offset_per_key,
        max_lengths_per_key,
        padding_value,
    )


def assert_outputs_equal(actual_outputs, expected_outputs):
    assert len(actual_outputs) == len(expected_outputs)
    for actual, expected in zip(actual_outputs, expected_outputs):
        assert actual.shape == expected.shape
        assert torch.equal(actual.cpu(), expected.cpu()), (
            f"结果不匹配\nexpected:\n{expected.cpu()}\nactual:\n{actual.cpu()}"
        )


def run_forward_case(namespace, case, values_dtype, lengths_dtype):
    values, lengths, offset_per_key = make_case_tensors(case, values_dtype, lengths_dtype)
    expected_outputs = stacked_jagged_1d_to_dense_reference(
        values,
        lengths,
        offset_per_key,
        case["max_lengths_per_key"],
        case["padding_value"],
    )
    actual_outputs = op_namespace(namespace).stacked_jagged_1d_to_dense(
        values.to(DEVICE),
        lengths.to(DEVICE),
        offset_per_key,
        case["max_lengths_per_key"],
        case["padding_value"],
    )
    assert_outputs_equal(actual_outputs, expected_outputs)


@pytest.mark.skipif(not NPU_ENABLE, reason="需要 NPU 设备")
@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize("case", FORWARD_CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("values_dtype", VALUES_DATA_TYPES)
@pytest.mark.parametrize("lengths_dtype", LENGTHS_DATA_TYPES)
def test_stacked_jagged_1d_to_dense_forward_full_coverage(namespace, case, values_dtype, lengths_dtype):
    run_forward_case(namespace, case, values_dtype, lengths_dtype)


@pytest.mark.skipif(not NPU_ENABLE, reason="需要 NPU 设备")
@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize("case", SHAPE_STRESS_CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("values_dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64])
def test_stacked_jagged_1d_to_dense_shape_stress(namespace, case, values_dtype):
    run_forward_case(namespace, case, values_dtype, torch.int64)
