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

import fbgemm_ascend  # noqa: F401

NPU_ENABLE = not torch.cuda.is_available()

if NPU_ENABLE:
    import torch_npu  # noqa: F401

    DEVICE = "npu:0"
else:
    DEVICE = "cuda:0"

VALUES_DATA_TYPES = [torch.float32, torch.float16, torch.bfloat16]
LENGTHS_DATA_TYPES = [torch.int32, torch.int64]
NAMESPACES = ["fbgemm"]

FORWARD_CASES = [
    {
        "id": "single_key_min_len",
        "num_keys": 1,
        "batch_size": 1,
        "dense_dim": 1,
        "max_lengths_per_key": [1],
        "length_mode": "full",
        "padding_value": 0,
    },
    {
        "id": "two_keys_mixed_zero",
        "num_keys": 2,
        "batch_size": 5,
        "dense_dim": 4,
        "max_lengths_per_key": [4, 6],
        "length_mode": "mixed_zero",
        "padding_value": -1,
    },
    {
        "id": "three_keys_staircase",
        "num_keys": 3,
        "batch_size": 4,
        "dense_dim": 8,
        "max_lengths_per_key": [3, 5, 7],
        "length_mode": "staircase",
        "padding_value": 2,
    },
    {
        "id": "all_zero_lengths",
        "num_keys": 3,
        "batch_size": 6,
        "dense_dim": 16,
        "max_lengths_per_key": [2, 4, 8],
        "length_mode": "all_zero",
        "padding_value": 9,
    },
    {
        "id": "full_lengths_varied_keys",
        "num_keys": 4,
        "batch_size": 3,
        "dense_dim": 32,
        "max_lengths_per_key": [1, 2, 4, 8],
        "length_mode": "full",
        "padding_value": 0,
    },
    {
        "id": "zero_max_length_key",
        "num_keys": 3,
        "batch_size": 4,
        "dense_dim": 5,
        "max_lengths_per_key": [0, 3, 5],
        "length_mode": "mixed_zero",
        "padding_value": -3,
    },
]

BACKWARD_CASES = [
    {
        "id": "backward_two_keys",
        "num_keys": 2,
        "batch_size": 5,
        "dense_dim": 4,
        "max_lengths_per_key": [4, 6],
        "length_mode": "mixed_zero",
        "padding_value": 0,
    },
    {
        "id": "backward_three_keys_staircase",
        "num_keys": 3,
        "batch_size": 4,
        "dense_dim": 8,
        "max_lengths_per_key": [3, 5, 7],
        "length_mode": "staircase",
        "padding_value": 0,
    },
    {
        "id": "backward_all_zero_lengths",
        "num_keys": 3,
        "batch_size": 6,
        "dense_dim": 16,
        "max_lengths_per_key": [2, 4, 8],
        "length_mode": "all_zero",
        "padding_value": 0,
    },
]

SHAPE_STRESS_CASES = [
    {
        "id": "many_keys_small_batch",
        "num_keys": 16,
        "batch_size": 4,
        "dense_dim": 8,
        "max_lengths_per_key": [8] * 16,
        "length_mode": "staircase",
        "padding_value": 0,
    },
    {
        "id": "large_batch_medium_sequence",
        "num_keys": 4,
        "batch_size": 128,
        "dense_dim": 16,
        "max_lengths_per_key": [64, 32, 16, 8],
        "length_mode": "random",
        "padding_value": 1,
    },
    {
        "id": "long_sequence",
        "num_keys": 2,
        "batch_size": 32,
        "dense_dim": 32,
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


def generate_lengths_row(batch_size, max_len, mode, dim_idx=0):
    if batch_size == 0:
        return torch.zeros((0,), dtype=torch.int64)
    if mode == "all_zero" or max_len == 0:
        return torch.zeros((batch_size,), dtype=torch.int64)
    if mode == "full":
        return torch.full((batch_size,), max_len, dtype=torch.int64)
    if mode == "mixed_zero":
        base = torch.arange(batch_size, dtype=torch.int64)
        return (base * (dim_idx + 1)) % (max_len + 1)
    if mode == "staircase":
        base = torch.arange(batch_size, dtype=torch.int64)
        return (base + dim_idx) % (max_len + 1)
    return torch.randint(0, max_len + 1, (batch_size,), dtype=torch.int64)


def generate_lengths_matrix(batch_size, max_lengths, mode, dtype=torch.int64):
    rows = [generate_lengths_row(batch_size, max_len, mode, idx) for idx, max_len in enumerate(max_lengths)]
    return torch.stack(rows, dim=0).to(dtype)


def generate_stacked_offsets_from_lengths(lengths):
    per_key_totals = lengths.to(torch.int64).sum(dim=1).tolist()
    offset_per_key = [0]
    for total in per_key_totals:
        offset_per_key.append(offset_per_key[-1] + total)
    return offset_per_key


def generate_values(total_l, dtype, shape_tail=()):
    shape = (total_l, *shape_tail)
    return torch.empty(shape, dtype=dtype).uniform_(-1.0, 1.0)


def make_case_tensors(case, values_dtype, lengths_dtype):
    lengths = generate_lengths_matrix(
        case["batch_size"],
        case["max_lengths_per_key"],
        case["length_mode"],
        lengths_dtype,
    )
    offset_per_key = generate_stacked_offsets_from_lengths(lengths)
    values = generate_values(offset_per_key[-1], values_dtype, shape_tail=(case["dense_dim"],))
    return values, lengths, offset_per_key


def stacked_jagged_2d_to_dense_reference(values, lengths, offset_per_key, max_lengths_per_key, padding_value):
    return torch.ops.fbgemm.stacked_jagged_2d_to_dense(
        values,
        lengths,
        offset_per_key,
        max_lengths_per_key,
        padding_value,
    )


def stacked_jagged_2d_to_dense_backward_reference(grad_outputs, offsets_tensor_per_key, offset_per_key):
    values = []
    for key_idx, grad_output in enumerate(grad_outputs):
        total_l = offset_per_key[key_idx + 1] - offset_per_key[key_idx]
        values.append(
            torch.ops.fbgemm.jagged_to_padded_dense_backward(
                grad_output,
                [offsets_tensor_per_key[key_idx]],
                total_l,
            )
        )
    return torch.cat(values, dim=0) if values else torch.empty((0, 0), dtype=grad_outputs[0].dtype)


def assert_outputs_equal(actual_outputs, expected_outputs):
    assert len(actual_outputs) == len(expected_outputs)
    for actual, expected in zip(actual_outputs, expected_outputs):
        actual_cpu = actual.cpu()
        assert actual_cpu.shape == expected.shape
        torch.equal(actual_cpu, expected)


def run_forward_case(namespace, case, values_dtype, lengths_dtype):
    values, lengths, offset_per_key = make_case_tensors(case, values_dtype, lengths_dtype)
    expected_outputs = stacked_jagged_2d_to_dense_reference(
        values,
        lengths,
        offset_per_key,
        case["max_lengths_per_key"],
        case["padding_value"],
    )
    actual_outputs, offsets_tensor_per_key = op_namespace(namespace).stacked_jagged_2d_to_dense_forward(
        values.to(DEVICE),
        lengths.to(DEVICE),
        offset_per_key,
        case["max_lengths_per_key"],
        case["padding_value"],
    )
    assert len(offsets_tensor_per_key) == len(case["max_lengths_per_key"])
    assert_outputs_equal(actual_outputs, expected_outputs)


def run_backward_case(namespace, case, values_dtype, lengths_dtype):
    values, lengths, offset_per_key = make_case_tensors(case, values_dtype, lengths_dtype)
    _, offsets_tensor_per_key = op_namespace(namespace).stacked_jagged_2d_to_dense_forward(
        values.to(DEVICE),
        lengths.to(DEVICE),
        offset_per_key,
        case["max_lengths_per_key"],
        case["padding_value"],
    )
    offsets_tensor_per_key_cpu = [offset.cpu() for offset in offsets_tensor_per_key]
    grad_outputs = [
        torch.empty((case["batch_size"], max_len, case["dense_dim"]), dtype=values_dtype).uniform_(-1.0, 1.0)
        for max_len in case["max_lengths_per_key"]
    ]
    expected_grad_values = stacked_jagged_2d_to_dense_backward_reference(
        grad_outputs,
        offsets_tensor_per_key_cpu,
        offset_per_key,
    )
    actual_grad_values = op_namespace(namespace).stacked_jagged_2d_to_dense_backward(
        case["batch_size"],
        case["dense_dim"],
        values.size(0),
        [grad_output.to(DEVICE) for grad_output in grad_outputs],
        offsets_tensor_per_key,
        offset_per_key,
    )
    assert actual_grad_values.shape == values.shape
    assert_outputs_equal(actual_grad_values, expected_grad_values)


@pytest.mark.skipif(not NPU_ENABLE, reason="需要 NPU 设备")
@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize("case", FORWARD_CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("values_dtype", VALUES_DATA_TYPES)
@pytest.mark.parametrize("lengths_dtype", LENGTHS_DATA_TYPES)
def test_stacked_jagged_2d_to_dense_forward_full_coverage(namespace, case, values_dtype, lengths_dtype):
    run_forward_case(namespace, case, values_dtype, lengths_dtype)


@pytest.mark.skipif(not NPU_ENABLE, reason="需要 NPU 设备")
@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize("case", BACKWARD_CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("values_dtype", VALUES_DATA_TYPES)
@pytest.mark.parametrize("lengths_dtype", LENGTHS_DATA_TYPES)
def test_stacked_jagged_2d_to_dense_backward_full_coverage(namespace, case, values_dtype, lengths_dtype):
    run_backward_case(namespace, case, values_dtype, lengths_dtype)


@pytest.mark.skipif(not NPU_ENABLE, reason="需要 NPU 设备")
@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize("case", SHAPE_STRESS_CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("values_dtype", VALUES_DATA_TYPES)
def test_stacked_jagged_2d_to_dense_shape_stress(namespace, case, values_dtype):
    run_forward_case(namespace, case, values_dtype, torch.int64)
