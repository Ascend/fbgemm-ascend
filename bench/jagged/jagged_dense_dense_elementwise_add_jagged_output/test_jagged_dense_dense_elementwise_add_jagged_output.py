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

import os
import sys

import pytest
import torch

import fbgemm_ascend  # noqa: F401

_COMM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jagged_to_padded_dense_test"))
if _COMM_DIR not in sys.path:
    sys.path.insert(0, _COMM_DIR)

from test_comm_utils import PRECISION_ERROR_RANGE  # noqa: E402

NPU_ENABLE = not torch.cuda.is_available()

if NPU_ENABLE:
    import torch_npu  # noqa: F401

    DEVICE = "npu:0"
else:
    DEVICE = "cuda:0"

VALUES_DATA_TYPES = [torch.float32, torch.float16, torch.bfloat16]
OFFSETS_DATA_TYPES = [torch.int32, torch.int64]
NAMESPACES = ["fbgemm"]

FORWARD_CASES = [
    {
        "id": "1d_scalar_min_len",
        "num_jagged_dim": 1,
        "batch_size": 1,
        "max_lengths": [1],
        "inner_dense_size": None,
        "length_mode": "full",
    },
    {
        "id": "1d_scalar_zero_and_truncated",
        "num_jagged_dim": 1,
        "batch_size": 5,
        "max_lengths": [4],
        "inner_dense_size": None,
        "length_mode": "mixed_zero",
    },
    {
        "id": "2d_vector_common",
        "num_jagged_dim": 1,
        "batch_size": 4,
        "max_lengths": [8],
        "inner_dense_size": 7,
        "length_mode": "random",
    },
    {
        "id": "2d_vector_full_padding_boundary",
        "num_jagged_dim": 1,
        "batch_size": 3,
        "max_lengths": [16],
        "inner_dense_size": 32,
        "length_mode": "staircase",
    },
    {
        "id": "2_jagged_dims_scalar",
        "num_jagged_dim": 2,
        "batch_size": 3,
        "max_lengths": [4, 5],
        "inner_dense_size": None,
        "length_mode": "mixed_zero",
    },
    {
        "id": "2_jagged_dims_vector",
        "num_jagged_dim": 2,
        "batch_size": 2,
        "max_lengths": [3, 4],
        "inner_dense_size": 6,
        "length_mode": "random",
    },
    {
        "id": "3_jagged_dims_vector",
        "num_jagged_dim": 3,
        "batch_size": 2,
        "max_lengths": [3, 3, 4],
        "inner_dense_size": 5,
        "length_mode": "staircase",
    },
    {
        "id": "4_jagged_dims_small",
        "num_jagged_dim": 4,
        "batch_size": 2,
        "max_lengths": [2, 3, 2, 3],
        "inner_dense_size": 3,
        "length_mode": "mixed_zero",
    },
    {
        "id": "5_jagged_dims_max_boundary",
        "num_jagged_dim": 5,
        "batch_size": 2,
        "max_lengths": [2, 2, 2, 2, 2],
        "inner_dense_size": 2,
        "length_mode": "mixed_zero",
    },
    {
        "id": "empty_total_l_scalar",
        "num_jagged_dim": 1,
        "batch_size": 3,
        "max_lengths": [5],
        "inner_dense_size": None,
        "length_mode": "all_zero",
    },
    {
        "id": "empty_total_l_vector",
        "num_jagged_dim": 1,
        "batch_size": 3,
        "max_lengths": [5],
        "inner_dense_size": 4,
        "length_mode": "all_zero",
    },
    {
        "id": "empty_dense_batch_scalar",
        "num_jagged_dim": 1,
        "batch_size": 0,
        "max_lengths": [5],
        "inner_dense_size": None,
        "length_mode": "all_zero",
    },
    {
        "id": "empty_dense_batch_vector",
        "num_jagged_dim": 1,
        "batch_size": 0,
        "max_lengths": [5],
        "inner_dense_size": 4,
        "length_mode": "all_zero",
    },
]

BACKWARD_CASES = [
    {
        "id": "backward_1d_scalar",
        "num_jagged_dim": 1,
        "batch_size": 4,
        "max_lengths": [7],
        "inner_dense_size": None,
        "length_mode": "mixed_zero",
    },
    {
        "id": "backward_2d_vector",
        "num_jagged_dim": 1,
        "batch_size": 4,
        "max_lengths": [9],
        "inner_dense_size": 8,
        "length_mode": "random",
    },
    {
        "id": "backward_2_jagged_dims_scalar",
        "num_jagged_dim": 2,
        "batch_size": 3,
        "max_lengths": [4, 5],
        "inner_dense_size": None,
        "length_mode": "staircase",
    },
    {
        "id": "backward_3_jagged_dims_vector",
        "num_jagged_dim": 3,
        "batch_size": 2,
        "max_lengths": [3, 4, 3],
        "inner_dense_size": 4,
        "length_mode": "mixed_zero",
    },
    {
        "id": "backward_5_jagged_dims_max_boundary",
        "num_jagged_dim": 5,
        "batch_size": 2,
        "max_lengths": [2, 2, 2, 2, 2],
        "inner_dense_size": 2,
        "length_mode": "mixed_zero",
    },
]

SHAPE_STRESS_CASES = [
    {
        "id": "long_sequence_small_inner",
        "num_jagged_dim": 1,
        "batch_size": 16,
        "max_lengths": [512],
        "inner_dense_size": 2,
        "length_mode": "staircase",
    },
    {
        "id": "large_inner_dense",
        "num_jagged_dim": 1,
        "batch_size": 8,
        "max_lengths": [64],
        "inner_dense_size": 128,
        "length_mode": "random",
    },
    {
        "id": "multi_dim_dense_volume",
        "num_jagged_dim": 2,
        "batch_size": 8,
        "max_lengths": [16, 16],
        "inner_dense_size": 16,
        "length_mode": "mixed_zero",
    },
    {
        "id": "max_jagged_dim_boundary",
        "num_jagged_dim": 5,
        "batch_size": 2,
        "max_lengths": [2, 2, 2, 2, 2],
        "inner_dense_size": 4,
        "length_mode": "mixed_zero",
    },
]


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(2026)


def op_namespace(namespace):
    return getattr(torch.ops, namespace)


def assert_close(actual, expected, dtype):
    tol = PRECISION_ERROR_RANGE.get(dtype, 1e-4)
    assert torch.allclose(
        actual.cpu(),
        expected.cpu(),
        atol=tol,
        rtol=tol,
    ), f"结果不匹配\nexpected:\n{expected.cpu()}\nactual:\n{actual.cpu()}"


def make_lengths(batch_size, max_len, mode, dim_idx):
    if batch_size == 0:
        return torch.zeros((0,), dtype=torch.int64)
    if mode == "all_zero":
        return torch.zeros((batch_size,), dtype=torch.int64)
    if mode == "full":
        return torch.full((batch_size,), max_len, dtype=torch.int64)
    if mode == "mixed_zero":
        base = torch.arange(batch_size, dtype=torch.int64)
        return (base * (dim_idx + 1)) % (max_len + 1)
    if mode == "staircase":
        base = torch.arange(batch_size, dtype=torch.int64)
        return base % (max_len + 1)
    return torch.randint(0, max_len + 1, (batch_size,), dtype=torch.int64)


def make_offsets(batch_size, max_lengths, offsets_dtype, length_mode):
    offsets = []
    num_parents = batch_size
    for dim_idx, max_len in enumerate(max_lengths):
        lengths = make_lengths(num_parents, max_len, length_mode, dim_idx)
        offset = torch.cat((torch.zeros((1,), dtype=torch.int64), torch.cumsum(lengths, dim=0)))
        offsets.append(offset.to(offsets_dtype))
        num_parents = int(offset[-1].item())
    return offsets


def make_values(total_l, inner_dense_size, dtype):
    shape = (total_l,) if inner_dense_size is None else (total_l, inner_dense_size)
    return torch.empty(shape, dtype=dtype).uniform_(-1.0, 1.0)


def make_dense(batch_size, max_lengths, inner_dense_size, dtype, non_contiguous=False):
    shape = (batch_size, *max_lengths) if inner_dense_size is None else (batch_size, *max_lengths, inner_dense_size)
    dense = torch.empty(shape, dtype=dtype).uniform_(-0.5, 0.5)
    if not non_contiguous:
        return dense
    if dense.dim() < 2:
        return dense
    source = torch.empty((*shape, 2), dtype=dtype)
    source.uniform_(-0.5, 0.5)
    source[..., 0] = dense
    return source[..., 0]


def make_case_tensors(case, values_dtype, offsets_dtype, non_contiguous_dense=False):
    offsets = make_offsets(
        case["batch_size"],
        case["max_lengths"],
        offsets_dtype,
        case["length_mode"],
    )
    total_l = int(offsets[-1][-1].item())
    values = make_values(total_l, case["inner_dense_size"], values_dtype)
    y_0 = make_dense(
        case["batch_size"],
        case["max_lengths"],
        case["inner_dense_size"],
        values_dtype,
        non_contiguous=non_contiguous_dense,
    )
    y_1 = make_dense(
        case["batch_size"],
        case["max_lengths"],
        case["inner_dense_size"],
        values_dtype,
        non_contiguous=non_contiguous_dense,
    )
    return values, offsets, y_0, y_1


def dense_to_jagged_reference(dense, offsets):
    total_l = int(offsets[-1][-1].item())
    inner_shape = tuple(dense.shape[len(offsets) + 1 :])
    if total_l == 0:
        return dense.new_empty((0, *inner_shape)) if inner_shape else dense.new_empty((0,))

    values = []

    def collect(dim_idx, parent_idx, dense_indices):
        start = int(offsets[dim_idx][parent_idx].item())
        end = int(offsets[dim_idx][parent_idx + 1].item())
        for child_idx in range(start, end):
            dense_pos = child_idx - start
            next_dense_indices = (*dense_indices, dense_pos)
            if dim_idx + 1 == len(offsets):
                values.append(dense[next_dense_indices])
            else:
                collect(dim_idx + 1, child_idx, next_dense_indices)

    for batch_idx in range(dense.shape[0]):
        collect(0, batch_idx, (batch_idx,))

    return torch.stack(values, dim=0)


def jagged_to_padded_dense_reference(values, offsets, max_lengths):
    dense = torch.zeros(
        (offsets[0].numel() - 1, *max_lengths, *values.shape[1:]),
        dtype=values.dtype,
        device=values.device,
    )
    if values.shape[0] == 0:
        return dense
    values_work = values.unsqueeze(-1) if values.dim() == 1 else values
    dense_work = dense.unsqueeze(-1) if values.dim() == 1 else dense

    def fill(dim_idx, parent_idx, dense_indices):
        start = int(offsets[dim_idx][parent_idx].item())
        end = int(offsets[dim_idx][parent_idx + 1].item())
        for child_idx in range(start, end):
            dense_pos = child_idx - start
            next_dense_indices = (*dense_indices, dense_pos)
            if dim_idx + 1 == len(offsets):
                dense_work[next_dense_indices] = values_work[child_idx]
            else:
                fill(dim_idx + 1, child_idx, next_dense_indices)

    for batch_idx in range(dense_work.shape[0]):
        fill(0, batch_idx, (batch_idx,))

    return dense_work.squeeze(-1) if values.dim() == 1 else dense_work


def run_forward_case(namespace, case, values_dtype, offsets_dtype, non_contiguous_dense=False):
    values, offsets, y_0, y_1 = make_case_tensors(
        case, values_dtype, offsets_dtype, non_contiguous_dense=non_contiguous_dense
    )
    ref = dense_to_jagged_reference(
        jagged_to_padded_dense_reference(values, offsets, case["max_lengths"]) + y_0 + y_1,
        offsets,
    )
    out, out_offsets = op_namespace(namespace).jagged_dense_dense_elementwise_add_jagged_output(
        values.to(DEVICE),
        [offset.to(DEVICE) for offset in offsets],
        y_0.to(DEVICE),
        y_1.to(DEVICE),
    )
    assert out.shape == values.shape
    assert len(out_offsets) == len(offsets)
    for actual_offset, expected_offset in zip(out_offsets, offsets):
        assert torch.equal(actual_offset.cpu(), expected_offset)
    assert_close(out, ref, values_dtype)


def run_backward_case(namespace, case, values_dtype, offsets_dtype):
    values, offsets, y_0, y_1 = make_case_tensors(case, values_dtype, offsets_dtype)
    x_npu = values.to(DEVICE).detach().clone().requires_grad_(True)
    y0_npu = y_0.to(DEVICE).detach().clone().requires_grad_(True)
    y1_npu = y_1.to(DEVICE).detach().clone().requires_grad_(True)
    out, _ = op_namespace(namespace).jagged_dense_dense_elementwise_add_jagged_output(
        x_npu,
        [offset.to(DEVICE) for offset in offsets],
        y0_npu,
        y1_npu,
    )
    grad_output = torch.empty_like(out).uniform_(-1.0, 1.0)
    out.backward(grad_output)
    ref_y_grad = jagged_to_padded_dense_reference(grad_output.cpu(), offsets, case["max_lengths"])

    assert_close(x_npu.grad, grad_output, values_dtype)
    assert_close(y0_npu.grad, ref_y_grad, values_dtype)
    assert_close(y1_npu.grad, ref_y_grad, values_dtype)


@pytest.mark.skipif(not NPU_ENABLE, reason="需要 NPU 设备")
@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize("case", FORWARD_CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("values_dtype", VALUES_DATA_TYPES)
@pytest.mark.parametrize("offsets_dtype", OFFSETS_DATA_TYPES)
def test_jagged_dense_dense_elementwise_add_jagged_output_forward_full_coverage(
    namespace, case, values_dtype, offsets_dtype
):
    run_forward_case(namespace, case, values_dtype, offsets_dtype)


@pytest.mark.skipif(not NPU_ENABLE, reason="需要 NPU 设备")
@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize("case", FORWARD_CASES[:8], ids=lambda c: c["id"])
@pytest.mark.parametrize("values_dtype", VALUES_DATA_TYPES)
def test_jagged_dense_dense_elementwise_add_jagged_output_forward_non_contiguous_dense(namespace, case, values_dtype):
    run_forward_case(namespace, case, values_dtype, torch.int64, non_contiguous_dense=True)


@pytest.mark.skipif(not NPU_ENABLE, reason="需要 NPU AutogradPrivateUse1 实现")
@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize("case", BACKWARD_CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("values_dtype", VALUES_DATA_TYPES)
@pytest.mark.parametrize("offsets_dtype", OFFSETS_DATA_TYPES)
def test_jagged_dense_dense_elementwise_add_jagged_output_backward_full_coverage(
    namespace, case, values_dtype, offsets_dtype
):
    run_backward_case(namespace, case, values_dtype, offsets_dtype)


@pytest.mark.skipif(not NPU_ENABLE, reason="需要 NPU 设备")
@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize("case", SHAPE_STRESS_CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("values_dtype", VALUES_DATA_TYPES)
def test_jagged_dense_dense_elementwise_add_jagged_output_shape_stress(namespace, case, values_dtype):
    run_forward_case(namespace, case, values_dtype, torch.int64)


@pytest.mark.skipif(not NPU_ENABLE, reason="需要 NPU 设备")
@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize("values_1d", [True, False])
@pytest.mark.parametrize("values_dtype", VALUES_DATA_TYPES)
def test_jagged_dense_dense_elementwise_add_jagged_output_backward_operator(namespace, values_1d, values_dtype):
    case = {
        "id": "direct_backward",
        "num_jagged_dim": 2,
        "batch_size": 3,
        "max_lengths": [4, 5],
        "inner_dense_size": None if values_1d else 6,
        "length_mode": "mixed_zero",
    }
    values, offsets, y_0, y_1 = make_case_tensors(case, values_dtype, torch.int64)
    grad_output = torch.empty_like(values).uniform_(-1.0, 1.0)
    ref_y_grad = jagged_to_padded_dense_reference(grad_output, offsets, case["max_lengths"])
    grad_x, grad_y0, grad_y1 = op_namespace(namespace).jagged_dense_dense_elementwise_add_jagged_output_backward(
        grad_output.to(DEVICE),
        values.to(DEVICE),
        [offset.to(DEVICE) for offset in offsets],
        y_0.to(DEVICE),
        y_1.to(DEVICE),
    )
    assert grad_x.shape == values.shape
    assert grad_y0.shape == y_0.shape
    assert grad_y1.shape == y_1.shape
    assert_close(grad_x, grad_output, values_dtype)
    assert_close(grad_y0, ref_y_grad, values_dtype)
    assert_close(grad_y1, ref_y_grad, values_dtype)
