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

FLOAT_VALUES_DATA_TYPES = [torch.float32, torch.float16, torch.bfloat16]
MUL_OP_VALUES_DATA_TYPES = [torch.float32, torch.float16]
OFFSETS_DATA_TYPES = [torch.int32, torch.int64]
NAMESPACES = ["fbgemm"]

FORWARD_CASES = [
    {
        "id": "1d_scalar_min_len",
        "num_jagged_dim": 1,
        "batch_size": 1,
        "max_lengths": [1],
        "inner_dense_size": 0,
        "length_mode": "full",
    },
    {
        "id": "1d_scalar_zero_and_truncated",
        "num_jagged_dim": 1,
        "batch_size": 5,
        "max_lengths": [4],
        "inner_dense_size": 0,
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
        "inner_dense_size": 0,
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
        "id": "empty_total_l_scalar",
        "num_jagged_dim": 1,
        "batch_size": 3,
        "max_lengths": [5],
        "inner_dense_size": 0,
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
        "inner_dense_size": 0,
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
        "inner_dense_size": 0,
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
        "inner_dense_size": 0,
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
    # 逐层生成 offsets：下一层的父节点数量来自上一层的总 child 数。
    offsets = []
    num_parents = batch_size
    for dim_idx, max_len in enumerate(max_lengths):
        lengths = make_lengths(num_parents, max_len, length_mode, dim_idx)
        offset = torch.cat((torch.zeros((1,), dtype=torch.int64), torch.cumsum(lengths, dim=0)))
        offsets.append(offset.to(offsets_dtype))
        num_parents = int(offset[-1].item())
    return offsets


def make_values(total_l, inner_dense_size, dtype):
    shape = (total_l, inner_dense_size)
    if dtype in [torch.int32, torch.int64]:
        return torch.randint(-100, 100, shape, dtype=dtype)
    return torch.empty(shape, dtype=dtype).uniform_(-1.0, 1.0)


def make_dense(batch_size, max_lengths, inner_dense_size, dtype, non_contiguous=False):
    # dense 的形状与算子输出保持一致，用 max_lengths 表示各 jagged 维度的 padded 上界。
    shape = (batch_size, *max_lengths, inner_dense_size)
    if dtype in [torch.int32, torch.int64]:
        dense = torch.randint(-100, 100, shape, dtype=dtype)
    else:
        dense = torch.empty(shape, dtype=dtype).uniform_(-0.5, 0.5)
    if not non_contiguous:
        return dense
    if dense.dim() < 2:
        return dense
    source = torch.empty((*shape, 2), dtype=dtype)
    if dtype in [torch.int32, torch.int64]:
        source.random_(-100, 100)
    else:
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
    dense = make_dense(
        case["batch_size"],
        case["max_lengths"],
        case["inner_dense_size"],
        values_dtype,
        non_contiguous=non_contiguous_dense,
    )
    return values, offsets, dense


def jagged_to_padded_dense_reference(values, offsets, max_lengths):
    return torch.ops.fbgemm.jagged_to_padded_dense(values, offsets, max_lengths, 0.0)


def dense_to_jagged_reference(dense, offsets):
    # 按 offsets 递归访问 dense 的有效区域，作为 backward 输出 values 梯度的 CPU 参考。
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


def run_forward_case(namespace, case, values_dtype, offsets_dtype, non_contiguous_dense=False):
    values, offsets, dense = make_case_tensors(
        case, values_dtype, offsets_dtype, non_contiguous_dense=non_contiguous_dense
    )
    ref = jagged_to_padded_dense_reference(values, offsets, case["max_lengths"]) + dense
    out = op_namespace(namespace).jagged_dense_elementwise_add(
        values.to(DEVICE),
        [offset.to(DEVICE) for offset in offsets],
        dense.to(DEVICE),
    )
    assert out.shape == dense.shape
    assert_close(out, ref, values_dtype)


def run_backward_case(namespace, case, values_dtype, offsets_dtype):
    # autograd 路径同时校验 jagged values 梯度和 dense tensor 梯度。
    values, offsets, dense = make_case_tensors(case, values_dtype, offsets_dtype)
    x_ref = values.detach().clone().requires_grad_(True)
    d_ref = dense.detach().clone().requires_grad_(True)
    ref_out = jagged_to_padded_dense_reference(x_ref, offsets, case["max_lengths"]) + d_ref
    grad_output = torch.empty_like(ref_out).uniform_(-1.0, 1.0)
    ref_out.backward(grad_output)

    x_npu = values.to(DEVICE).detach().clone().requires_grad_(True)
    d_npu = dense.to(DEVICE).detach().clone().requires_grad_(True)
    out = op_namespace(namespace).jagged_dense_elementwise_add(
        x_npu,
        [offset.to(DEVICE) for offset in offsets],
        d_npu,
    )
    out.backward(grad_output.to(DEVICE))

    assert_close(x_npu.grad, x_ref.grad, values_dtype)
    assert_close(d_npu.grad, d_ref.grad, values_dtype)


@pytest.mark.skipif(not NPU_ENABLE, reason="需要 NPU 设备")
@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize("case", FORWARD_CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("values_dtype", FLOAT_VALUES_DATA_TYPES)
@pytest.mark.parametrize("offsets_dtype", OFFSETS_DATA_TYPES)
def test_jagged_dense_elementwise_add_forward_full_coverage(namespace, case, values_dtype, offsets_dtype):
    run_forward_case(namespace, case, values_dtype, offsets_dtype)


@pytest.mark.skipif(not NPU_ENABLE, reason="需要 NPU 设备")
@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize("case", FORWARD_CASES[:8], ids=lambda c: c["id"])
@pytest.mark.parametrize("values_dtype", FLOAT_VALUES_DATA_TYPES)
def test_jagged_dense_elementwise_add_forward_non_contiguous_dense(namespace, case, values_dtype):
    run_forward_case(namespace, case, values_dtype, torch.int64, non_contiguous_dense=True)


@pytest.mark.skipif(not NPU_ENABLE, reason="需要 NPU AutogradPrivateUse1 实现")
@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize("case", BACKWARD_CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("values_dtype", FLOAT_VALUES_DATA_TYPES)
@pytest.mark.parametrize("offsets_dtype", OFFSETS_DATA_TYPES)
def test_jagged_dense_elementwise_add_backward_full_coverage(namespace, case, values_dtype, offsets_dtype):
    run_backward_case(namespace, case, values_dtype, offsets_dtype)


@pytest.mark.skipif(not NPU_ENABLE, reason="需要 NPU 设备")
@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize("case", SHAPE_STRESS_CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("values_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_jagged_dense_elementwise_add_shape_stress(namespace, case, values_dtype):
    run_forward_case(namespace, case, values_dtype, torch.int64)


def run_jagged_output_op(op, values, offsets, dense):
    if values.dim() == 1:
        out, out_offsets = op(values.unsqueeze(-1), offsets, dense.unsqueeze(-1))
        return out.squeeze(-1), out_offsets
    return op(values, offsets, dense)


def run_jagged_output_forward_case(namespace, case, values_dtype, offsets_dtype, operation, non_contiguous_dense=False):
    values, offsets, dense = make_case_tensors(
        case, values_dtype, offsets_dtype, non_contiguous_dense=non_contiguous_dense
    )
    op = getattr(op_namespace(namespace), f"jagged_dense_elementwise_{operation}")
    ref, ref_offsets = run_jagged_output_op(op, values, offsets, dense)
    out, out_offsets = op(
        values.to(DEVICE),
        [offset.to(DEVICE) for offset in offsets],
        dense.to(DEVICE),
    )

    assert out.shape == ref.shape
    assert len(out_offsets) == len(ref_offsets)
    for actual_offset, expected_offset in zip(out_offsets, ref_offsets):
        assert torch.equal(actual_offset.cpu(), expected_offset.cpu())
    assert_close(out, ref, values_dtype)


def run_jagged_output_backward_case(namespace, case, values_dtype, offsets_dtype, operation):
    values, offsets, dense = make_case_tensors(case, values_dtype, offsets_dtype)
    op = getattr(op_namespace(namespace), f"jagged_dense_elementwise_{operation}")

    x_ref = values.detach().clone().requires_grad_(True)
    d_ref = dense.detach().clone().requires_grad_(True)
    ref, ref_offsets = run_jagged_output_op(op, x_ref, offsets, d_ref)
    grad_output = torch.empty_like(ref).uniform_(-1.0, 1.0)
    ref.backward(grad_output)

    x_npu = values.to(DEVICE).detach().clone().requires_grad_(True)
    d_npu = dense.to(DEVICE).detach().clone().requires_grad_(True)
    out, out_offsets = op(
        x_npu,
        [offset.to(DEVICE) for offset in offsets],
        d_npu,
    )
    out.backward(grad_output.to(DEVICE))

    assert len(out_offsets) == len(ref_offsets)
    for actual_offset, expected_offset in zip(out_offsets, ref_offsets):
        assert torch.equal(actual_offset.cpu(), expected_offset.cpu())
    assert_close(x_npu.grad, x_ref.grad, values_dtype)
    assert_close(d_npu.grad, d_ref.grad, values_dtype)


@pytest.mark.skipif(not NPU_ENABLE, reason="需要 NPU 设备")
@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize("case", FORWARD_CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("operation", ["add_jagged_output", "mul"])
@pytest.mark.parametrize("values_dtype", FLOAT_VALUES_DATA_TYPES)
@pytest.mark.parametrize("offsets_dtype", OFFSETS_DATA_TYPES)
def test_jagged_dense_elementwise_binary_jagged_output_forward_full_coverage(
    namespace, case, operation, values_dtype, offsets_dtype
):
    run_jagged_output_forward_case(namespace, case, values_dtype, offsets_dtype, operation)


@pytest.mark.skipif(not NPU_ENABLE, reason="需要 NPU 设备")
@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize("case", FORWARD_CASES[:8], ids=lambda c: c["id"])
@pytest.mark.parametrize("operation", ["add_jagged_output", "mul"])
@pytest.mark.parametrize("values_dtype", FLOAT_VALUES_DATA_TYPES)
def test_jagged_dense_elementwise_binary_jagged_output_forward_non_contiguous_dense(
    namespace, case, operation, values_dtype
):
    run_jagged_output_forward_case(namespace, case, values_dtype, torch.int64, operation, non_contiguous_dense=True)


@pytest.mark.skipif(not NPU_ENABLE, reason="需要 NPU AutogradPrivateUse1 实现")
@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize("case", BACKWARD_CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("operation", ["add_jagged_output"])
@pytest.mark.parametrize("values_dtype", FLOAT_VALUES_DATA_TYPES)
@pytest.mark.parametrize("offsets_dtype", OFFSETS_DATA_TYPES)
def test_jagged_dense_elementwise_add_jagged_output_jagged_output_backward(
    namespace, case, operation, values_dtype, offsets_dtype
):
    run_jagged_output_backward_case(namespace, case, values_dtype, offsets_dtype, operation)


@pytest.mark.skipif(not NPU_ENABLE, reason="需要 NPU AutogradPrivateUse1 实现")
@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize("case", BACKWARD_CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("operation", ["mul"])
@pytest.mark.parametrize("values_dtype", MUL_OP_VALUES_DATA_TYPES)
@pytest.mark.parametrize("offsets_dtype", OFFSETS_DATA_TYPES)
def test_jagged_dense_elementwise_mul_jagged_output_backward(namespace, case, operation, values_dtype, offsets_dtype):
    run_jagged_output_backward_case(namespace, case, values_dtype, offsets_dtype, operation)
