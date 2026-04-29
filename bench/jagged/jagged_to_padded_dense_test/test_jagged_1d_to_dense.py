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

import fbgemm_ascend

from test_comm_utils import (
    Scenario,
    VALUES_DATA_TYPES,
    generate_jagged_tensor,
)

NPU_ENABLE = False if torch.cuda.is_available() else True

if NPU_ENABLE:
    import torch_npu

    DEVICE = "npu:0"
else:
    DEVICE = "cuda:0"


def generate_jagged_1d_tensor(batch_size, max_seq_len, data_types):
    """
    生成 1D 不规则(Jagged)张量测试数据，用于 jagged_1d_to_dense。
    Returns:
        values_1d: 1D tensor [total_sequences]
        seq_offsets: 序列偏移量数组 [B+1]
        total_sequences: 总序列长度
    """
    jagged_tensor, seq_offsets, total_sequences = generate_jagged_tensor(
        batch_size, max_seq_len, num_heads=1, attention_dim=1, data_types=data_types
    )
    # 通用的生成方法generate_jagged_tensor出来的shape是(generate_jagged_tensor, 1, 1),所以这里要squeeze成1D的
    values_1d = jagged_tensor.squeeze(-1).squeeze(-1)
    return values_1d, seq_offsets, total_sequences


def jagged_1d_to_dense_wrapper(values, offsets, max_sequence_length, padding_value):
    return Jagged1DToDense.apply(values, offsets, max_sequence_length, padding_value)


class Jagged1DToDense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, offsets, max_sequence_length, padding_value):
        ctx.save_for_backward(offsets)
        ctx.total_L = values.shape[0]

        return torch.ops.fbgemm.jagged_1d_to_dense(
            values=values.to(DEVICE),
            offsets=offsets.to(DEVICE),
            max_sequence_length=max_sequence_length,
            padding_value=padding_value,
        )

    @staticmethod
    def backward(ctx, grad_output):
        offsets = list(ctx.saved_tensors)
        total_L = ctx.total_L
        if total_L is None:
            total_L = offsets[0][-1].item()

        grad_values = torch.ops.fbgemm.jagged_to_padded_dense_backward(
            grad_output.to(DEVICE),
            [offsets[0].to(DEVICE)],
            total_L,
        )
        return grad_values, None, None, None, None


# 简单 param 组合
test_params_1d = {
    "batch_size": [2, 4],
    "max_seq_len": [64, 128],
    "values_data_type": list(VALUES_DATA_TYPES),
    "offsets_data_type": [torch.int32, torch.int64],
}


@pytest.mark.parametrize(
    "batch_size, max_seq_len, values_data_type, offsets_data_type",
    [tuple(v) for v in itertools.product(*test_params_1d.values())],
)
def test_jagged_1d_to_dense(batch_size, max_seq_len, values_data_type, offsets_data_type):
    """
    测试 jagged_1d_to_dense 算子
    1. 生成 1D jagged 数据
    2. 计算 CPU golden 参考
    3. 调用 NPU jagged_1d_to_dense
    4. 前向比对
    5. 验证自动求导与梯度一致
    """
    data_types = (values_data_type, offsets_data_type)
    values_1d, seq_offsets, _ = generate_jagged_1d_tensor(
        batch_size, max_seq_len, data_types
    )
    fbgemm_offsets = torch.from_numpy(seq_offsets)

    # ===== 前向传播验证 =====
    fbgemm_dense = torch.ops.fbgemm.jagged_1d_to_dense(
        values_1d,
        fbgemm_offsets,
        max_seq_len,
        padding_value=0,
    )

    npu_dense = torch.ops.fbgemm.jagged_1d_to_dense(
        values_1d.to(DEVICE),
        fbgemm_offsets.to(DEVICE),
        max_seq_len,
        padding_value=0,
    )

    assert torch.equal(
        fbgemm_dense.reshape(-1),
        npu_dense.cpu().reshape(-1),
    ), f"NPU结果与参考不匹配\nFBGEMM ref:\n{fbgemm_dense}\nNPU:\n{npu_dense.cpu()}"

    # ===== 反向传播验证（仅浮点类型支持 requires_grad，int 类型不参与反向）=====
    if torch.is_floating_point(values_1d):
        values_1d_npu = (
            values_1d.clone().to(values_data_type).to(DEVICE).requires_grad_(True)
        )
        values_1d_npu_py = (
            values_1d.clone().to(values_data_type).to(DEVICE).requires_grad_(True)
        )

        npu_dense_for_grad = torch.ops.fbgemm.jagged_1d_to_dense(
            values_1d_npu,
            fbgemm_offsets.to(DEVICE),
            max_seq_len,
            padding_value=0,
        )

        npu_py_dense_for_grad = jagged_1d_to_dense_wrapper(
            values_1d_npu_py,
            fbgemm_offsets.to(DEVICE),
            max_seq_len,
            padding_value=0,
        )

        grad_output = torch.randn_like(
            npu_dense_for_grad, dtype=values_data_type, device=DEVICE
        )

        npu_dense_for_grad.backward(grad_output.to(DEVICE))
        npu_grad_input = values_1d_npu.grad

        npu_py_dense_for_grad.backward(grad_output.to(DEVICE))
        npu_py_grad_input = values_1d_npu_py.grad

        assert torch.equal(
            npu_py_grad_input.cpu(),
            npu_grad_input.cpu(),
        ), f"梯度不匹配\nPython 梯度:\n{npu_py_grad_input.cpu()}\nNPU 梯度:\n{npu_grad_input.cpu()}"


SCENARIOS_1D = [
    # 浮点常规场景：覆盖多组 batch / max_seq_len / dtype 组合
    Scenario(
        scenario_id="fp32_padding_values",
        base={"dtype": torch.float32},
        sweep={
            "batch_size": [2, 4],
            "max_seq_len": [64, 128],
            "dtype": [
                torch.float32,
                torch.int64,
                torch.float16,
                torch.bfloat16,
                torch.int32,
            ],
        },
    ),
    # 浮点边界场景：小/大 batch 与极端 max_seq_len
    Scenario(
        scenario_id="fp32_edge_cases",
        base={"dtype": torch.float32},
        sweep={"batch_size": [1, 8, 16], "max_seq_len": [1, 512, 1024]},
    ),
    # 浮点精度场景：极小/极大维度覆盖
    Scenario(
        scenario_id="fp32_precision",
        base={"dtype": torch.float32, "batch_size": 2, "max_seq_len": 32},
        sweep={},
    ),
    # int64 基线：覆盖多种 batch / seq / head 组合
    Scenario(
        scenario_id="int64_baseline",
        base={"dtype": torch.int64},
        sweep={
            "batch_size": [2, 4],
            "max_seq_len": [64, 128],
        },
    ),
    # 超大shape测试
    Scenario(
        scenario_id="large_shape_test",
        base={
            "max_seq_len": 4096,
            "batch_size": 511,
        },
        sweep={
            "dtype": [torch.int64, torch.float32],
        },
    ),
]

ALL_SCENARIO_CASES_1D = [{"id": sc.scenario_id, "runs": sc.expand()} for sc in SCENARIOS_1D]


@pytest.mark.parametrize("case", ALL_SCENARIO_CASES_1D, ids=lambda c: c["id"])
def test_jagged_1d_to_dense_scenarios(case):
    for idx, run in enumerate(case["runs"]):
        run = dict(run)
        run.setdefault("case_tag", f"{case['id']}_run{idx}")
        run_case(run)


def run_case(params):
    """场景测试的 run_case：仅前向，用 jagged_1d_to_dense CPU实现作为参考."""
    dtype = params.get("dtype", torch.float32)
    batch_size = params["batch_size"]
    max_seq_len = params["max_seq_len"]
    data_types = (dtype, torch.int64)
    values_1d, seq_offsets, _ = generate_jagged_1d_tensor(
        batch_size, max_seq_len, data_types
    )
    offsets_tensor = torch.from_numpy(seq_offsets)

    reference_dense = torch.ops.fbgemm.jagged_1d_to_dense(
        values_1d,
        offsets_tensor,
        max_seq_len,
        padding_value=0,
    )  # [B, max_seq_len]

    npu_dense = torch.ops.fbgemm.jagged_1d_to_dense(
        values_1d.to(DEVICE),
        offsets_tensor.to(DEVICE),
        max_seq_len,
        padding_value=0,
    )
    npu_cpu = npu_dense.cpu()

    assert torch.equal(
        reference_dense.reshape(-1),
        npu_cpu.reshape(-1),
    ), (
        f"NPU结果与参考结果不匹配\n"
        f"参考 shape: {reference_dense.shape}, NPU shape: {npu_cpu.shape}\n"
        f"参考:\n{reference_dense}\nNPU:\n{npu_cpu}"
    )
