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

import random
import pytest
import torch
import torch_npu
import fbgemm_gpu
import fbgemm_ascend

torch.npu.config.allow_internal_format = False

DEVICE = "npu:0"
# 检查 NPU 是否可用
npu_available = torch.npu.is_available()
if npu_available:
    torch_npu.npu.set_device(DEVICE)


def ref_forward(input_group, indices_group):
    """参考实现"""
    output_group = []
    for inp, idx in zip(input_group, indices_group):
        output_group.append(torch.index_select(inp, 0, idx))
    return output_group


def op_forward(input_group, indices_group):
    """算子实现"""
    inp_npu = [t.to(DEVICE) for t in input_group]
    idx_npu = [i.to(DEVICE) for i in indices_group]
    out = torch.ops.fbgemm.group_index_select_dim0(inp_npu, idx_npu)
    return [o.cpu() for o in out] if isinstance(out, list) else [out.cpu()]


def ref_backward(input_group, indices_group, grad_group):
    """参考实现：计算梯度"""
    inputs = [t.clone().detach().requires_grad_(True) for t in input_group]
    outputs = [torch.index_select(inp, 0, idx) for inp, idx in zip(inputs, indices_group)]
    for out, grad in zip(outputs, grad_group):
        out.backward(grad)
    return [inp.grad for inp in inputs]


def op_backward(input_group, indices_group, grad_group):
    """算子实现：计算梯度"""
    inputs = [t.clone().detach().to(DEVICE).requires_grad_(True) for t in input_group]
    idx_npu = [i.to(DEVICE) for i in indices_group]
    grad_npu = [g.to(DEVICE) for g in grad_group]
    
    outputs = torch.ops.fbgemm.group_index_select_dim0(inputs, idx_npu)
    if not isinstance(outputs, list):
        outputs = [outputs]
    
    # 全部 cat 后统一 backward
    cat_out = torch.concat([o.flatten() for o in outputs])
    cat_grad = torch.concat([g.flatten() for g in grad_npu])
    cat_out.backward(cat_grad)
    
    return [inp.grad.cpu() for inp in inputs]


@pytest.mark.parametrize("num_groups", [1, 4, 8])
@pytest.mark.parametrize("num_indices", [1, 16, 64])
@pytest.mark.parametrize("shape", [[8], [16, 16]])
def test_forward(num_groups, num_indices, shape):
    input_group = []
    indices_group = []
    
    for _ in range(num_groups):
        num_rows = random.randint(1, 32)
        tensor_shape = (num_rows,) + tuple(shape)
        indices = torch.randint(0, num_rows, (num_indices,))
        indices_group.append(indices)
        input_group.append(torch.rand(tensor_shape))
    
    ref_out = ref_forward(input_group, indices_group)
    op_out = op_forward(input_group, indices_group)
    
    for i, (ref, op) in enumerate(zip(ref_out, op_out)):
        assert torch.allclose(ref, op, rtol=1e-5, atol=1e-6), f"Group {i} mismatch"


@pytest.mark.parametrize("num_groups", [1, 4, 8])
@pytest.mark.parametrize("num_indices", [1, 16, 64])
@pytest.mark.parametrize("shape", [[8], [16, 16]])
def test_backward(num_groups, num_indices, shape):
    input_group = []
    indices_group = []
    grad_group = []
    
    for _ in range(num_groups):
        num_rows = random.randint(1, 32)
        tensor_shape = (num_rows,) + tuple(shape)
        indices = torch.randint(0, num_rows, (num_indices,))
        indices_group.append(indices)
        input_group.append(torch.rand(tensor_shape))
        grad_group.append(torch.rand((num_indices,) + tuple(shape)))
    
    ref_grad = ref_backward(input_group, indices_group, grad_group)
    op_grad = op_backward(input_group, indices_group, grad_group)
    
    for i, (ref, op) in enumerate(zip(ref_grad, op_grad)):
        assert torch.allclose(ref, op, rtol=1e-5, atol=1e-6), f"Group {i} gradient mismatch"


def test_empty_input():
    """测试空输入"""
    input_group = []
    indices_group = []

    with pytest.raises(Exception):
        torch.ops.fbgemm.group_index_select_dim0(input_group, indices_group)


def test_empty_indices():
    """测试空索引列表"""
    input_group = [torch.rand(10, 8).to(DEVICE)]
    indices_group = []
    
    with pytest.raises(Exception):
        torch.ops.fbgemm.group_index_select_dim0(input_group, indices_group)


def test_group_count_mismatch():
    """测试 group 数量不匹配"""
    input_group = [
        torch.tensor([[1, 2], [3, 4], [5, 6]]).to(DEVICE),
        torch.tensor([[7, 8], [9, 10]]).to(DEVICE),
        torch.tensor([[11, 12], [13, 14], [15, 16]]).to(DEVICE)
    ]
    indices_group = [
        torch.tensor([0, 2]).to(DEVICE),
        torch.tensor([0, 1]).to(DEVICE)
    ]
    
    with pytest.raises(Exception):
        torch.ops.fbgemm.group_index_select_dim0(input_group, indices_group)


def test_index_out_of_bounds():
    """测试索引越界"""
    input_group = [torch.rand(10, 8).to(DEVICE)]
    indices_group = [torch.tensor([5, 12, 3]).to(DEVICE)]
    
    with pytest.raises(Exception):
        torch.ops.fbgemm.group_index_select_dim0(input_group, indices_group)


def test_negative_index():
    """测试负索引"""
    input_group = [torch.rand(10, 8).to(DEVICE)]
    indices_group = [torch.tensor([2, -1, 5]).to(DEVICE)]
    
    with pytest.raises(Exception):
        torch.ops.fbgemm.group_index_select_dim0(input_group, indices_group)


def test_int_input_type():
    """测试 int 类型输入（不支持）"""
    input_group = [torch.randint(0, 10, (10, 8), dtype=torch.int32).to(DEVICE)]
    indices_group = [torch.randint(0, 10, (3,)).to(DEVICE)]
    
    with pytest.raises(Exception):
        torch.ops.fbgemm.group_index_select_dim0(input_group, indices_group)


def test_empty_tensor_in_group():
    """测试 group 中包含空张量"""
    input_group = [
        torch.rand(10, 8).to(DEVICE),
        torch.empty(0, 8).to(DEVICE),
        torch.rand(5, 8).to(DEVICE)
    ]
    indices_group = [
        torch.randint(0, 10, (3,)).to(DEVICE),
        torch.empty(0, dtype=torch.int64).to(DEVICE),
        torch.randint(0, 5, (2,)).to(DEVICE)
    ]

    with pytest.raises(Exception):
        torch.ops.fbgemm.group_index_select_dim0(input_group, indices_group)


def test_large_index_value():
    """测试超大索引值"""
    input_group = [torch.rand(1000, 8).to(DEVICE)]
    indices_group = [torch.tensor([999999]).to(DEVICE)]
    
    with pytest.raises(Exception):
        torch.ops.fbgemm.group_index_select_dim0(input_group, indices_group)
