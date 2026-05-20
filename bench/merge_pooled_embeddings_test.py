#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
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

import numpy as np
import torch_npu  # noqa:F401
import torch

import fbgemm_ascend  # noqa:F401

torch.npu.config.allow_internal_format = False
numNpus = 8


def get_fused_loss_op(input_tensors, dst):
    result = torch.ops.mxrec.all_to_one_device(input_tensors, dst)
    return result


def make_pitched_tensor(height: int, width: int, dtype: torch.dtype, device, alignment: int = 256) -> torch.Tensor:
    elemSize = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else torch.iinfo(dtype).bits // 8
    widthBytes = width * elemSize
    pitchBytes = int(np.ceil(widthBytes / alignment) * alignment)
    pitchElems = pitchBytes // elemSize
    storage = torch.randn((height, pitchElems), dtype=dtype, device=device)
    storageView = storage[:, :width]
    return storageView.contiguous() if alignment == 0 else storageView


@pytest.mark.parametrize("dst", ["npu:0"])
@pytest.mark.parametrize("pitched", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_all_to_one_device(dst, pitched, dtype):
    dstDevice = torch.device(dst)
    with torch.npu.device(dstDevice):
        if pitched:
            inputs = [make_pitched_tensor(10, 20, dtype, "cpu", alignment=256) for _ in range(numNpus)]
        else:
            inputs = [torch.randn(10, 20, dtype=dtype, device="cpu") for _ in range(numNpus)]

        npu_inputs = [input.to(f"npu:{i % numNpus}") for i, input in enumerate(inputs)]
        npu_outpus = torch.ops.fbgemm.all_to_one_device(npu_inputs, dstDevice)
        for i, o in zip(inputs, npu_outpus):
            torch.equal(o.to("cpu"), i)


@pytest.mark.parametrize("pitched", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_all_to_one_device_cpu(pitched, dtype):
    if pitched:
        inputs = [make_pitched_tensor(10, 20, dtype, "cpu", alignment=256) for _ in range(numNpus)]
    else:
        inputs = [torch.randn(10, 20, dtype=dtype, device="cpu") for _ in range(numNpus)]

    npu_inputs = [input.to(f"npu:{i % numNpus}") for i, input in enumerate(inputs)]
    npu_outpus = torch.ops.fbgemm.all_to_one_device(npu_inputs, "cpu")
    for i, o in zip(inputs, npu_outpus):
        torch.equal(o, i)


@pytest.mark.parametrize("dst", ["npu:0"])
@pytest.mark.parametrize("cat_dim", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_merge_pooled_embeddings(dst, cat_dim, dtype):
    """测试 merge_pooled_embeddings 算子在不同NPU设备上的功能"""
    batch_size = 32
    embedding_dim = 128
    num_embeddings = 4

    dstDevice = torch.device(dst)
    uncat_dim_size = batch_size if cat_dim == 1 else embedding_dim

    # 创建分布在不同NPU设备上的tensor
    npu_tensors = []
    for i in range(num_embeddings):
        with torch.npu.device(f"npu:{i % numNpus}"):
            if cat_dim == 1:
                tensor = torch.randn(batch_size, embedding_dim, dtype=dtype)
            else:
                tensor = torch.randn(embedding_dim, batch_size, dtype=dtype)
            npu_tensors.append(tensor)

    # 调用 merge_pooled_embeddings
    merged = torch.ops.fbgemm.merge_pooled_embeddings(npu_tensors, uncat_dim_size, dstDevice, cat_dim=cat_dim)

    # 验证输出形状
    if cat_dim == 1:
        expected_shape = (batch_size, embedding_dim * num_embeddings)
    else:
        expected_shape = (embedding_dim * num_embeddings, batch_size)
    assert merged.shape == expected_shape, f"Expected shape {expected_shape}, got {merged.shape}"

    # 验证输出设备
    assert merged.device == dstDevice


@pytest.mark.parametrize("cat_dim", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_merge_pooled_embeddings_cpu(cat_dim, dtype):
    """测试 merge_pooled_embeddings 算子目标设备为CPU的情况"""
    batch_size = 32
    embedding_dim = 128
    num_embeddings = 4

    uncat_dim_size = batch_size if cat_dim == 1 else embedding_dim

    # 创建分布在不同NPU设备上的tensor
    npu_tensors = []
    for i in range(num_embeddings):
        with torch.npu.device(f"npu:{i % numNpus}"):
            if cat_dim == 1:
                tensor = torch.randn(batch_size, embedding_dim, dtype=dtype)
            else:
                tensor = torch.randn(embedding_dim, batch_size, dtype=dtype)
            npu_tensors.append(tensor)

    # 调用 merge_pooled_embeddings，目标设备为CPU
    merged = torch.ops.fbgemm.merge_pooled_embeddings(npu_tensors, uncat_dim_size, torch.device("cpu"), cat_dim=cat_dim)

    # 验证输出形状
    if cat_dim == 1:
        expected_shape = (batch_size, embedding_dim * num_embeddings)
    else:
        expected_shape = (embedding_dim * num_embeddings, batch_size)
    assert merged.shape == expected_shape, f"Expected shape {expected_shape}, got {merged.shape}"

    # 验证输出设备
    assert merged.device.type == "cpu"


@pytest.mark.parametrize("dst", ["npu:0"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_merge_pooled_embeddings_same_device(dst, dtype):
    """测试所有输入tensor都在目标设备上的情况"""
    batch_size = 32
    embedding_dim = 128
    num_embeddings = 4

    dstDevice = torch.device(dst)
    uncat_dim_size = batch_size

    # 所有tensor都在目标设备上
    with torch.npu.device(dstDevice):
        npu_tensors = [
            torch.randn(batch_size, embedding_dim, dtype=dtype, device=dstDevice) for _ in range(num_embeddings)
        ]

    # 调用 merge_pooled_embeddings
    merged = torch.ops.fbgemm.merge_pooled_embeddings(npu_tensors, uncat_dim_size, dstDevice, cat_dim=1)

    # 验证输出形状
    expected_shape = (batch_size, embedding_dim * num_embeddings)
    assert merged.shape == expected_shape, f"Expected shape {expected_shape}, got {merged.shape}"

    # 验证输出设备
    assert merged.device == dstDevice


def test_merge_pooled_embeddings_compare_with_native():
    """测试merge_pooled_embeddings结果与native cat的一致性"""
    batch_size = 32
    embedding_dim = 128
    num_embeddings = 4
    dst = "npu:0"

    dstDevice = torch.device(dst)
    uncat_dim_size = batch_size

    # 创建分布在不同NPU设备上的tensor
    npu_tensors = []
    for i in range(num_embeddings):
        with torch.npu.device(f"npu:{i % numNpus}"):
            tensor = torch.randn(batch_size, embedding_dim, dtype=torch.float32)
            npu_tensors.append(tensor)

    # 调用 merge_pooled_embeddings
    merged = torch.ops.fbgemm.merge_pooled_embeddings(npu_tensors, uncat_dim_size, dstDevice, cat_dim=1)

    # 将所有tensor复制到目标设备并用native cat对比
    with torch.npu.device(dstDevice):
        gathered = [t.to(dstDevice) for t in npu_tensors]
        expected = torch.cat(gathered, dim=1)

    # 验证结果一致性（允许一定的浮点误差）
    assert torch.allclose(merged, expected, atol=1e-3), "Results do not match!"
