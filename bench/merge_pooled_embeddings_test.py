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
import itertools
import logging
import sysconfig

from pathlib import Path
import pytest

import numpy as np
import torch_npu
import torch

import fbgemm_gpu
import fbgemm_ascend

torch.npu.config.allow_internal_format = False
numNpus = 8


def get_fused_loss_op(input_tensors, dst):
    result = torch.ops.mxrec.all_to_one_device(
        input_tensors,
        dst
    )
    return result


def make_pitched_tensor(
        height: int,
        width: int,
        dtype: torch.dtype,
        device,
        alignment: int = 256
) -> torch.Tensor:
    elem_size = (
        torch.finfo(dtype).bits // 8
        if dtype.is_floating_point
        else torch.iinfo(dtype).bits // 8
    )
    width_bytes = width * elem_size
    pitch_bytes = int(np.ceil(width_bytes /alignment) * alignment)
    pitch_elems = pitch_bytes // elem_size
    storage = torch.randn((height, pitch_elems), dtype=dtype, device=device)
    view = storage[:, :width]
    return view.contiguous() if alignment == 0 else view


@pytest.mark.parametrize("dst", ["npu:0"])
@pytest.mark.parametrize("pitched", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_all_to_one_device(dst, pitched, dtype):
    dstDevice = torch.device(dst)
    with torch.npu.device(dstDevice):
        if pitched:
            inputs = [
                make_pitched_tensor(10, 20, dtype, "cpu", alignment=256)
                for _ in range(numNpus)
            ]
        else:
            inputs = [torch.randn(10, 20, dtype=dtype, device="cpu") for _ in range(numNpus)]

        npu_inputs = [
            input.to(f"npu:{i % numNpus}")
            for i, input in enumerate(inputs)
        ]
        npu_outpus = torch.ops.fbgemm.all_to_one_device(npu_inputs, dstDevice)
        for i, o in zip(inputs, npu_outpus):
            torch.equal(o.to("cpu"), i)


@pytest.mark.parametrize("pitched", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_all_to_one_device_cpu(pitched, dtype):
    if pitched:
        inputs = [
            make_pitched_tensor(10, 20, dtype, "cpu", alignment=256)
            for _ in range(numNpus)
        ]
    else:
        inputs = [torch.randn(10, 20, dtype=dtype, device="cpu") for _ in range(numNpus)]

    npu_inputs = [
        input.to(f"npu:{i % numNpus}")
        for i, input in enumerate(inputs)
    ]
    npu_outpus = torch.ops.fbgemm.all_to_one_device(npu_inputs, "cpu")
    for i, o in zip(inputs, npu_outpus):
        torch.equal(o, i)