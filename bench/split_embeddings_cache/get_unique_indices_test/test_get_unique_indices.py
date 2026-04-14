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
import random
import sysconfig

import numpy as np
import pytest
import torch
import torch_npu
import fbgemm_ascend

SEED = 142


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)


def golden_get_unique_indices_impl(linear_indices, compute_count, return_inverse):
    arr = linear_indices.cpu().numpy()

    if return_inverse:
        unique_values, inverse_indices, counts = np.unique(
            arr, return_inverse=True, return_counts=True
        )
    else:
        unique_values, counts = np.unique(arr, return_counts=True)

    unique_tensor = torch.from_numpy(unique_values)
    length_tensor = torch.tensor([len(unique_values)], dtype=torch.int32)

    if compute_count:
        count_tensor = torch.from_numpy(counts.astype(np.int32))
    else:
        count_tensor = None

    if return_inverse:
        inverse_tensor = torch.from_numpy(np.argsort(arr, kind="stable").astype(np.int32))
        return unique_tensor, length_tensor, count_tensor, inverse_tensor

    return unique_tensor, length_tensor, count_tensor


def call_get_unique_indices(linear_indices, max_indices, compute_count, return_inverse):
    op_name = (
        "get_unique_indices_with_inverse"
        if return_inverse
        else "get_unique_indices"
    )

    op = getattr(torch.ops.fbgemm, op_name)

    if return_inverse:
        outputs = op(linear_indices.to("npu"), max_indices, compute_count, True)
    else:
        outputs = op(linear_indices.to("npu"), max_indices, compute_count)

    torch_npu.npu.synchronize()

    if return_inverse:
        unique, length, count, inverse = outputs
        return (
            unique.cpu(),
            length.cpu(),
            count.cpu() if count is not None else None,
            inverse.cpu() if inverse is not None else None,
        )

    unique, length, count = outputs
    return unique.cpu(), length.cpu(), count.cpu() if count is not None else None


def make_input(length, max_value, dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)

    # 采用左闭右开区间 [0, max_value) 采样
    values = torch.randint(
        0,
        max_value,
        (length,),
        dtype=torch.int64,
        generator=generator,
    )
    return values.to(dtype)


@pytest.mark.parametrize("length", [0, 16, 256, 4096, 65536, 1 << 18, 1 << 20, 1 << 23, 1 << 25])
@pytest.mark.parametrize("max_value", [8, 64, 512, 4096, 32768, 1 << 17, 1 << 19, 1 << 23])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("return_inverse", [False, True])
@pytest.mark.parametrize("compute_count", [False, True])
def test_get_unique_indices(
    length,
    max_value,
    dtype,
    return_inverse,
    compute_count,
):

    linear_indices = make_input(length, max_value, dtype)

    max_indices = max_value

    result = call_get_unique_indices(
        linear_indices,
        max_indices,
        compute_count,
        return_inverse,
    )

    expected = golden_get_unique_indices_impl(
        linear_indices,
        compute_count,
        return_inverse,
    )

    if return_inverse:
        r_unique, r_len, r_count, r_inverse = result
        e_unique, e_len, e_count, e_inverse = expected
    else:
        r_unique, r_len, r_count = result
        e_unique, e_len, e_count = expected

    unique_length = int(r_len.item())

    assert unique_length == int(e_len.item())

    assert torch.equal(
        r_unique[:unique_length],
        e_unique,
    )

    if compute_count:
        assert torch.equal(
            r_count[:unique_length],
            e_count,
        )

    if return_inverse:
        assert r_inverse is not None
        assert torch.equal(r_inverse, e_inverse)
