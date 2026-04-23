#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
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
import sysconfig
import itertools

import numpy as np
import torch
import hypothesis.strategies as st
import fbgemm_gpu
import fbgemm_ascend


def npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available() and torch.npu.device_count() > 0


if npu_available():
    DEVICE = "npu:0"
    torch.npu.set_device(DEVICE)

# Used for `@unittest.skipIf`
npu_unavailable: tuple[bool, str] = (
    not npu_available(),
    "NPU is not available or no NPUs detected",
)


def cpu_and_maybe_npu() -> st.SearchStrategy:
    return st.sampled_from(
        [torch.device("cpu")] + ([torch.device("npu")] if npu_available() else [])
    )


def generate_jagged_tensor(
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device: torch.device,
        fold_inner_dense: bool = False,
        # dynamo to mark the input as dynamic shape to make sure symbolic
        # shape is generated
        mark_dynamic: bool = False,
) -> tuple[torch.Tensor, list[torch.LongTensor], np.typing.NDArray]:
    max_lengths = np.random.randint(low=1, high=10, size=(num_jagged_dim,))
    x_offsets: list[torch.LongTensor] = []
    num_lengths = outer_dense_size
    for d in range(num_jagged_dim):
        # Sometimes length[i] exceed max_L meaning jagged->dense will be
        # truncation vs. padding
        lengths = torch.randint(
            # PT2 specialize 0/1 dims as non-symbolic shape. So we need
            # to make it non 0/1 for testing. In real cases it'll likelyl
            # not 0/1 anyway (if so, they'll be recompiled)
            low=0 if not mark_dynamic else 1,
            high=max_lengths[d] * 2,
            size=(num_lengths,),
            device=device,
        )
        offset = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        if mark_dynamic:
            torch._dynamo.mark_dynamic(offset, 0)
        x_offsets.append(offset)
        num_lengths = x_offsets[-1][-1].item()

    x_values = torch.rand(
        x_offsets[-1][-1] * inner_dense_size,
        dtype=dtype,
        device=device,
    )
    if inner_dense_size != 1 or not fold_inner_dense:
        x_values = x_values.reshape(x_offsets[-1][-1].item(), inner_dense_size)

    if mark_dynamic:
        for i in range(inner_dense_size):
            torch._dynamo.mark_dynamic(x_values, i)

    return x_values, x_offsets, max_lengths


def to_padded_dense(
        values: torch.Tensor,
        offsets: list[torch.LongTensor],
        max_lengths: np.typing.NDArray,
        padding_value: float = 0,
) -> torch.Tensor:
    outer_dense_size = len(offsets[0]) - 1
    # canonicalize by unsqueeze the last dim if the inner dense dimension
    # is 1 and folded.
    inner_dense_size = 1 if values.ndim == 1 else values.size(-1)
    dense = torch.empty(
        (outer_dense_size,) + tuple(max_lengths) + (inner_dense_size,),
        dtype=values.dtype,
        device=values.device,
    )
    for i in range(outer_dense_size):
        for jagged_coord in itertools.product(
                *(list(range(max_l)) for max_l in max_lengths)
        ):
            cur_offset = i
            is_zero = False
            for d in range(len(max_lengths)):
                begin = offsets[d][cur_offset].item()
                end = offsets[d][cur_offset + 1].item()
                if jagged_coord[d] >= end - begin:
                    is_zero = True
                    break
                cur_offset = begin + jagged_coord[d]
            dense[(i,) + jagged_coord] = (
                padding_value
                if is_zero
                else values[cur_offset]
            )
    return dense.squeeze(-1) if values.ndim == 1 else dense
