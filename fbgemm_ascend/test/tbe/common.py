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
import enum
import sysconfig
import os
from typing import Optional, TypeVar, Callable

import numpy as np
import torch
import torch_npu
import hypothesis.strategies as st
import fbgemm_gpu
from fbgemm_gpu import split_table_batched_embeddings_ops_common
from fbgemm_gpu import split_table_batched_embeddings_ops_training

MAX_EXAMPLES = 40

Deviceable = TypeVar(
    "Deviceable", torch.nn.EmbeddingBag, torch.nn.Embedding, torch.Tensor
)

TEST_WITH_ROCM: bool = os.getenv("FBGEMM_TEST_WITH_ROCM", "0") == "1"
torch.cuda.current_device = torch.npu.current_device
torch.cuda.get_device_properties = torch.npu.get_device_properties
torch.Tensor.cuda = torch.Tensor.npu


class ComputeDevice(enum.IntEnum):
    CPU = 0
    CUDA = 1
    MTIA = 2
    NPU = 3


split_table_batched_embeddings_ops_training.ComputeDevice = ComputeDevice


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


def use_cpu_strategy() -> st.SearchStrategy[bool]:
    return (
        st.booleans()
        if (npu_available() and not TEST_WITH_ROCM)
        # fmt: off
        else st.just(False) if (npu_available() and TEST_WITH_ROCM) else st.just(True)
        # fmt: on
    )


def format_ref_tensors_in_mixed_B_layout(
        ref_tensors: list[torch.Tensor], Bs_rank_feature: list[list[int]]
) -> torch.Tensor:
    # Relayout the reference tensor
    # Jagged dimension: (rank, table, local batch)
    num_ranks = len(Bs_rank_feature[0])
    split_tensors = [[] for _ in range(num_ranks)]  # shape (rank, table)
    for t, ref_tensor in enumerate(ref_tensors):
        assert ref_tensor.shape[0] == sum(Bs_rank_feature[t])
        tensors = ref_tensor.split(Bs_rank_feature[t])
        for r, tensor in enumerate(tensors):
            split_tensors[r].append(tensor.flatten())
    concat_list = []
    for r in range(num_ranks):
        concat_list += split_tensors[r]
    return torch.cat(concat_list, dim=0)


def gen_mixed_B_batch_sizes(
        B: int, T: int, num_ranks: Optional[int] = None
) -> tuple[list[list[int]], list[int]]:
    if num_ranks is None:
        num_ranks = np.random.randint(low=1, high=4)
    low = max(int(0.25 * B), 1)
    high = int(B)
    if low == high:
        Bs_rank_feature = [[B] * num_ranks for _ in range(T)]
    else:
        Bs_rank_feature = [
            np.random.randint(low=low, high=high, size=num_ranks).tolist()
            for _ in range(T)
        ]
    Bs = [sum(Bs_feature) for Bs_feature in Bs_rank_feature]
    return Bs_rank_feature, Bs


def to_device(t: Deviceable, use_cpu: bool) -> Deviceable:
    if use_cpu:
        return t.cpu()
    elif torch.cuda.is_available():
        return t.cuda()
    elif torch.npu.is_available():
        return t.npu()
    else:
        return t.to(device="mtia")


def get_offsets_from_dense(indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    (B, L) = indices.size()
    return (
        indices.contiguous().view(-1),
        torch.tensor(
            np.cumsum(np.asarray([0] + [L for _ in range(B)])[:-1]).astype(np.int64)
        ),
    )


def b_indices(
        b: Callable[..., torch.Tensor],
        x: torch.Tensor,
        per_sample_weights: Optional[torch.Tensor] = None,
        use_cpu: bool = False,
        do_pooling: bool = True,
) -> torch.Tensor:
    (indices, offsets) = get_offsets_from_dense(x)
    if do_pooling:
        return b(
            to_device(indices, use_cpu),
            to_device(offsets, use_cpu),
            per_sample_weights=per_sample_weights,
        )
    else:
        return b(to_device(indices, use_cpu))


def get_table_batched_offsets_from_dense(
        merged_indices: torch.Tensor,
        L: Optional[int] = None,
        total_B: Optional[int] = None,
        use_cpu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if L is None and total_B is None:
        (T, B, L) = merged_indices.size()
        total_B = T * B
    lengths = np.ones(total_B) * L
    return (
        to_device(merged_indices.contiguous().view(-1), use_cpu),
        to_device(
            torch.tensor(([0] + np.cumsum(lengths).tolist())).long(),
            use_cpu,
        ),
    )


def get_new_embedding_location(
        device: torch.device, cache_load_factor: float
) -> split_table_batched_embeddings_ops_common.EmbeddingLocation:
    """
    Based on the cache_load_factor and device, return the embedding location intended
    for the TBE weights.
    """
    # Only support CPU and NPU device
    assert device.type == "cpu" or device.type == "npu"
    if cache_load_factor < 0 or cache_load_factor > 1:
        raise ValueError(
            f"cache_load_factor must be between 0.0 and 1.0, got {cache_load_factor}"
        )

    if device.type == "cpu":
        return split_table_batched_embeddings_ops_common.EmbeddingLocation.HOST
    # UVM only
    elif cache_load_factor == 0:
        return split_table_batched_embeddings_ops_common.EmbeddingLocation.MANAGED
    # HBM only
    elif cache_load_factor == 1.0:
        return split_table_batched_embeddings_ops_common.EmbeddingLocation.DEVICE
    # UVM caching
    else:
        return split_table_batched_embeddings_ops_common.EmbeddingLocation.MANAGED_CACHING


split_table_batched_embeddings_ops_common.get_new_embedding_location = get_new_embedding_location
