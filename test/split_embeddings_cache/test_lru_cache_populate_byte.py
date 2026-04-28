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

import logging
import random
from dataclasses import dataclass
from itertools import accumulate
from typing import Dict, List, Optional, Tuple, Union

import unittest
import numpy as np
import pytest
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    CacheAlgorithm,
    construct_cache_state,
    EmbeddingLocation,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    nbit_construct_split_state,
    random_quant_scaled_tensor,
    rounded_row_size_in_bytes,
)

from fbgemm_gpu.utils.loader import load_torch_module

from torch import nn, Tensor

# NPU 环境需要保留下列 import（勿删）。
from torch_npu.contrib import transfer_to_npu
import fbgemm_ascend

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def linearize_cache_indices_python(
    cache_hash_size_cumsum: Tensor,
    indices: Tensor,
    offsets: Tensor,
    b_offsets: Optional[Tensor] = None,
    max_b: int = -1,
) -> Tensor:
    """
    Pure PyTorch linearize_cache_indices (same semantics as FBGEMM reference in
    fbgemm_gpu/test/tbe/cache/linearize_cache_indices_test.py). Use on NPU or
    anywhere torch.ops.fbgemm.linearize_cache_indices is unavailable.

    Non-cached tables use cache_hash_size_cumsum[t] < 0; those segments are
    filled with the sentinel cache_hash_size_cumsum[-1]. Pruned rows
    (indices < 0) are also set to that sentinel.
    """
    T = cache_hash_size_cumsum.numel() - 1
    if T <= 0:
        raise ValueError("cache_hash_size_cumsum must have length >= 2")

    device = indices.device
    max_offset = cache_hash_size_cumsum[-1].to(device=device, dtype=torch.int64)
    linear = indices.to(dtype=torch.int64).clone()

    offsets_list = offsets.detach().tolist()
    ch_list = cache_hash_size_cumsum.detach().tolist()

    if b_offsets is not None:
        if max_b <= 0:
            raise ValueError("max_b must be > 0 when b_offsets is provided")
        bo = b_offsets.detach().tolist()
        for t in range(T):
            indices_start = offsets_list[bo[t]]
            indices_end = offsets_list[bo[t + 1]]
            hoff = ch_list[t]
            if hoff >= 0:
                linear[indices_start:indices_end] += hoff
            else:
                linear[indices_start:indices_end] = max_offset
    else:
        total_segments = offsets.numel() - 1
        if total_segments % T != 0:
            raise ValueError(
                f"offsets length inconsistent with T: len-1={total_segments}, T={T}"
            )
        B = total_segments // T
        for t in range(T):
            indices_start = offsets_list[t * B]
            indices_end = offsets_list[(t + 1) * B]
            hoff = ch_list[t]
            if hoff >= 0:
                linear[indices_start:indices_end] += hoff
            else:
                linear[indices_start:indices_end] = max_offset

    linear[indices < 0] = max_offset
    return linear


# pyre-ignore
def benchmark_same_input(iters: int, f, *args) -> float:
    """
    Returns average execution time in milliseconds across "iters".
    """
    # Warm-up
    f(*args)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        f(*args)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iters


# pyre-ignore
def benchmark_different_inputs(f, args) -> float:
    """
    Returns average execution time in milliseconds across "iters".
    """
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for arg in args:
        f(arg)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / len(args)


def get_num_cached_tables(num_tables: int, cached_tables_ratio: float) -> int:
    """
    Controls how # of cached tables are determined based on parameters.
    """
    return round(num_tables * cached_tables_ratio)


def create_table_offsets(
    num_tables: int, cached_tables_ratio: float, num_embeddings: int
) -> Tensor:
    """
    Returns "table size cumsum", which is information of UVM caching for tables.
    """
    num_cached_tables = get_num_cached_tables(num_tables, cached_tables_ratio)
    np_list = np.arange(0, num_embeddings * num_cached_tables, num_embeddings)
    num_uncached_tables = num_tables - num_cached_tables
    while num_uncached_tables > 0:
        added = random.randint(1, num_uncached_tables)
        pos = random.randint(0, len(np_list) - 1)
        np_list = np.insert(np_list, pos, [np_list[pos]] * added)
        num_uncached_tables -= added
    cache_hash_size_cumsum: Tensor = torch.tensor(np_list).cuda()
    return cache_hash_size_cumsum


def create_embedding_specs(
    num_tables: int,
    cached_tables_ratio: float,
    num_embeddings: int,
    embedding_dims: int,
) -> List[Tuple[str, int, int, SparseType, EmbeddingLocation]]:
    """
    Returns embedding specs to be used with IntNBitTableBatchedEmbeddingBagsCodegen.
    """
    num_cached_tables = get_num_cached_tables(num_tables, cached_tables_ratio)
    num_uncached_tables = num_tables - num_cached_tables
    embedding_specs = []
    for _ in range(min(num_cached_tables, num_uncached_tables)):
        embedding_specs.append(
            (
                "",
                num_embeddings,
                embedding_dims,
                SparseType.INT8,
                EmbeddingLocation.DEVICE,
            )
        )
        embedding_specs.append(
            (
                "",
                num_embeddings,
                embedding_dims,
                SparseType.INT8,
                EmbeddingLocation.MANAGED_CACHING,
            )
        )
    if num_cached_tables > num_uncached_tables:
        for _ in range(num_cached_tables - num_uncached_tables):
            embedding_specs.append(
                (
                    "",
                    num_embeddings,
                    embedding_dims,
                    SparseType.INT8,
                    EmbeddingLocation.MANAGED_CACHING,
                )
            )
    else:
        for _ in range(num_uncached_tables - num_cached_tables):
            embedding_specs.append(
                (
                    "",
                    num_embeddings,
                    embedding_dims,
                    SparseType.INT8,
                    EmbeddingLocation.DEVICE,
                )
            )
    return embedding_specs


def create_request(
    num_tables: int, num_embeddings: int, batch: int, avg_pooling_factor: int
) -> Tuple[Tensor, Tensor]:
    """
    Returns [indices, offsets], which are inputs of embedding bags.
    """
    indices: Tensor = torch.randint(
        0, num_embeddings, (num_tables * batch * avg_pooling_factor,), dtype=torch.int32
    ).cuda()

    # Pooling factors are intentionally diversified between [1, pf / 2, pf, pf* 2, pf * 4, pf * 8].
    # where pf == avg_pooling_factor.
    pooling_factors = []
    for _ in range(num_tables - 1):
        half_avg_pooling_factor = avg_pooling_factor // 2
        if half_avg_pooling_factor > 0:
            pooling_factors.append(
                random.choices(
                    [
                        1,
                        half_avg_pooling_factor,
                        avg_pooling_factor,
                        2 * avg_pooling_factor,
                        4 * avg_pooling_factor,
                        8 * avg_pooling_factor,
                    ],
                    weights=[5, 10, 15, 1, 1, 3],
                )[0]
            )
        else:
            pooling_factors.append(
                random.choices(
                    [1, avg_pooling_factor, 2 * avg_pooling_factor], weights=[2, 20, 1]
                )[0]
            )

    # Last one is whatever is the remainder.
    curr_total_pooling_factors = sum(pooling_factors)
    pooling_factors.append(num_tables * avg_pooling_factor - curr_total_pooling_factors)

    offsets_list = [0]
    for pooling_factor in pooling_factors:
        if pooling_factor == 1:
            for _ in range(batch):
                offsets_list.append(pooling_factor)
        else:
            finish_offset = offsets_list[-1] + pooling_factor * batch
            for _ in range(batch - 1):
                selected = max(
                    int(random.gauss(pooling_factor, 0.1 * pooling_factor)), 1
                )
                last_offset = offsets_list[-1]
                offsets_list.append(last_offset + selected)
            offsets_list.append(finish_offset)
    offsets: Tensor = torch.tensor(offsets_list, dtype=torch.int32).cuda()
    return (indices, offsets)


@dataclass
class ManualLruCacheBuffers:
    """
    Tensors required by torch.ops.fbgemm.lru_cache_populate_byte, built without
    IntNBitTableBatchedEmbeddingBagsCodegen (e.g. NPU or environments where the
    module is unavailable). Shapes match FBGEMM's construct_cache_state +
    nbit_construct_split_state + _apply_cache_state logic.
    """

    weights_uvm: Tensor
    cache_hash_size_cumsum: Tensor
    total_cache_hash_size: int
    cache_index_table_map: Tensor
    weights_offsets: Tensor
    weights_tys: Tensor
    D_offsets: Tensor
    lxu_cache_state: Tensor
    lxu_cache_weights: Tensor
    lxu_state: Tensor


def _default_cache_assoc() -> int:
    if torch.version.hip is not None:
        return 64
    return 32


def _allocate_uvm_weights(
    uvm_size: int, device: torch.device, uvm_host_mapped: bool
) -> Tensor:
    # Unified memory via fbgemm.new_unified_tensor is not supported here; use
    # plain device storage (uint8 on `device`). See lru_cache_populate_byte.md §7
    # to restore new_unified_tensor when supported.
    _ = uvm_host_mapped  # used again when unified UVM path is restored (see md)
    if uvm_size == 0:
        return torch.empty(0, device=device, dtype=torch.uint8)
    return torch.zeros(uvm_size, device=device, dtype=torch.uint8)


def create_manual_lru_cache_buffers(
    embedding_specs: List[Tuple[str, int, int, SparseType, EmbeddingLocation]],
    device: torch.device,
    cache_load_factor: float,
    cache_assoc: int = -1,
    row_alignment: int = 16,
    uvm_host_mapped: bool = False,
) -> ManualLruCacheBuffers:
    """
    Mirror IntNBitTableBatchedEmbeddingBagsCodegen buffer layout for LRU populate:
    UVM split (nbit_construct_split_state), cache metadata (construct_cache_state),
    and HBM cache tensors (sets x ways, lxu_cache_weights rows = sets * ways).
    """
    if cache_assoc < 0:
        cache_assoc = _default_cache_assoc()

    rows = [e[1] for e in embedding_specs]
    locations = [e[4] for e in embedding_specs]
    t_tables = len(embedding_specs)
    feature_table_map = list(range(t_tables))

    cache_state = construct_cache_state(rows, locations, feature_table_map)
    total_cache_hash_size = cache_state.total_cache_hash_size
    if total_cache_hash_size == 0:
        raise ValueError(
            "create_manual_lru_cache_buffers requires at least one MANAGED_CACHING table."
        )
    if cache_load_factor <= 0:
        raise ValueError("cache_load_factor must be > 0 when using a non-empty cache.")

    weight_split = nbit_construct_split_state(
        embedding_specs,
        cacheable=True,
        row_alignment=row_alignment,
        cacheline_alignment=True,
    )

    weights_offsets = torch.tensor(
        [weight_split.offsets[t] for t in feature_table_map],
        device=device,
        dtype=torch.int64,
    )
    weights_tys = torch.tensor(
        [embedding_specs[t][3].as_int() for t in feature_table_map],
        device=device,
        dtype=torch.uint8,
    )
    dims = [embedding_specs[t][2] for t in feature_table_map]
    d_off = [0] + list(accumulate(dims))
    d_offsets = torch.tensor(d_off, device=device, dtype=torch.int32)

    cache_hash_size_cumsum = torch.tensor(
        cache_state.cache_hash_size_cumsum, device=device, dtype=torch.int64
    )
    cache_index_table_map = torch.tensor(
        cache_state.cache_index_table_map, device=device, dtype=torch.int32
    )

    cached_dims = [
        rounded_row_size_in_bytes(spec[2], spec[3], row_alignment)
        for spec in embedding_specs
        if spec[4] == EmbeddingLocation.MANAGED_CACHING
    ]
    max_d_cache = max(cached_dims)

    cache_sets = (
        int(total_cache_hash_size * cache_load_factor) + cache_assoc - 1
    ) // cache_assoc
    cache_sets = 1 if cache_sets == 0 else cache_sets

    lxu_cache_state = torch.zeros(
        (cache_sets, cache_assoc), device=device, dtype=torch.int64
    ).fill_(-1)
    lxu_cache_weights = torch.zeros(
        (cache_sets * cache_assoc, max_d_cache),
        device=device,
        dtype=torch.uint8,
    )
    lxu_state = torch.zeros((cache_sets, cache_assoc), device=device, dtype=torch.int64)

    weights_uvm = _allocate_uvm_weights(
        weight_split.uvm_size, device, uvm_host_mapped
    )
    if weights_uvm.numel() > 0:
        random_quant_scaled_tensor(weights_uvm.shape, device, weights_uvm)

    return ManualLruCacheBuffers(
        weights_uvm=weights_uvm,
        cache_hash_size_cumsum=cache_hash_size_cumsum,
        total_cache_hash_size=total_cache_hash_size,
        cache_index_table_map=cache_index_table_map,
        weights_offsets=weights_offsets,
        weights_tys=weights_tys,
        D_offsets=d_offsets,
        lxu_cache_state=lxu_cache_state,
        lxu_cache_weights=lxu_cache_weights,
        lxu_state=lxu_state,
    )


def _int_n_bit_codegen_class() -> type:
    from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
        IntNBitTableBatchedEmbeddingBagsCodegen,
    )

    return IntNBitTableBatchedEmbeddingBagsCodegen


def _call_lru_cache_populate_byte(
    buffers: Union[ManualLruCacheBuffers, nn.Module],
    linear_indices: Tensor,
    timestep: int,
) -> None:
    if isinstance(buffers, ManualLruCacheBuffers):
        wu, ch, tot, cim, wo, wt, d_off = (
            buffers.weights_uvm,
            buffers.cache_hash_size_cumsum,
            buffers.total_cache_hash_size,
            buffers.cache_index_table_map,
            buffers.weights_offsets,
            buffers.weights_tys,
            buffers.D_offsets,
        )
        lxs, lxw, lxu = (
            buffers.lxu_cache_state,
            buffers.lxu_cache_weights,
            buffers.lxu_state,
        )
    else:
        wu = buffers.weights_uvm
        ch = buffers.cache_hash_size_cumsum
        tot = buffers.total_cache_hash_size
        cim = buffers.cache_index_table_map
        wo = buffers.weights_offsets
        wt = buffers.weights_tys
        d_off = buffers.D_offsets
        lxs = buffers.lxu_cache_state
        lxw = buffers.lxu_cache_weights
        lxu = buffers.lxu_state

    torch.ops.fbgemm.lru_cache_populate_byte(
        wu,
        ch,
        tot,
        cim,
        wo,
        wt,
        d_off,
        linear_indices,
        lxs,
        lxw,
        timestep,
        lxu,
    )


def _lru_populate_tensors(
    buffers: Union[ManualLruCacheBuffers, nn.Module],
) -> Tuple[
    Tensor,
    Tensor,
    int,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    if isinstance(buffers, ManualLruCacheBuffers):
        return (
            buffers.weights_uvm,
            buffers.cache_hash_size_cumsum,
            buffers.total_cache_hash_size,
            buffers.cache_index_table_map,
            buffers.weights_offsets,
            buffers.weights_tys,
            buffers.D_offsets,
            buffers.lxu_cache_state,
            buffers.lxu_cache_weights,
            buffers.lxu_state,
        )
    return (
        buffers.weights_uvm,
        buffers.cache_hash_size_cumsum,
        buffers.total_cache_hash_size,
        buffers.cache_index_table_map,
        buffers.weights_offsets,
        buffers.weights_tys,
        buffers.D_offsets,
        buffers.lxu_cache_state,
        buffers.lxu_cache_weights,
        buffers.lxu_state,
    )


def verify_lxu_cache_weights_match_uvm(
    buffers: Union[ManualLruCacheBuffers, nn.Module],
    row_alignment: int = 16,
) -> None:
    """
    For every occupied slot in lxu_cache_state (linear index != -1 and < total
    cache hash size), compare the row in lxu_cache_weights with the
    corresponding bytes in weights_uvm (same layout as FBGEMM insert kernel).
    """
    (
        weights_uvm,
        cache_hash_size_cumsum,
        total_cache_hash_size,
        cache_index_table_map,
        weights_offsets,
        weights_tys,
        d_offsets,
        lxu_cache_state,
        lxu_cache_weights,
        _,
    ) = _lru_populate_tensors(buffers)

    ch_cum = cache_hash_size_cumsum
    state_flat = lxu_cache_state.reshape(-1)

    for flat_pos in range(state_flat.numel()):
        lin = int(state_flat[flat_pos].item())
        if lin < 0 or lin >= total_cache_hash_size:
            continue

        t = int(cache_index_table_map[lin].item())
        ty = SparseType.from_int(int(weights_tys[t].item()))
        d_start = int(d_offsets[t].item())
        d_end = int(d_offsets[t + 1].item())
        d_emb = d_end - d_start
        row_bytes = rounded_row_size_in_bytes(d_emb, ty, row_alignment)

        row_in_table = lin - int(ch_cum[t].item())
        base = int(weights_offsets[t].item()) + row_in_table * row_bytes

        uvm_row = weights_uvm[base : base + row_bytes]
        cache_row = lxu_cache_weights[flat_pos, :row_bytes]
        if not torch.equal(uvm_row, cache_row):
            raise AssertionError(
                f"UVM vs cache mismatch at set*assoc+slot={flat_pos}, "
                f"linear_idx={lin}, table={t}, row_in_table={row_in_table}"
            )


def verify_lxu_state_matches_last_timestamp_map(
    buffers: Union[ManualLruCacheBuffers, nn.Module],
    last_ts: Dict[int, int],
) -> None:
    """
    For every occupied cache slot, linear index idx must satisfy
    lxu_state[slot] == last_ts[idx], where last_ts records the timestep of the
    most recent populate batch that contained idx (works with any timestep
    sequence; accuracy test uses strictly increasing stamps per round).
    """
    _, _, total_cache_hash_size, _, _, _, _, lxu_cache_state, _, lxu_state = (
        _lru_populate_tensors(buffers)
    )

    seen_in_state: Dict[int, Tuple[int, int]] = {}
    print("lxu_cache_state ", lxu_cache_state)
    for s in range(lxu_cache_state.size(0)):
        for w in range(lxu_cache_state.size(1)):
            idx = int(lxu_cache_state[s, w].item())
            if idx < 0 or idx >= total_cache_hash_size:
                continue
            if idx in seen_in_state:
                raise AssertionError(
                    f"linear_idx {idx} appears in multiple cache slots "
                    f"{seen_in_state[idx]} and ({s},{w})"
                )
            seen_in_state[idx] = (s, w)

    for idx, (s, w) in seen_in_state.items():
        if idx not in last_ts:
            raise AssertionError(
                f"linear_idx {idx} present in lxu_cache_state at ({s},{w}) "
                f"but missing from last_ts map"
            )
        got = int(lxu_state[s, w].item())
        exp = last_ts[idx]
        if got != exp:
            raise AssertionError(
                f"lxu_state[{s},{w}]={got} expected last_ts[{idx}]={exp}"
            )


def verify_lru_cache_after_populate(
    buffers: Union[ManualLruCacheBuffers, nn.Module],
    last_ts: Dict[int, int],
    row_alignment: int = 16,
) -> None:
    """UVM byte check plus lxu_state vs last_ts map (per-id latest timestep)."""
    verify_lxu_cache_weights_match_uvm(buffers, row_alignment=row_alignment)
    verify_lxu_state_matches_last_timestamp_map(buffers, last_ts)


def _sync_compute_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def run_lru_cache_populate_byte_accuracy(
    num_tables: int = 1,
    num_embeddings: int = 256,
    embedding_dims: int = 64,
    batch: int = 32,
    avg_pooling_factor: int = 2,
    cache_load_factor: float = 1.0,
) -> None:
    """
    Small deterministic run: after each populate, check cache rows vs UVM and
    that every cached linear index has lxu_state equal to the latest timestep
    recorded for that id (last_ts map). Timestep increases by one each round.
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    elif hasattr(torch, "npu") and torch.npu.is_available():
        device = torch.device(f"npu:{torch.npu.current_device()}")
    else:
        raise RuntimeError(
            "run_lru_cache_populate_byte_accuracy requires CUDA or NPU with fbgemm ops."
        )

    random.seed(0)
    torch.manual_seed(0)

    embedding_specs: List[
        Tuple[str, int, int, SparseType, EmbeddingLocation]
    ] = [
        (
            "",
            num_embeddings,
            embedding_dims,
            SparseType.INT8,
            EmbeddingLocation.MANAGED_CACHING,
        ),
    ]

    buffers = create_manual_lru_cache_buffers(
        embedding_specs, device, cache_load_factor
    )
    ch = buffers.cache_hash_size_cumsum
    total_h = buffers.total_cache_hash_size

    last_ts: Dict[int, int] = {}
    timestep = 1
    for _ in range(3):
        indices, offsets = create_request(
            num_tables, num_embeddings, batch, avg_pooling_factor
        )
        linear = linearize_cache_indices_python(ch, indices, offsets)
        valid = linear[linear != total_h]
        if valid.numel() > 0:
            for idx_t in torch.unique(valid):
                idx = int(idx_t.item())
                if 0 <= idx < total_h:
                    last_ts[idx] = timestep

        _call_lru_cache_populate_byte(buffers, linear, timestep)
        _sync_compute_device()
        verify_lru_cache_after_populate(buffers, last_ts)
        timestep += 1

    logging.info(
        "run_lru_cache_populate_byte_accuracy: all checks passed "
        f"(tables={num_tables}, E={num_embeddings}, batch={batch}, steps=3, "
        "monotonic timesteps + last_ts map)."
    )


@pytest.mark.parametrize(
    (
        "num_tables",
        "num_embeddings",
        "embedding_dims",
        "batch",
        "avg_pooling_factor",
        "cache_load_factor",
    ),
    [
        (1, 256, 64, 32, 2, 1.0),
    ],
)
def test_run_lru_cache_populate_byte_accuracy(
    num_tables: int,
    num_embeddings: int,
    embedding_dims: int,
    batch: int,
    avg_pooling_factor: int,
    cache_load_factor: float,
) -> None:
    if not torch.cuda.is_available() and not (
        hasattr(torch, "npu") and torch.npu.is_available()
    ):
        pytest.skip(
            "run_lru_cache_populate_byte_accuracy requires CUDA or NPU with fbgemm ops."
        )
    run_lru_cache_populate_byte_accuracy(
        num_tables=num_tables,
        num_embeddings=num_embeddings,
        embedding_dims=embedding_dims,
        batch=batch,
        avg_pooling_factor=avg_pooling_factor,
        cache_load_factor=cache_load_factor,
    )

if __name__ == "__main__":
    unittest.main()