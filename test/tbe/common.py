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
import logging
import os
import random
import unittest
from typing import Callable, Optional, TypeVar

import numpy as np
import torch
import hypothesis.strategies as st
from fbgemm_gpu.split_embedding_configs import FP8QuantizationConfig, SparseType
from fbgemm_gpu import split_table_batched_embeddings_ops_common
from fbgemm_gpu import split_table_batched_embeddings_ops_training
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    CacheAlgorithm,
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.utils import dequantize_embs, fake_quantize_embs, round_up
from hypothesis import assume
from hypothesis.strategies import composite

MAX_EXAMPLES = 40

logging.getLogger().setLevel(logging.INFO)

Deviceable = TypeVar("Deviceable", torch.nn.EmbeddingBag, torch.nn.Embedding, torch.Tensor)

TEST_WITH_ROCM: bool = os.getenv("FBGEMM_TEST_WITH_ROCM", "0") == "1"
torch.cuda.is_available = torch.npu.is_available
torch.cuda.device_count = torch.npu.device_count
torch.cuda.current_device = torch.npu.current_device
torch.cuda.get_device_properties = torch.npu.get_device_properties
torch.Tensor.cuda = torch.Tensor.npu


class ComputeDevice(enum.IntEnum):
    CPU = 0
    CUDA = 1
    MTIA = 2
    NPU = 3


split_table_batched_embeddings_ops_training.ComputeDevice = ComputeDevice


def _print_close_mismatches(
    golden: torch.Tensor,
    test: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    max_print: int = 20,
) -> None:
    gold = golden.detach().cpu()
    test_cpu = test.detach().cpu()
    close_mask = torch.isclose(test_cpu, gold, rtol=rtol, atol=atol, equal_nan=True)
    mismatch = torch.nonzero(~close_mask, as_tuple=False)

    logging.info(
        "Found %d mismatches / %d elements (rtol=%s, atol=%s)",
        mismatch.shape[0],
        gold.numel(),
        rtol,
        atol,
    )
    for idx_tensor in mismatch[:max_print]:
        idx = tuple(idx_tensor.tolist())
        gv = gold[idx]
        tv = test_cpu[idx]
        logging.info(
            "mismatch %s: golden=%s test=%s "
            "gold.isnan=%s test.isnan=%s "
            "gold.isposinf=%s test.isposinf=%s "
            "gold.isneginf=%s test.isneginf=%s "
            "gold.isfinite=%s test.isfinite=%s",
            idx,
            gv.item(),
            tv.item(),
            torch.isnan(gv).item(),
            torch.isnan(tv).item(),
            torch.isposinf(gv).item(),
            torch.isposinf(tv).item(),
            torch.isneginf(gv).item(),
            torch.isneginf(tv).item(),
            torch.isfinite(gv).item(),
            torch.isfinite(tv).item(),
        )


def _print_weight_prepare_debug(
    *,
    table_idx: int,
    weight_ty: SparseType,
    prepared: torch.Tensor,
    ref_weight: torch.Tensor,
    max_print: int = 20,
) -> None:
    prepared_cpu = prepared.detach().cpu().float()
    ref_cpu = ref_weight.detach().cpu().float()
    rows = min(prepared_cpu.shape[0], ref_cpu.shape[0])
    cols = min(prepared_cpu.shape[1], ref_cpu.shape[1])
    prepared_cpu = prepared_cpu[:rows, :cols].contiguous()
    ref_cpu = ref_cpu[:rows, :cols].contiguous()
    close_mask = torch.isclose(prepared_cpu, ref_cpu, rtol=1.0e-2, atol=1.0e-2, equal_nan=True)
    mismatch = torch.nonzero(~close_mask, as_tuple=False)
    if mismatch.numel() == 0:
        return

    logging.info(
        "weight prepare mismatch: table=%d weight_ty=%s shape=%s mismatches=%d/%d",
        table_idx,
        weight_ty,
        tuple(prepared_cpu.shape),
        mismatch.shape[0],
        prepared_cpu.numel(),
    )
    logging.info("prepared[:2,:8]=%s", prepared_cpu[:2, :8])
    logging.info("ref_weight[:2,:8]=%s", ref_cpu[:2, :8])
    for idx_tensor in mismatch[:max_print]:
        idx = tuple(idx_tensor.tolist())
        pv = prepared_cpu[idx]
        rv = ref_cpu[idx]
        logging.info(
            "weight mismatch %s: prepared=%s ref=%s prepared.isnan=%s ref.isnan=%s",
            idx,
            pv.item(),
            rv.item(),
            torch.isnan(pv).item(),
            torch.isnan(rv).item(),
        )


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
        else st.just(False)
        if (npu_available() and TEST_WITH_ROCM)
        else st.just(True)
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


def gen_mixed_B_batch_sizes(B: int, T: int, num_ranks: Optional[int] = None) -> tuple[list[list[int]], list[int]]:
    if num_ranks is None:
        num_ranks = np.random.randint(low=1, high=4)
    low = max(int(0.25 * B), 1)
    high = int(B)
    if low == high:
        Bs_rank_feature = [[B] * num_ranks for _ in range(T)]
    else:
        Bs_rank_feature = [np.random.randint(low=low, high=high, size=num_ranks).tolist() for _ in range(T)]
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
        torch.tensor(np.cumsum(np.asarray([0] + [L for _ in range(B)])[:-1]).astype(np.int64)),
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
    assert device.type in ("cpu", "npu")
    if cache_load_factor < 0 or cache_load_factor > 1:
        raise ValueError(f"cache_load_factor must be between 0.0 and 1.0, got {cache_load_factor}")

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


@composite
def get_nbit_weights_ty(draw) -> Optional[SparseType]:
    mixed_weights_ty = draw(st.booleans())
    if mixed_weights_ty:
        return None
    return draw(
        st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.FP8,
                SparseType.INT8,
                SparseType.INT4,
                SparseType.INT2,
            ]
        )
    )


class NBitFowardTestCommon(unittest.TestCase):
    def execute_nbit_forward_(  # noqa: C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        pooling_mode: PoolingMode,
        weights_ty: SparseType,
        use_cache: bool,
        cache_algorithm: CacheAlgorithm,
        use_cpu: bool,
        use_array_for_index_remapping: bool,
        do_pruning: bool,
        mixed_weights_ty: bool,
        indices_dtype: torch.dtype,
        output_dtype: SparseType,
    ) -> None:
        assume(pooling_mode == PoolingMode.SUM or not weighted)
        assume(not mixed or pooling_mode != PoolingMode.NONE)

        mode = "sum"
        do_pooling = True
        if pooling_mode == PoolingMode.MEAN:
            mode = "mean"
        elif pooling_mode == PoolingMode.NONE:
            do_pooling = False

        E = int(10**log_E)

        if not mixed_weights_ty:
            weights_ty_list = [weights_ty] * T
        else:
            weights_ty_list = [
                np.random.choice(
                    [
                        SparseType.FP32,
                        SparseType.FP16,
                        SparseType.FP8,
                        SparseType.INT8,
                        SparseType.INT4,
                    ]
                    + ([SparseType.INT2] if output_dtype != SparseType.FP32 else [])
                )
                for _ in range(T)
            ]
        # weights_ty_list = [SparseType.FP32,SparseType.FP32,SparseType.FP32]

        d_alignment = max(1 if ty.bit_rate() % 8 == 0 else int(8 / ty.bit_rate()) for ty in weights_ty_list)
        D = round_up(D, d_alignment)

        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                round_up(
                    np.random.randint(low=int(max(0.25 * D, 1)), high=int(1.0 * D)),
                    d_alignment,
                )
                for _ in range(T)
            ]
            Ds = [min(d, 128) for d in Ds]
            Es = [np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)]

        if do_pooling:
            bs = [to_device(torch.nn.EmbeddingBag(e, d, mode=mode, sparse=True), use_cpu) for (e, d) in zip(Es, Ds)]
            bs_cpu = [torch.nn.EmbeddingBag(e, d, mode=mode, sparse=True).cpu() for (e, d) in zip(Es, Ds)]
        else:
            bs = [to_device(torch.nn.Embedding(e, d, sparse=True), use_cpu) for (e, d) in zip(Es, Ds)]
            bs_cpu = [torch.nn.Embedding(e, d, sparse=True).cpu() for (e, d) in zip(Es, Ds)]

        if use_cpu:
            managed = [EmbeddingLocation.HOST] * T
        elif use_cache:
            managed = [EmbeddingLocation.MANAGED_CACHING] * T
            if mixed:
                average_D = sum(Ds) // T
                for t, d in enumerate(Ds):
                    managed[t] = EmbeddingLocation.DEVICE if d < average_D else managed[t]
        else:
            # The original community GPU test mixes DEVICE and MANAGED here.
            # NPU does not support CUDA UVM / EmbeddingLocation.MANAGED, so the
            # standalone NPU variant keeps the no-cache path on DEVICE only.
            managed = [EmbeddingLocation.DEVICE] * T

        if SparseType.FP8 in weights_ty_list:
            fp8_config = FP8QuantizationConfig(random.choice([4, 5]), 7)
            has_fp8_weight = True
        else:
            fp8_config = None
            has_fp8_weight = False

        xs = [to_device(torch.randint(low=0, high=e, size=(B, L)), use_cpu) for e in Es]
        xws = [to_device(torch.randn(size=(B, L)), use_cpu) for _ in range(T)]

        if do_pruning:
            x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
            xw = torch.cat([xw.view(1, B, L) for xw in xws], dim=0)
            indices, offsets = get_table_batched_offsets_from_dense(x, use_cpu=use_cpu)

            dense_indices = torch.randint(low=0, high=E, size=(T, B, L)).view(-1).int()
            original_E = E
            current_device = "cpu" if use_cpu else torch.cuda.current_device()

            indices = indices.view(-1).int()
            offsets = offsets.view(-1).int()

            index_remappings_array = []
            for t in range(T):
                indice_t = indices.view(T, B, L)[t].view(-1).to(dtype=indices_dtype, device=current_device)
                dense_indice_t = dense_indices.view(T, B, L)[t].view(-1).to(dtype=indices_dtype, device=current_device)
                index_remappings_array_t = torch.tensor(
                    [-1] * original_E,
                    dtype=indices_dtype,
                    device=current_device,
                )
                index_remappings_array_t[indice_t] = dense_indice_t
                index_remappings_array.append(index_remappings_array_t.cpu())
        else:
            index_remappings_array = [torch.arange(e, dtype=indices_dtype) for e in Es]
            x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
            xw = torch.cat([xw.view(1, B, L) for xw in xws], dim=0)
            indices, offsets = get_table_batched_offsets_from_dense(x, use_cpu=use_cpu)

        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                ("", e, d, w_ty, EmbeddingLocation(m)) for (e, d, m, w_ty) in zip(Es, Ds, managed, weights_ty_list)
            ],
            pooling_mode=pooling_mode,
            index_remapping=index_remappings_array if B != 0 else None,
            device="cpu" if use_cpu else torch.cuda.current_device(),
            cache_algorithm=cache_algorithm,
            use_array_for_index_remapping=use_array_for_index_remapping,
            output_dtype=output_dtype,
            fp8_exponent_bits=(fp8_config.get("exponent_bits") if has_fp8_weight else None),
            fp8_exponent_bias=(fp8_config.get("exponent_bias") if has_fp8_weight else None),
            indices_dtype=indices_dtype,
        )
        cc.fill_random_weights()

        # Keep this disabled while debugging pooled-path mismatches so that
        # the module path matches the direct eager execution path.
        if not use_cpu:
            cc = torch.jit.script(cc)

        for t in range(T):
            weights, scale_shift = cc.split_embedding_weights()[t]
            if scale_shift is not None:
                (e_count, width) = scale_shift.shape
                self.assertEqual(width, 4)
                if weights_ty_list[t] == SparseType.INT2:
                    scales = np.random.uniform(0.1, 1, size=(e_count,)).astype(np.float16)
                    shifts = np.random.uniform(-2, 2, size=(e_count,)).astype(np.float16)
                elif weights_ty_list[t] == SparseType.INT4:
                    scales = np.random.uniform(0.01, 0.1, size=(e_count,)).astype(np.float16)
                    shifts = np.random.uniform(-2, 2, size=(e_count,)).astype(np.float16)
                else:
                    scales = np.random.uniform(0.001, 0.01, size=(e_count,)).astype(np.float16)
                    shifts = np.random.uniform(-2, 2, size=(e_count,)).astype(np.float16)
                scale_shift[:, :] = torch.tensor(np.stack([scales, shifts], axis=1).astype(np.float16).view(np.uint8))

            fake_quantize_embs(
                weights,
                scale_shift,
                bs[t].weight.detach(),
                weights_ty_list[t],
                use_cpu=use_cpu,
                fp8_config=fp8_config if has_fp8_weight else None,
            )

            if weights_ty_list[t] == SparseType.FP32:
                prepared = weights.view(torch.float32)
            elif weights_ty_list[t] == SparseType.FP16:
                prepared = weights.view(torch.float16).float()
            elif weights_ty_list[t] == SparseType.FP8:
                prepared = dequantize_embs(
                    weights,
                    None,
                    weights_ty_list[t],
                    use_cpu=use_cpu,
                    fp8_config=fp8_config,
                ).float()
            else:
                prepared = dequantize_embs(
                    weights,
                    scale_shift,
                    weights_ty_list[t],
                    use_cpu=use_cpu,
                ).float()
            _print_weight_prepare_debug(
                table_idx=t,
                weight_ty=weights_ty_list[t],
                prepared=prepared,
                ref_weight=bs[t].weight.detach(),
            )
            bs_cpu[t].weight.detach().copy_(bs[t].weight.detach().cpu())

        indices = indices.to(dtype=indices_dtype)
        offsets = offsets.to(dtype=indices_dtype)

        if not use_cpu:
            fc2 = cc(indices, offsets) if not weighted else cc(indices, offsets, xw.contiguous().view(-1))
        else:
            cc = cc.cpu()
            indices, offsets = indices.cpu(), offsets.cpu()
            fc2 = cc(indices, offsets) if not weighted else cc(indices, offsets, xw.contiguous().view(-1).cpu())

        if do_pooling and B == 0:
            self.assertEqual(fc2.size(), (0, cc.total_D))
            return

        new_indices = []
        for t in range(T):
            new_indices_t = torch.zeros([B, L], dtype=torch.int32)
            for i in range(B):
                for j in range(L):
                    old_index = xs[t][i, j]
                    new_indices_t[i][j] = index_remappings_array[t][old_index]
            new_indices.append(new_indices_t)

        fs = (
            [b_indices(b, x, use_cpu=use_cpu, do_pooling=do_pooling) for (b, x) in zip(bs, new_indices)]
            if not weighted
            else [
                b_indices(
                    b,
                    x,
                    per_sample_weights=xw.view(-1),
                    use_cpu=use_cpu,
                    do_pooling=do_pooling,
                )
                for (b, x, xw) in zip(bs, new_indices, xws)
            ]
        )
        if do_pooling:
            f = torch.cat([f.view(B, -1) for f in fs], dim=1)
        else:
            f = torch.cat(fs, dim=0).view(-1, D)

        fs_cpu = (
            [b_indices(b, x.cpu(), use_cpu=True, do_pooling=do_pooling) for (b, x) in zip(bs_cpu, new_indices)]
            if not weighted
            else [
                b_indices(
                    b,
                    x.cpu(),
                    per_sample_weights=xw.view(-1).cpu(),
                    use_cpu=True,
                    do_pooling=do_pooling,
                )
                for (b, x, xw) in zip(bs_cpu, new_indices, xws)
            ]
        )
        if do_pooling:
            f_cpu_ref = torch.cat([f_.view(B, -1) for f_ in fs_cpu], dim=1)
        else:
            f_cpu_ref = torch.cat(fs_cpu, dim=0).view(-1, D)

        if fc2.dtype == torch.quint4x2:
            fc2_float = torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfFrontToFloatOrHalf(
                fc2.cpu(), bit_rate=4, output_dtype=0
            )
        else:
            fc2_float = fc2.float()

        npu_ref_cpu = f.float().cpu()
        cpu_ref_cpu = f_cpu_ref.float().cpu()
        logging.info(
            "ref parity: max_abs(npu_ref-cpu_ref)=%s mean_abs(npu_ref-cpu_ref)=%s",
            (npu_ref_cpu - cpu_ref_cpu).abs().max().item(),
            (npu_ref_cpu - cpu_ref_cpu).abs().mean().item(),
        )
        if npu_ref_cpu.dim() >= 2:
            logging.info("npu_ref[:4,:8]=%s", npu_ref_cpu[:4, :8])
            logging.info("cpu_ref[:4,:8]=%s", cpu_ref_cpu[:4, :8])
        golden_cpu = cpu_ref_cpu
        test_cpu = fc2_float.cpu()
        try:
            torch.testing.assert_close(
                test_cpu,
                golden_cpu,
                atol=1.0e-2,
                rtol=1.0e-2,
            )
        except AssertionError:
            logging.info(
                "execute_nbit_forward_ mismatch context: "
                "T=%s D=%s B=%s L=%s pooling_mode=%s weighted=%s "
                "use_cache=%s use_array_for_index_remapping=%s do_pruning=%s "
                "mixed=%s mixed_weights_ty=%s indices_dtype=%s output_dtype=%s "
                "weights_ty_list=%s",
                T,
                D,
                B,
                L,
                pooling_mode,
                weighted,
                use_cache,
                use_array_for_index_remapping,
                do_pruning,
                mixed,
                mixed_weights_ty,
                indices_dtype,
                output_dtype,
                weights_ty_list,
            )
            logging.info("fc2_float shape=%s, ref shape=%s", tuple(test_cpu.shape), tuple(golden_cpu.shape))
            logging.info("indices[:32]=%s", indices[:32].detach().cpu())
            logging.info("offsets[:32]=%s", offsets[:32].detach().cpu())
            if test_cpu.dim() >= 2:
                logging.info("test[:2,:8]=%s", test_cpu[:2, :8])
                logging.info("golden[:2,:8]=%s", golden_cpu[:2, :8])
            else:
                logging.info("test[:16]=%s", test_cpu[:16])
                logging.info("golden[:16]=%s", golden_cpu[:16])
            _print_close_mismatches(
                golden_cpu,
                test_cpu,
                rtol=1.0e-2,
                atol=1.0e-2,
            )
            raise
