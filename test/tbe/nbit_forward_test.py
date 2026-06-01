#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import random
import unittest
from typing import Any, Optional, Union

import hypothesis.strategies as st
import numpy as np
import torch

import fbgemm_ascend  # noqa F401
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    CacheAlgorithm,
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.utils import dequantize_embs, generate_requests, quantize_embs, round_up
from hypothesis import HealthCheck, given, settings, Verbosity

from .common import NBitFowardTestCommon, get_nbit_weights_ty, npu_unavailable


VERBOSITY: Verbosity = Verbosity.verbose
MAX_EXAMPLES_LONG_RUNNING = 15


class NBitForwardTest(NBitFowardTestCommon):
    def execute_nbit_forward_fused_pooled_emb_quant_(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_ty: SparseType,
        output_dtype: SparseType,
        weighted: bool,
        ref_module: Union[
            IntNBitTableBatchedEmbeddingBagsCodegen,
            SplitTableBatchedEmbeddingBagsCodegen,
        ],
    ) -> None:
        d_alignment = max(weights_ty.align_size() for _ in range(T))
        d_alignment = max(d_alignment, output_dtype.align_size())
        D = round_up(D, d_alignment)
        Ds = [D] * T

        if ref_module == SplitTableBatchedEmbeddingBagsCodegen:
            D = min(round_up(D, 4), 2048)
            Ds = [D] * T

        E = int(10**log_E)
        Es = [np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)]
        weights_ty_list = [weights_ty] * T
        managed = [EmbeddingLocation.DEVICE] * T

        op = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                ("", e, d, w_ty, EmbeddingLocation(m)) for (e, d, m, w_ty) in zip(Es, Ds, managed, weights_ty_list)
            ],
            output_dtype=output_dtype,
            device=torch.cuda.current_device(),
        )
        op.fill_random_weights()

        use_quant_ref = False
        if ref_module == SplitTableBatchedEmbeddingBagsCodegen:
            op_ref = SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[
                    (
                        e,
                        d,
                        EmbeddingLocation.DEVICE,
                        ComputeDevice.CUDA,
                    )
                    for (e, d) in zip(Es, Ds)
                ],
                weights_precision=SparseType.FP32,
                output_dtype=SparseType.FP32,
                device=torch.cuda.current_device(),
            )
        else:
            op_ref = IntNBitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[
                    ("", e, d, w_ty, EmbeddingLocation(m)) for (e, d, m, w_ty) in zip(Es, Ds, managed, weights_ty_list)
                ],
                output_dtype=SparseType.FP32,
                device=torch.cuda.current_device(),
            )
            op_ref.fill_random_weights()
            use_quant_ref = True

        split_weights = op.split_embedding_weights()
        ref_split_weights = op_ref.split_embedding_weights()
        for t in range(T):
            weights, scale_shift = split_weights[t]
            if use_quant_ref:
                ref_weights, ref_scale_shift = ref_split_weights[t]
                self.assertEqual(weights.size(), ref_weights.size())
            else:
                ref_weights = ref_split_weights[t]
                ref_scale_shift = None

            element_size = weights_ty_list[t].bit_rate() / 8.0
            rand_tensor = torch.rand(weights.shape[0], int(weights.shape[1] / element_size))
            rand_weights, rand_scale_shift = quantize_embs(rand_tensor, weights_ty_list[t])
            weights.copy_(rand_weights)
            if use_quant_ref:
                ref_weights.copy_(rand_weights)
            else:
                deq_rand_weights = dequantize_embs(rand_weights, rand_scale_shift, weights_ty_list[t], use_cpu=False)
                assert deq_rand_weights.dtype == torch.float32
                ref_weights.copy_(deq_rand_weights)

            if rand_scale_shift is not None:
                self.assertIsNotNone(scale_shift)
                scale_shift.copy_(rand_scale_shift)
                if use_quant_ref:
                    self.assertIsNotNone(ref_scale_shift)
                    ref_scale_shift.copy_(rand_scale_shift)

        requests = generate_requests(1, B, T, L, min(Es), reuse=0.1, weighted=weighted)
        for req in requests:
            if weighted:
                indices, offsets, per_sample_weights = req.unpack_3()
            else:
                indices, offsets = req.unpack_2()
                per_sample_weights = None

            op_indices = indices.int()
            op_offsets = offsets.int()
            ref_indices = indices.long() if ref_module == SplitTableBatchedEmbeddingBagsCodegen else op_indices
            ref_offsets = offsets.long() if ref_module == SplitTableBatchedEmbeddingBagsCodegen else op_offsets

            lowp_pooled_output = op(
                indices=op_indices,
                offsets=op_offsets,
                per_sample_weights=per_sample_weights,
            )
            fp32_pooled_output = op_ref(
                indices=ref_indices,
                offsets=ref_offsets,
                per_sample_weights=per_sample_weights,
            )

            lowp_pooled_emb_split = [d + 8 if output_dtype == SparseType.INT8 else d for d in Ds]
            lowp_pooled_output_per_table = torch.split(lowp_pooled_output, lowp_pooled_emb_split, dim=1)
            deq_lowp_pooled_output_per_table = [
                (
                    torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(t.contiguous())
                    if output_dtype == SparseType.INT8
                    else t.float()
                )
                for t in lowp_pooled_output_per_table
            ]
            fp32_pooled_output_per_table = torch.split(fp32_pooled_output, Ds, dim=1)
            dq_fp32_pooled_output_per_table = [
                (
                    torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
                        torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(t.contiguous()).contiguous()
                    ).contiguous()
                    if output_dtype == SparseType.INT8
                    else t.half().float()
                )
                for t in fp32_pooled_output_per_table
            ]
            cat_deq_lowp_pooled_output = torch.cat(deq_lowp_pooled_output_per_table, dim=1)
            cat_dq_fp32_pooled_output = torch.cat(dq_fp32_pooled_output_per_table, dim=1)
            torch.testing.assert_close(
                cat_deq_lowp_pooled_output,
                cat_dq_fp32_pooled_output,
                rtol=1e-2,
                atol=1e-2,
                equal_nan=True,
            )

    @unittest.skipIf(*npu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weights_ty=st.sampled_from([SparseType.FP32]),
        output_dtype=st.sampled_from([SparseType.FP32]),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    def test_nbit_forward_fused_pooled_emb_quant_against_ref(
        self,
        **kwargs: Any,
    ) -> None:
        # Community also covers weighted cases here. The NPU training
        # reference path currently rejects indice_weights, so this NPU
        # variant is intentionally narrowed to the unweighted path.
        self.execute_nbit_forward_fused_pooled_emb_quant_(
            weighted=False,
            ref_module=SplitTableBatchedEmbeddingBagsCodegen,
            **kwargs,
        )

    @unittest.skipIf(*npu_unavailable)
    def test_nbit_forward_fused_pooled_emb_quant_nan_weighted(self) -> None:
        self.skipTest(
            "Community test depends on torch.ops.fbgemm.initialize_nan_shared_mem, "
            "which is a GPU-only test helper and has no NPU equivalent."
        )

    @unittest.skipIf(*npu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weights_ty=st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.INT8,
                SparseType.INT4,
            ]
        ),
        output_dtype=st.sampled_from([SparseType.FP16, SparseType.BF16, SparseType.INT8]),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    def test_nbit_forward_fused_pooled_emb_quant(
        self,
        **kwargs: Any,
    ) -> None:
        self.execute_nbit_forward_fused_pooled_emb_quant_(
            weighted=False,
            ref_module=IntNBitTableBatchedEmbeddingBagsCodegen,
            **kwargs,
        )

    @given(
        indices_dtype=st.sampled_from([torch.int32, torch.int64]),
        weights_ty_and_D=st.sampled_from(
            [
                (SparseType.FP32, 1024),
                (SparseType.FP16, 2048),
                (SparseType.INT8, 4092),
                (SparseType.FP8, 4096),
                (SparseType.INT4, 4088),
                (SparseType.INT2, 4080),
            ]
        ),
    )
    @settings(deadline=None)
    @unittest.skipIf(*npu_unavailable)
    def test_nbit_forward_gpu_no_cache_max_sizes(
        self,
        indices_dtype: torch.dtype,
        weights_ty_and_D: tuple[SparseType, int],
    ) -> None:
        weights_ty, D = weights_ty_and_D
        self.execute_nbit_forward_(
            T=1,
            D=D,
            B=128,
            log_E=2,
            L=4,
            weighted=False,
            mixed=False,
            pooling_mode=PoolingMode.SUM,
            weights_ty=weights_ty,
            use_cache=False,
            cache_algorithm=CacheAlgorithm.LRU,
            use_cpu=False,
            use_array_for_index_remapping=True,
            do_pruning=False,
            mixed_weights_ty=False,
            indices_dtype=indices_dtype,
            output_dtype=SparseType.FP16,
        )

    @unittest.skipIf(*npu_unavailable)
    @given(
        nbit_weights_ty=get_nbit_weights_ty(),
        use_array_for_index_remapping=st.booleans(),
        do_pruning=st.booleans(),
        indices_dtype=st.sampled_from([torch.int32, torch.int64]),
        output_dtype=st.sampled_from([SparseType.FP32, SparseType.FP16]),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
    )
    def test_nbit_forward_gpu_no_cache(
        self,
        nbit_weights_ty: Optional[SparseType],
        use_array_for_index_remapping: bool,
        indices_dtype: torch.dtype,
        do_pruning: bool,
        output_dtype: SparseType,
    ) -> None:
        if nbit_weights_ty == SparseType.INT2 and output_dtype == SparseType.FP32:
            self.skipTest("The combination of INT2 and FP32 as weight and output types, respectively, is not supported")
        if indices_dtype != torch.int32 and not use_array_for_index_remapping:
            self.skipTest(
                "Hash-based index_remapping is an experimental feature and is currently not supported "
                "for indices.dtype == torch.int64 and indices.device != cpu"
            )

        pooling_mode = random.choice([PoolingMode.SUM, PoolingMode.MEAN, PoolingMode.NONE])
        mixed = False if pooling_mode == PoolingMode.NONE else random.choice([True, False])
        weighted = random.choice([True, False]) if pooling_mode == PoolingMode.SUM else False

        if nbit_weights_ty is None:
            weights_ty = SparseType.INT8
            mixed_weights_ty = True
        else:
            weights_ty = nbit_weights_ty
            mixed_weights_ty = False

        self.execute_nbit_forward_(
            T=random.randint(1, 50),
            D=random.randint(2, 2048),
            B=random.randint(0, 128),
            log_E=random.randint(2, 4),
            L=random.randint(0, 32),
            weighted=weighted,
            mixed=mixed,
            pooling_mode=pooling_mode,
            weights_ty=weights_ty,
            use_cache=False,
            cache_algorithm=CacheAlgorithm.LRU,
            use_cpu=False,
            use_array_for_index_remapping=use_array_for_index_remapping,
            do_pruning=do_pruning,
            mixed_weights_ty=mixed_weights_ty,
            indices_dtype=indices_dtype,
            output_dtype=output_dtype,
        )

    @unittest.skipIf(*npu_unavailable)
    @given(
        nbit_weights_ty=st.sampled_from([SparseType.INT8]),
        pooling_mode=st.sampled_from([PoolingMode.NONE]),
        output_dtype=st.sampled_from([SparseType.FP16, SparseType.BF16]),
        D=st.sampled_from([32, 256, 384, 512, 1024]),
        B=st.integers(min_value=8, max_value=32),
        T=st.integers(min_value=10, max_value=20),
        L=st.integers(min_value=10, max_value=100),
        MAXH=st.integers(min_value=50, max_value=100),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
    )
    def test_nbit_forward_cpu_gpu_dequantize_parity(
        self,
        nbit_weights_ty: SparseType,
        pooling_mode: PoolingMode,
        output_dtype: SparseType,
        D: int,
        B: int,
        T: int,
        L: int,
        MAXH: int,
    ) -> None:
        D_alignment = 1 if nbit_weights_ty.bit_rate() % 8 == 0 else int(8 / nbit_weights_ty.bit_rate())
        D = round_up(D, D_alignment)
        table_rows = [np.random.randint(low=1, high=MAXH + 1) for _ in range(T)]

        quant_cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[("", h, D, nbit_weights_ty, EmbeddingLocation.HOST) for h in table_rows],
            pooling_mode=pooling_mode,
            device="cpu",
            output_dtype=nbit_weights_ty,
        )
        quant_cc.fill_random_weights()

        dequant_cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[("", h, D, nbit_weights_ty, EmbeddingLocation.HOST) for h in table_rows],
            pooling_mode=pooling_mode,
            device="cpu",
            output_dtype=output_dtype,
        )
        dequant_cc.fill_random_weights()

        split_weights = quant_cc.split_embedding_weights()
        ref_split_weights = dequant_cc.split_embedding_weights()
        for t in range(T):
            weights, scale_shift = split_weights[t]
            ref_weights, ref_scale_shift = ref_split_weights[t]
            self.assertEqual(weights.size(), ref_weights.size())
            element_size = SparseType.INT8.bit_rate() / 8.0
            rand_tensor = torch.rand(ref_weights.shape[0], int(ref_weights.shape[1] / element_size))
            rand_weights, rand_scale_shift = quantize_embs(rand_tensor, SparseType.INT8)
            ref_weights.copy_(rand_weights)
            weights.copy_(ref_weights)
            if rand_scale_shift is not None:
                self.assertIsNotNone(scale_shift)
                self.assertIsNotNone(ref_scale_shift)
                ref_scale_shift.copy_(rand_scale_shift)
                scale_shift.copy_(ref_scale_shift)

        lengths_list = [torch.randint(1, L + 1, (B,)) for _ in range(T)]
        indices_list = [torch.randint(0, h, (int(length.sum().item()),)) for length, h in zip(lengths_list, table_rows)]
        indices = torch.cat(indices_list, 0)
        lengths = torch.cat(lengths_list, 0)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        quant_cc_output = quant_cc(indices.int(), offsets.int())
        dequant_cc_output = dequant_cc(indices.int(), offsets.int())
        dequant_output_from_quant_cc = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloatOrHalf(
            quant_cc_output.cpu(),
            output_dtype.as_int(),
            quant_padding_float_type=False,
            scale_bias_last=False,
        )
        torch.testing.assert_close(
            dequant_cc_output.cpu(),
            dequant_output_from_quant_cc.cpu(),
            equal_nan=False,
        )


if __name__ == "__main__":
    unittest.main()
