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
import random
import unittest
from typing import Callable

import hypothesis.strategies as st
import numpy as np
import torch
from common import (
    MAX_EXAMPLES,
    npu_unavailable,
    use_cpu_strategy
)
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import EmbeddingLocation
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
    rounded_row_size_in_bytes,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    INT8_EMB_ROW_DIM_OFFSET,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.utils import round_up
from hypothesis import given, settings, Verbosity

VERBOSITY: Verbosity = Verbosity.verbose

# pyre-ignore
additional_decorators: dict[str, list[Callable]] = {}


class SplitTableBatchedEmbeddingsTest(unittest.TestCase):
    @unittest.skipIf(*npu_unavailable)
    def test_split_embedding_codegen_forward(  # noqa C901
            self,
    ) -> None:
        # Dummy test in order to run generated opcheck tests on
        # split_embedding_codegen_forward_weighted_cuda and
        # split_embedding_codegen_forward_unweighted_cuda.
        # Sizes and values of int tensors were generated from running
        # one test instance of test_backward_adagrad_fp16_pmSUM and outputting
        # sizes/dtypes/values.
        def _do_test(weighted: bool) -> None:
            flatten_dev_weights = torch.rand(1, dtype=torch.float).npu()
            uvm_weights = torch.rand(10456, dtype=torch.float).npu()
            lxu_cache_weights = torch.rand(544, 4, dtype=torch.float).npu()
            weights_placements = torch.tensor([2, 2, 2]).to(
                dtype=torch.int, device="npu"
            )
            weights_offsets = torch.tensor([0, 2784, 2784]).to(
                dtype=torch.long, device="npu"
            )
            D_offsets = torch.tensor([0, 4, 8]).to(dtype=torch.int, device="npu")
            total_D = 12
            max_D = 4
            indices = torch.tensor(
                [
                    680,
                    213,
                    293,
                    439,
                    1004,
                    885,
                    986,
                    1162,
                    433,
                    1327,
                    187,
                    89,
                ]
            ).to(dtype=torch.long, device="npu")
            offsets = torch.tensor([0, 2, 4, 6, 8, 10, 12]).to(
                dtype=torch.long, device="npu"
            )
            pooling_mode = False
            indice_weights = torch.rand(12, dtype=torch.float).npu()
            lxu_cache_locations = torch.tensor(
                [
                    224,
                    352,
                    353,
                    192,
                    384,
                    64,
                    288,
                    1,
                    96,
                    194,
                    0,
                    193,
                ]
            ).to(dtype=torch.int, device="npu")
            uvm_cache_stats = torch.tensor(
                [
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                ]
            ).to(dtype=torch.int, device="npu")

            output_dtype = 0  # SparseType.FP32
            is_experimental = False

            op_args = [
                flatten_dev_weights,
                uvm_weights,
                lxu_cache_weights,
                weights_placements,
                weights_offsets,
                D_offsets,
                total_D,
                max_D,
                indices,
                offsets,
                pooling_mode,
            ]
            if weighted:
                op_args += [indice_weights]
            op_args += [
                lxu_cache_locations,
                uvm_cache_stats,
                output_dtype,
                is_experimental,
            ]

            op_name = "split_embedding_codegen_forward"
            op_name += "_weighted" if weighted else "_unweighted"
            op_name += "_cuda"

            getattr(torch.ops.fbgemm, op_name)(*op_args)

        _do_test(False)  # not supported weighted ops

    @unittest.skipIf(*npu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=64),
        log_E=st.integers(min_value=2, max_value=3),
        N=st.integers(min_value=0, max_value=50),
        weights_ty=st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.INT8,
                SparseType.INT4,
                SparseType.INT2,
            ]
        ),
        output_dtype=st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.INT8,
            ]
        ),
        use_cpu=use_cpu_strategy(),
        test_internal=st.booleans(),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_embedding_inplace_update(
            self,
            T: int,  # num of embedding tables
            D: int,  # embedding dim
            log_E: int,  # embedding table row number
            N: int,  # num of update rows per table
            weights_ty: SparseType,
            output_dtype: SparseType,
            use_cpu: bool,
            test_internal: bool,  # test with OSS op or internal customized op
    ) -> None:
        if not use_cpu and test_internal:
            return

        D_alignment = max(weights_ty.align_size(), output_dtype.align_size())
        D = round_up(D, D_alignment)
        Ds = [
            round_up(
                np.random.randint(low=int(max(0.25 * D, 1)), high=int(1.0 * D)),
                D_alignment,
            )
            for _ in range(T)
        ]
        E = int(10 ** log_E)
        Es = [np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)]
        row_alignment = 1 if use_cpu else 16
        current_device = "cpu" if use_cpu else torch.npu.current_device()
        location = EmbeddingLocation.HOST if use_cpu else EmbeddingLocation.DEVICE

        weights_ty_list = [weights_ty] * T

        # create two embedding bag op with random weights
        locations = [location] * T
        op = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                ("", E, D, W_TY, L)
                for (E, D, W_TY, L) in zip(Es, Ds, weights_ty_list, locations)
            ],
            output_dtype=output_dtype,
            device=current_device,
        )
        op.fill_random_weights()
        op_ref = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                ("", E, D, W_TY, L)
                for (E, D, W_TY, L) in zip(Es, Ds, weights_ty_list, locations)
            ],
            output_dtype=output_dtype,
            device=current_device,
        )
        op_ref.fill_random_weights()

        # randomly generate update table and row indices
        update_table_indices = []
        update_table_indices2 = []
        update_row_indices = []
        update_row_indices2 = []
        for t in range(T):
            n = np.random.randint(low=0, high=N) if N > 0 else 0
            if n == 0:
                continue
            update_table_indices.append(t)
            update_row_id_list = random.sample(range(Es[t]), n)
            update_row_indices.append(update_row_id_list)
            update_table_indices2.extend([t] * n)
            update_row_indices2.extend(update_row_id_list)

        # generate update tensor based on weights from "op_ref" embedding table
        update_weights_list = []
        ref_split_weights = op_ref.split_embedding_weights(split_scale_shifts=False)

        update_weight_size = sum(
            [
                rounded_row_size_in_bytes(
                    Ds[t],
                    weights_ty_list[t],
                    row_alignment,
                )
                for t in update_table_indices2
            ]
        )
        update_weights_tensor2 = torch.randint(
            low=0,
            high=255,
            size=(update_weight_size,),
            dtype=torch.uint8,
            device=current_device,
        )

        update_offsets = 0
        for i in range(len(update_table_indices)):
            table_idx = update_table_indices[i]
            (ref_weights, _) = ref_split_weights[table_idx]

            D_bytes = rounded_row_size_in_bytes(
                Ds[table_idx], weights_ty_list[table_idx], row_alignment
            )

            update_weights = []
            for row_idx in update_row_indices[i]:
                update_weights.append(ref_weights[row_idx].tolist())
                # fmt: off
                update_weights_tensor2[update_offsets: update_offsets + D_bytes] = (
                    ref_weights[row_idx]
                )
                # fmt: on
                update_offsets += D_bytes

            update_weights_tensor = torch.tensor(
                update_weights,
                device=current_device,
                dtype=torch.uint8,
            )
            update_weights_list.append(update_weights_tensor)

        # run inplace update on "op" embedding table
        if not test_internal:
            # Test scatter_ based OSS solution
            op.embedding_inplace_update(
                update_table_indices,
                update_row_indices,
                update_weights_list,
            )
        else:
            # Test customized op
            op.embedding_inplace_update_internal(
                update_table_indices2,
                update_row_indices2,
                update_weights_tensor2,
            )

        # verify weights are equal with "op_ref" for the updated rows in "op"
        split_weights = op.split_embedding_weights(split_scale_shifts=False)
        for i in range(len(update_table_indices)):
            t = update_table_indices[i]
            for r in update_row_indices[i]:
                (weights, _) = split_weights[t]
                (ref_weights, _) = ref_split_weights[t]
                self.assertEqual(weights.size(), ref_weights.size())
                torch.testing.assert_close(
                    weights[r],
                    ref_weights[r],
                    rtol=1e-2,
                    atol=1e-2,
                    equal_nan=True,
                )

    @unittest.skipIf(*npu_unavailable)
    def test_update_hyper_parameters(self) -> None:
        # Create an abstract split table
        D = 8
        T = 2
        E = 10 ** 2
        Ds = [D] * T
        Es = [E] * T

        hyperparameters = {
            "eps": 0.1,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
        }
        cc = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    E,
                    D,
                    EmbeddingLocation.DEVICE,
                    ComputeDevice.NPU,
                )
                for (E, D) in zip(Es, Ds)
            ],
            learning_rate=0.1,
            **hyperparameters,  # pyre-ignore[6]
        )

        # Update hyperparameters
        updated_parameters = {
            key: value + 1.0 for key, value in hyperparameters.items()
        } | {"lr": 1.0, "lower_bound": 2.0}
        cc.update_hyper_parameters(updated_parameters)
        self.assertAlmostEqual(cc.get_learning_rate(), updated_parameters["lr"])
        self.assertAlmostEqual(cc.optimizer_args.eps, updated_parameters["eps"])
        self.assertAlmostEqual(cc.optimizer_args.beta1, updated_parameters["beta1"])
        self.assertAlmostEqual(cc.optimizer_args.beta2, updated_parameters["beta2"])
        self.assertAlmostEqual(
            cc.optimizer_args.weight_decay, updated_parameters["weight_decay"]
        )
        self.assertAlmostEqual(cc.gwd_lower_bound, updated_parameters["lower_bound"])

        # Update hyperparameters with invalid parameter name
        invalid_parameter = "invalid_parameter"
        with self.assertRaisesRegex(
                NotImplementedError,
                f"Setting hyper-parameter {invalid_parameter} is not supported",
        ):
            cc.update_hyper_parameters({"invalid_parameter": 1.0})


if __name__ == "__main__":
    unittest.main()
