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

import unittest
from typing import List

import fbgemm_gpu  # noqa E402
import fbgemm_ascend  # noqa E402

import hypothesis.strategies as st
import numpy as np
import torch

from hypothesis import assume, given, settings, Verbosity

from ..common import MAX_EXAMPLES, use_cpu_strategy

from itertools import accumulate

from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    PoolingMode,
)

from fbgemm_gpu.split_table_batched_embeddings_ops_training_common import (
    generate_vbe_metadata,
)

VERBOSITY: Verbosity = Verbosity.verbose


class SplitEmbeddingsUtilsTest(unittest.TestCase):
    @given(
        T=st.integers(min_value=1, max_value=64),
        B=st.integers(min_value=1, max_value=64),
        max_L=st.integers(min_value=1, max_value=64),
        bounds_check_mode=st.sampled_from(
            [
                BoundsCheckMode.FATAL,
                BoundsCheckMode.WARNING,
                BoundsCheckMode.IGNORE,
            ]
        ),
        weighted=st.booleans(),
        dtype=st.sampled_from(
            [
                torch.int64,
                torch.int32,
            ]
        ),
        mixed_B=st.booleans(),
        bounds_check_version=st.sampled_from((1, 2)),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_bounds_check(  # noqa C901
        self,
        T: int,
        B: int,
        max_L: int,
        bounds_check_mode: BoundsCheckMode,
        weighted: bool,
        dtype: torch.dtype,
        mixed_B: bool,
        bounds_check_version: int,
    ) -> None:
        if not torch.npu.is_available():
            self.skipTest("npu not available")
        rows_per_table = torch.tensor(
            np.random.randint(low=1, high=1000, size=(T,))
        ).long()
        if not mixed_B:
            Bs = [B] * T
        else:
            low = max(int(0.25 * B), 1)
            high = int(B)
            if low == high:
                Bs = [B] * T
            else:
                Bs = [np.random.randint(low=low, high=high) for _ in range(T)]
        B_offsets = [0] + list(accumulate(Bs))
        Ls = np.random.randint(low=0, high=max_L, size=(B_offsets[-1],))
        indices = [
            np.random.randint(
                low=0,
                high=rows_per_table[t],
                size=sum(Ls[B_offsets[t] : B_offsets[t + 1]]),
            )
            for t in range(T)
        ]
        indices = torch.tensor(np.concatenate(indices, axis=0)).to(dtype)
        weights = (
            torch.rand(indices.shape, dtype=torch.float, device=indices.device)
            if weighted
            else None
        )
        offsets = torch.tensor([0] + np.cumsum(Ls.flatten()).tolist()).to(dtype)
        warning = torch.tensor([0]).long()

        if mixed_B:
            B_offsets = torch.tensor(
                B_offsets, device="cpu", dtype=torch.int32
            )
            max_B = max(Bs)
            vbe_metadata = generate_vbe_metadata(
                offsets,
                [[b] for b in Bs],
                pooling_mode=PoolingMode.SUM,
                feature_dims_cpu=torch.tensor(
                    [-1] * T, device="cpu", dtype=torch.int64
                ),  # unused
                device=torch.device("cpu"),
            )
            B_offsets = vbe_metadata.B_offsets
            assert isinstance(B_offsets, torch.Tensor), "B_offsets must be tensor"
            info_B_num_bits, info_B_mask = torch.ops.fbgemm.get_infos_metadata(
                vbe_metadata.B_offsets,  # unused tensor
                vbe_metadata.max_B,
                B_offsets.numel() - 1,  # T
            )
            row_output_offsets, b_t_map = torch.ops.fbgemm.generate_vbe_metadata(
                B_offsets,
                vbe_metadata.B_offsets_rank_per_feature,
                vbe_metadata.output_offsets_feature_rank,
                torch.tensor(
                    [-1] * (T + 1),
                    device=torch.device("cpu"),
                    dtype=torch.int,
                ),  # unused D_offsets
                -1,  # unused max_D
                False,  # nobag
                vbe_metadata.max_B_feature_rank,
                info_B_num_bits,
                offsets.numel() - 1,  # total_B
            )
            row_output_offsets = row_output_offsets.npu()
            b_t_map = b_t_map.npu()
            B_offsets = B_offsets.npu()
            vbe_args = {
                "B_offsets": B_offsets,
                "max_B": max_B,
                "b_t_map": b_t_map,
                "info_B_num_bits": info_B_num_bits,
                "info_B_mask": info_B_mask,
            }
        else:
            vbe_args = {}

        self.assertEqual(indices.numel(), np.sum(Ls).item())
        self.assertEqual(offsets[-1], np.sum(Ls).item())
        indices, offsets, rows_per_table, warning = (
            indices.npu(),
            offsets.npu(),
            rows_per_table.npu(),
            warning.npu(),
        )
        if weighted:
            weights = weights.npu()
        indices_copy = indices.clone()
        offsets_copy = offsets.clone()
        torch.npu.synchronize()
        # 测试indices索引在范围内的正常的情况
        torch.ops.fbgemm.bounds_check_indices(
            rows_per_table,
            indices,
            offsets,
            bounds_check_mode,
            warning,
            weights,
            bounds_check_version=bounds_check_version,
            **vbe_args,
        )
        torch.npu.synchronize()
        indices_copy_cpu = indices_copy.cpu()
        indices_cpu = indices.cpu()
        torch.testing.assert_close(indices_copy_cpu, indices_cpu)
        torch.npu.synchronize()
        # 测试 indices 索引不在范围内的情况
        indices[:] = torch.iinfo(dtype).max
        torch.npu.synchronize()
        if bounds_check_mode != BoundsCheckMode.FATAL:
            torch.ops.fbgemm.bounds_check_indices(
                rows_per_table,
                indices,
                offsets,
                bounds_check_mode,
                warning,
                weights,
                bounds_check_version=bounds_check_version,
                **vbe_args,
            )
            torch.npu.synchronize()
            torch.testing.assert_close(indices, torch.zeros_like(indices))
            if bounds_check_mode == BoundsCheckMode.WARNING:
                self.assertEqual(warning.item(), indices.numel())

        # 测试 offsets 边界不正确的情况
        torch.npu.synchronize()
        indices.copy_(indices_copy)
        offsets.copy_(offsets_copy)
        torch.npu.synchronize()
        if offsets.numel() > 0:
            offsets[0] = -100
        if offsets.numel() > 1:
            offsets[-1] += 100
        torch.npu.synchronize()
        if bounds_check_mode != BoundsCheckMode.FATAL:
            torch.ops.fbgemm.bounds_check_indices(
                rows_per_table,
                indices,
                offsets,
                bounds_check_mode,
                warning,
                weights,
                bounds_check_version=bounds_check_version,
                **vbe_args,
            )
            torch.npu.synchronize()
            if offsets.numel() > 0:
                self.assertEqual(offsets[0].item(), 0)
            if offsets.numel() > 1:
                self.assertEqual(offsets[-1].item(), indices.numel())
            if bounds_check_mode == BoundsCheckMode.WARNING:
                # -1 because when we have 2 elements in offsets, we have only 1
                # warning for the pair.
                self.assertGreaterEqual(warning.item(), min(2, offsets.numel() - 1))

        # test offsets.size(0) ! = B * T + 1 case. Here we test with T >= 2 case.
        # If T == 1, we will always get the even division.
        # (does not apply to mixed_B = True)
        if not mixed_B and T >= 2:
            indices = indices_copy.clone()
            offsets = offsets_copy.clone()
            offsets = torch.cat(
                (
                    offsets,
                    torch.tensor(
                        [indices.numel()] * (T - 1),
                        dtype=offsets.dtype,
                        device=offsets.device,
                    ),
                ),
                dim=0,
            )
            with self.assertRaises(RuntimeError):
                torch.ops.fbgemm.bounds_check_indices(
                    rows_per_table,
                    indices,
                    offsets,
                    bounds_check_mode,
                    warning,
                    weights,
                    bounds_check_version=bounds_check_version,
                )

        # 测试weights和indices的长度不一致的情况
        weights = torch.rand(
            (indices.size(0) + 1,), dtype=torch.float, device=indices.device
        )
        with self.assertRaises(RuntimeError):
            torch.ops.fbgemm.bounds_check_indices(
                rows_per_table,
                indices,
                offsets,
                bounds_check_mode,
                warning,
                weights,
                **vbe_args,
                bounds_check_version=bounds_check_version,
            )

    @given(
        T=st.integers(min_value=1, max_value=5),
        B=st.integers(min_value=1, max_value=8),
        L=st.integers(min_value=0, max_value=8),
        use_cpu=use_cpu_strategy(),
        use_cpu_hashtable=st.booleans(),
        use_array_for_index_remapping=st.booleans(),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_pruning(
        self,
        T: int,
        B: int,
        L: int,
        use_cpu: bool,
        use_cpu_hashtable: bool,
        use_array_for_index_remapping: bool,
    ) -> None:
        E = int(1000)
        LOAD_FACTOR = 0.8
        pruning_ratio = 0.5

        capacities: List[int] = [int(B * L / LOAD_FACTOR) + 1 for _ in range(T)]
        original_E = int(E / (1.0 - pruning_ratio))

        # Enforce the size of original_E/B/L to get the unique indices
        assume(original_E > B * L)

        current_device = "cpu" if use_cpu else torch.cuda.current_device()

        if use_cpu_hashtable:
            assume(use_cpu)

        indices = torch.randint(low=0, high=original_E, size=(T, B, L))
        for t in range(T):
            while (
                torch.unique(
                    indices[t], return_counts=False, return_inverse=False
                ).numel()
                != indices[t].numel()
            ):
                indices[t] = torch.randint(low=0, high=original_E, size=(B, L))

        indices = indices.view(-1).int()
        dense_indices = torch.randint(low=0, high=E, size=(T, B, L)).view(-1).int()
        offsets = torch.tensor([L * b_t for b_t in range(B * T + 1)]).int()

        # Initialize and insert Hashmap index remapping based data structure
        hash_table = torch.empty(
            (sum(capacities), 2),
            dtype=torch.int64,
        )
        hash_table[:, :] = -1
        hash_table_offsets = torch.tensor([0] + np.cumsum(capacities).tolist()).long()

        torch.ops.fbgemm.pruned_hashmap_insert(
            indices, dense_indices, offsets, hash_table, hash_table_offsets
        )

        if use_cpu_hashtable:
            # pyre-ignore[16]
            ht = torch.classes.fbgemm.PrunedMapCPU()
            ht.insert(indices, dense_indices, offsets, T)

        # Initialize and insert Array index remapping based data structure
        index_remappings_array = torch.tensor(
            [-1] * original_E * T,
            dtype=torch.int64,
            device=current_device,
        )
        index_remappings_array_offsets = torch.empty(
            T + 1,
            dtype=torch.int64,
            device=current_device,
        )
        index_remappings_array_offsets[0] = 0
        for t in range(T):
            indice_t = (indices.view(T, B, L))[t].long().view(-1).to(current_device)
            dense_indice_t = (
                (dense_indices.view(T, B, L))[t].long().view(-1).to(current_device)
            )
            selected_indices = torch.add(indice_t, t * original_E)[:E]
            index_remappings_array[selected_indices] = dense_indice_t
            index_remappings_array_offsets[t + 1] = (
                index_remappings_array_offsets[t] + original_E
            )

        # Move data when using device
        if not use_cpu:
            (
                indices,
                dense_indices,
                offsets,
                hash_table,
                hash_table_offsets,
                index_remappings_array,
                index_remappings_array_offsets,
            ) = (
                indices.to(current_device),
                dense_indices.to(current_device),
                offsets.to(current_device),
                hash_table.to(current_device),
                hash_table_offsets.to(current_device),
                index_remappings_array.to(current_device),
                index_remappings_array_offsets.to(current_device),
            )

        # Lookup
        if use_cpu_hashtable:
            dense_indices_ = ht.lookup(indices, offsets)
        elif not use_array_for_index_remapping:  # hashmap based pruning
            dense_indices_ = torch.ops.fbgemm.pruned_hashmap_lookup(
                indices, offsets, hash_table, hash_table_offsets
            )
        else:  # array based pruning
            dense_indices_ = torch.ops.fbgemm.pruned_array_lookup(
                indices,
                offsets,
                index_remappings_array,
                index_remappings_array_offsets,
            )

        # Validate the lookup result
        torch.testing.assert_close(dense_indices, dense_indices_)

        # For array based pruning, it will be out-of-boundary for arbitrarily
        # large indices. We will rely on bound checker to make sure indices
        # are within the boundary.
        if not use_array_for_index_remapping:
            # now, use a value that does not exist in the original set of indices
            # and so should be pruned out.
            indices[:] = np.iinfo(np.int32).max

            if use_cpu_hashtable:
                dense_indices_ = ht.lookup(indices, offsets)
            elif not use_array_for_index_remapping:  # hashmap based pruning
                dense_indices_ = torch.ops.fbgemm.pruned_hashmap_lookup(
                    indices, offsets, hash_table, hash_table_offsets
                )
            else:  # array based pruning
                dense_indices_ = torch.ops.fbgemm.pruned_array_lookup(
                    indices,
                    offsets,
                    index_remappings_array,
                    index_remappings_array_offsets,
                )
            torch.testing.assert_close(dense_indices.clone().fill_(-1), dense_indices_)


if __name__ == "__main__":
    unittest.main()
