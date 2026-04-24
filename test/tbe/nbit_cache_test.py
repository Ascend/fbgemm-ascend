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

import hypothesis.strategies as st
import numpy as np
import torch
from common import npu_unavailable, MAX_EXAMPLES
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    EmbeddingSpecInfo,
    get_new_embedding_location,
    RecordCacheMetrics,
    tensor_to_device,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from hypothesis import given, settings, Verbosity

VERBOSITY: Verbosity = Verbosity.verbose


class NBitCacheTest(unittest.TestCase):
    @unittest.skipIf(*npu_unavailable)
    @given(
        L=st.integers(min_value=0, max_value=16),
        H=st.integers(min_value=512, max_value=1024),
        S=st.integers(min_value=0, max_value=128),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_nbit_cache_update_function(self, L: int, H: int, S: int) -> None:
        # Generate synthetic data
        linear_cache_indices_cpu = torch.randint(L, H, (S,))
        lxu_cache_locations_cpu = torch.clone(linear_cache_indices_cpu)

        indices = [True if np.random.rand() < 0.5 else False for _ in range(S)]
        lxu_cache_locations_cpu[indices] = -1

        cache_miss_ids = torch.clone(linear_cache_indices_cpu)
        cache_miss_ids[lxu_cache_locations_cpu != -1] = -2

        # Calculate the correct output
        unique_cache_miss_ids = torch.unique(cache_miss_ids)
        expect_out = sum(unique_cache_miss_ids >= 0)
        linear_cache_indices = linear_cache_indices_cpu.to(torch.int32).npu()
        lxu_cache_locations = lxu_cache_locations_cpu.to(torch.int32).npu()
        expected_unique_access = len(torch.unique(linear_cache_indices_cpu))
        expected_total_access = len(linear_cache_indices_cpu)

        # Create an abstract split table
        D = 8
        T = 2
        E = 10 ** 3
        Ds = [D] * T
        Es = [E] * T
        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    SparseType.INT8,
                    EmbeddingLocation.MANAGED_CACHING,
                )
                for (E, D) in zip(Es, Ds)
            ],
            device=torch.npu.current_device(),
            record_cache_metrics=RecordCacheMetrics(True, False),
        )
        cc.fill_random_weights()

        cc._update_cache_miss_counter(lxu_cache_locations, linear_cache_indices)
        (
            cache_miss_forward_count,
            unique_cache_miss_count,
            unique_access_count,
            total_access_count,
        ) = cc.get_cache_miss_counter().cpu()

        self.assertEqual(unique_cache_miss_count, expect_out)
        self.assertLessEqual(cache_miss_forward_count, unique_cache_miss_count)
        self.assertEqual(unique_access_count, expected_unique_access)
        self.assertEqual(total_access_count, expected_total_access)

    @unittest.skipIf(*npu_unavailable)
    @given(
        device_to_str=st.sampled_from(["npu", "cpu"]),
        cache_load_factor=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(verbosity=VERBOSITY, max_examples=1, deadline=None)
    def test_nbit_move_to_device_with_cache(
            self,
            device_to_str: str,
            cache_load_factor: float,
    ) -> None:
        # Create an abstract split table
        D = 8
        T = 2
        E = 10 ** 3
        Ds = [D] * T
        Es = [E] * T
        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    SparseType.INT8,
                    EmbeddingLocation.MANAGED_CACHING,
                )
                for (E, D) in zip(Es, Ds)
            ],
            device=torch.npu.current_device(),
            gather_uvm_cache_stats=True,
            cache_assoc=1,  # Direct Mapped
        )
        cc.fill_random_weights()

        default_uvm_tensor = cc.weights_uvm.clone()

        emb_location = get_new_embedding_location(
            torch.device(device_to_str), cache_load_factor
        )

        cc.move_to_device_with_cache(
            torch.device(device_to_str), cache_load_factor=cache_load_factor
        )

        all_emb_locations = [es[EmbeddingSpecInfo.embedding_location] for es in cc.embedding_specs]
        self.assertTrue(all(emb_location == loc for loc in all_emb_locations))

        self.assertTrue(cc.current_device == torch.device(device_to_str))
        other_tensor: torch.Tensor
        if emb_location == EmbeddingLocation.DEVICE:
            other_tensor = cc.weights_dev.clone()
        elif (
                emb_location == EmbeddingLocation.MANAGED
                or emb_location == EmbeddingLocation.MANAGED_CACHING
        ):
            other_tensor = cc.weights_uvm.clone()
        else:  # emb_location == EmbeddingLocation.HOST
            other_tensor = cc.weights_host.clone()

        self.assertEqual(len(default_uvm_tensor), len(other_tensor))

        # Move tensor for comparison
        tensor_to_device(other_tensor, torch.device(device_to_str))
        for i in range(len(default_uvm_tensor)):
            self.assertTrue(torch.equal(default_uvm_tensor[i], other_tensor[i]))


if __name__ == "__main__":
    unittest.main()
