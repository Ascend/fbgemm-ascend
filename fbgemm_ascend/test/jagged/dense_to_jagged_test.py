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
import torch
import torch._dynamo
from hypothesis import given, settings, Verbosity
import fbgemm_gpu

from common import (
    cpu_and_maybe_npu,
    npu_unavailable,
    generate_jagged_tensor
)


class DenseToJaggedTest(unittest.TestCase):
    def _test_dense_to_jagged(
            self,
            num_jagged_dim: int,
            outer_dense_size: int,
            inner_dense_size: int,
            dtype: torch.dtype,
            device: torch.device,
            precompute_total_L: bool,
    ) -> None:
        # Generate multi-dim jagged tensor
        values_2d, offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )
        values_2d = values_2d.clone().detach().requires_grad_(True)
        # jagged -> dense
        if max_lengths.size != 1:
            return
        dense = torch.ops.fbgemm.jagged_to_padded_dense(values_2d, offsets, max_lengths)
        # dense -> jagged (op which is being tested)
        if precompute_total_L:
            total_L = values_2d.size(0)
            jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                dense, offsets, total_L
            )
        else:
            jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                dense, offsets
            )

        # jagged -> dense
        dense2 = torch.ops.fbgemm.jagged_to_padded_dense(
            jagged_values, jagged_offsets, max_lengths
        )

        # verify forward
        torch.testing.assert_close(dense, dense2)

        # verify backward
        dense.retain_grad()
        ref_output_values = jagged_values.clone().detach().requires_grad_(True)
        ref_values = dense.clone().detach().requires_grad_(True)
        jagged_values.backward(ref_output_values)
        torch.testing.assert_close(dense.grad, ref_values)

    @given(
        num_jagged_dim=st.integers(1, 5),
        outer_dense_size=st.integers(1, 5),
        inner_dense_size=st.integers(1, 5),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device=cpu_and_maybe_npu(),
        precompute_total_L=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_dense_to_jagged(
            self,
            num_jagged_dim: int,
            outer_dense_size: int,
            inner_dense_size: int,
            dtype: torch.dtype,
            device: torch.device,
            precompute_total_L: bool,
    ) -> None:
        self._test_dense_to_jagged(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            device,
            precompute_total_L,
        )

    @unittest.skipIf(*npu_unavailable)
    @given(
        num_jagged_dim=st.just(1),
        outer_dense_size=st.integers(1, 6000),
        inner_dense_size=st.sampled_from([8, 16, 23, 24, 48, 50, 64, 72, 96, 192]),
        dtype=st.just(torch.half),
        device=cpu_and_maybe_npu(),
        precompute_total_L=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_dense_to_jagged_opt(
            self,
            num_jagged_dim: int,
            outer_dense_size: int,
            inner_dense_size: int,
            dtype: torch.dtype,
            device: torch.device,
            precompute_total_L: bool,
    ) -> None:
        self._test_dense_to_jagged(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            device,
            precompute_total_L,
        )

    # (8000+1) * 8 (size of the element of LongTensor/int64_t offsets)
    # = ~62.5KB > 48KB default shared memory on V100/A100.
    @unittest.skipIf(*npu_unavailable)
    @given(
        num_jagged_dim=st.just(1),
        outer_dense_size=st.just(8000),
        inner_dense_size=st.just(16),
        dtype=st.just(torch.half),
        device=cpu_and_maybe_npu(),
        precompute_total_L=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_dense_to_jagged_opt_large_batch(
            self,
            num_jagged_dim: int,
            outer_dense_size: int,
            inner_dense_size: int,
            dtype: torch.dtype,
            device: torch.device,
            precompute_total_L: bool,
    ) -> None:
        self._test_dense_to_jagged(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            device,
            precompute_total_L,
        )

    @given(
        num_jagged_dim=st.integers(1, 5),
        outer_dense_size=st.integers(0, 5),
        inner_dense_size=st.integers(0, 5),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device=st.sampled_from([torch.device("meta")]),
        precompute_total_L=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_dense_to_jagged_meta_backend(
            self,
            num_jagged_dim: int,
            outer_dense_size: int,
            inner_dense_size: int,
            dtype: torch.dtype,
            device: torch.device,
            precompute_total_L: bool,
    ) -> None:
        device = torch.device("cpu")
        values_2d, offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )
        values_2d = values_2d.clone().detach().requires_grad_(True)

        # jagged -> dense
        dense = torch.ops.fbgemm.jagged_to_padded_dense(values_2d, offsets, max_lengths)

        # dense -> jagged (op which is being tested)
        if precompute_total_L:
            total_L = values_2d.size(0)
            dense.to(device)
            jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                dense, offsets, total_L
            )
        else:
            dense.to(device)
            jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                dense, offsets
            )

        jagged_values.to(device)
        # jagged -> dense
        dense2 = torch.ops.fbgemm.jagged_to_padded_dense(
            jagged_values, jagged_offsets, max_lengths
        )

        # verify forward
        assert dense.size() == dense2.size()


if __name__ == "__main__":
    unittest.main()
