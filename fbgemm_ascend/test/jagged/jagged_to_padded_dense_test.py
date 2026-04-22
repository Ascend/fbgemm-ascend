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
from hypothesis import assume, given, settings, Verbosity

from common import (
    generate_jagged_tensor,
    cpu_and_maybe_npu,
    to_padded_dense
)


class JaggedToPaddedDenseTest(unittest.TestCase):
    @given(
        num_jagged_dim=st.just(1),  # Only supports ops with num_jagged_dim=1
        outer_dense_size=st.integers(0, 5),
        inner_dense_size=st.integers(0, 5),
        fold_inner_dense=st.just(False),  # Not supports ops with 1D values
        padding_value=st.sampled_from([0, -1e-8]),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=cpu_and_maybe_npu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_to_padded_dense(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        fold_inner_dense: bool,
        padding_value: float,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        # CPU doesn't support bfloat16
        assume(device_type != "cpu" or dtype != torch.bfloat16)
        assume(not fold_inner_dense or inner_dense_size == 1)

        device = torch.device(device_type)
        x_values, x_offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            torch.float,
            device,
            fold_inner_dense,
        )

        output_ref = to_padded_dense(
            x_values, x_offsets, max_lengths, padding_value=padding_value
        )
        output = torch.ops.fbgemm.jagged_to_padded_dense(
            x_values,
            x_offsets,
            max_lengths,
            padding_value=padding_value,
        )

        torch.testing.assert_close(output, output_ref)

    @given(
        num_jagged_dim=st.integers(1, 5),
        outer_dense_size=st.integers(0, 5),
        inner_dense_size=st.integers(0, 5),
        padding_value=st.just(0),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=st.just("meta"),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_to_padded_dense_meta_backend(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        padding_value: float,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        assume(device_type != "cpu" or dtype != torch.bfloat16)
        device = torch.device("cpu")

        x_values, x_offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, torch.float, device
        )

        output_ref = to_padded_dense(
            x_values, x_offsets, max_lengths, padding_value=padding_value
        )
        x_values.to(device_type)
        output = torch.ops.fbgemm.jagged_to_padded_dense(
            x_values,
            x_offsets,
            max_lengths,
            padding_value=padding_value,
        )

        assert output.size() == output_ref.size()


if __name__ == "__main__":
    unittest.main()
