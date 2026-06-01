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
import torch._dynamo
from hypothesis import given, settings, Verbosity
from torch.autograd import gradcheck

from common import (
    generate_jagged_tensor,
    to_padded_dense,
)


class ElementwiseBinaryTest(unittest.TestCase):
    def _test_jagged_elementwise_binary(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        operation: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        x_values, x_offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )
        y = torch.rand(
            outer_dense_size * np.prod(max_lengths) * inner_dense_size,
            dtype=dtype,
            device=device,
        ).reshape((outer_dense_size,) + tuple(max_lengths) + (inner_dense_size,))

        x_padded = to_padded_dense(x_values, x_offsets, max_lengths)
        if operation == "add":
            output_ref = x_padded + y
            output = torch.ops.fbgemm.jagged_dense_elementwise_add(x_values, x_offsets, y)
        elif operation == "add_jagged_output":
            # create a jagged tensor and then densify
            y = to_padded_dense(
                torch.rand(
                    (
                        max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                        inner_dense_size,
                    ),
                    dtype=dtype,
                    device=device,
                ),
                x_offsets,
                max_lengths,
            )
            output_ref = x_padded + y
            (
                output,
                output_offsets,
            ) = torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(x_values, x_offsets, y)
            output = to_padded_dense(output, output_offsets, max_lengths)
        elif operation == "mul":
            output_ref = x_padded * y
            output, output_offsets = torch.ops.fbgemm.jagged_dense_elementwise_mul(x_values, x_offsets, y)
            output = to_padded_dense(output, output_offsets, max_lengths)
        else:
            raise AssertionError(f"Unknown operation {operation}")

        torch.testing.assert_close(output, output_ref)

        if operation == "add":
            f = torch.ops.fbgemm.jagged_dense_elementwise_add
        elif operation == "add_jagged_output":
            # pyre-fixme[2]: Parameter must be annotated.
            def add_jagged_output_func(*args) -> torch.Tensor:
                return torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(*args)[0]

            f = add_jagged_output_func
        else:
            assert operation == "mul"

            # pyre-fixme[2]: Parameter must be annotated.
            def mul_func(*args) -> torch.Tensor:
                return torch.ops.fbgemm.jagged_dense_elementwise_mul(*args)[0]

            f = mul_func

        gradcheck(
            f,
            (
                x_values.float().requires_grad_(True),
                x_offsets,
                y.float().requires_grad_(True),
            ),
            eps=1e-2,
            atol=1e-3,
            rtol=1e-3,
        )

    @given(
        num_jagged_dim=st.integers(1, 4),
        outer_dense_size=st.integers(0, 4),
        inner_dense_size=st.integers(0, 4),
        operation=st.sampled_from(["add", "add_jagged_output", "mul"]),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device=st.just("npu"),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_elementwise_binary(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        operation: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._test_jagged_elementwise_binary(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            operation,
            dtype,
            device,
        )

    @given(
        num_jagged_dim=st.just(1),
        outer_dense_size=st.integers(0, 8),
        inner_dense_size=st.sampled_from([16, 64, 96, 192]),
        operation=st.sampled_from(["add_jagged_output", "mul"]),
        dtype=st.just(torch.half),
        device=st.just("npu"),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_jagged_elementwise_binary_opt(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        operation: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._test_jagged_elementwise_binary(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            operation,
            dtype,
            device,
        )

    @given(
        num_jagged_dim=st.integers(1, 5),
        outer_dense_size=st.integers(2, 5),
        inner_dense_size=st.integers(2, 5),
        operation=st.sampled_from(["add", "add_jagged_output", "mul"]),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device=st.just("npu"),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_elementwise_binary_dynamic_shape(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        operation: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        # Start a fresh compile for each parameter of the test case
        torch._dynamo.reset()

        x_values, x_offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            device,
            mark_dynamic=True,
        )
        y = torch.rand(
            outer_dense_size * np.prod(max_lengths) * inner_dense_size,
            dtype=dtype,
            device=device,
        ).reshape((outer_dense_size,) + tuple(max_lengths) + (inner_dense_size,))

        x_padded = to_padded_dense(x_values, x_offsets, max_lengths)

        def jagged_dense_elementwise_add(
            x_values: torch.Tensor, x_offsets: list[torch.LongTensor], y: torch.Tensor
        ) -> torch.Tensor:
            return torch.ops.fbgemm.jagged_dense_elementwise_add(x_values, x_offsets, y)

        def jagged_dense_elementwise_add_jagged_output(
            x_values: torch.Tensor, x_offsets: list[torch.LongTensor], y: torch.Tensor
        ) -> tuple[torch.Tensor, list[torch.LongTensor]]:
            return torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(x_values, x_offsets, y)

        def jagged_dense_elementwise_mul(
            x_values: torch.Tensor, x_offsets: list[torch.LongTensor], y: torch.Tensor
        ) -> tuple[torch.Tensor, list[torch.LongTensor]]:
            return torch.ops.fbgemm.jagged_dense_elementwise_mul(x_values, x_offsets, y)

        if operation == "add":
            output_ref = x_padded + y
            output = jagged_dense_elementwise_add(x_values, x_offsets, y)

        elif operation == "add_jagged_output":
            # create a jagged tensor and then densify
            y = to_padded_dense(
                torch.rand(
                    (
                        max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                        inner_dense_size,
                    ),
                    dtype=dtype,
                    device=device,
                ),
                x_offsets,
                max_lengths,
            )
            output_ref = x_padded + y
            (
                output,
                output_offsets,
            ) = jagged_dense_elementwise_add_jagged_output(x_values, x_offsets, y)
            output = to_padded_dense(output, output_offsets, max_lengths)

        elif operation == "mul":
            output_ref = x_padded * y
            output, output_offsets = jagged_dense_elementwise_mul(x_values, x_offsets, y)
            output = to_padded_dense(output, output_offsets, max_lengths)
        else:
            raise AssertionError(f"Unknown operation {operation}")

        assert output.size() == output_ref.size()


if __name__ == "__main__":
    unittest.main()
