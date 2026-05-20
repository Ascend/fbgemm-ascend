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

import numpy as np
import torch
import torch._dynamo
from hypothesis import given, settings, Verbosity
import hypothesis.strategies as st

from common import var_list_to_coo


class Jagged2DToDenseTest(unittest.TestCase):
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    @given(
        B=st.integers(min_value=1, max_value=128),
        D=st.integers(min_value=1, max_value=128),
        max_sequence_length=st.integers(min_value=1, max_value=200),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
    )
    def test_jagged_2d_to_dense(
        self,
        B: int,
        D: int,
        max_sequence_length: int,
        dtype: torch.dtype,
    ) -> None:
        D = D * 4
        lengths_ = np.random.randint(low=0, high=max_sequence_length, size=B)
        total_lengths = lengths_.sum()
        lengths = torch.from_numpy(lengths_)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)

        ref_values = torch.rand(total_lengths, D)
        ref_output_values = var_list_to_coo(
            lengths,
            ref_values,
            max_sequence_length,
            D,
        ).to_dense()
        ref_output_values = ref_output_values.to(dtype)

        # test cpu forward
        values = ref_values.clone().to(dtype).detach().requires_grad_(True)
        output_values = torch.ops.fbgemm.jagged_2d_to_dense(
            values=values,
            offsets=offsets,
            max_sequence_length=max_sequence_length,
        )
        torch.testing.assert_close(ref_output_values, output_values)

        if torch.npu.is_available():
            # test npu forward
            ref_values = ref_values.npu()
            values = ref_values.clone().to(dtype).detach().requires_grad_(True)
            offsets = offsets.npu()
            ref_output_values = ref_output_values.npu()
            output_values = torch.ops.fbgemm.jagged_2d_to_dense(
                values=values,
                offsets=offsets,
                max_sequence_length=max_sequence_length,
            )
            torch.testing.assert_close(ref_output_values, output_values)

            # test npu backward
            output_values.backward(ref_output_values)
            ref_values = ref_values.to(dtype)
            torch.testing.assert_close(ref_values, values.grad)


if __name__ == "__main__":
    unittest.main()
