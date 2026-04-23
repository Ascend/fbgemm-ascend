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

import hypothesis.strategies as st
import torch
import torch._dynamo
from hypothesis import given, settings

from common import npu_unavailable


class KeyedJaggedIndexSelectTest(unittest.TestCase):
    @unittest.skipIf(*npu_unavailable)
    @given(
        max_seq_length=st.integers(5, 10),
        input_batch_size=st.integers(1, 128),
        output_batch_size=st.integers(1, 128),
        num_batches=st.integers(1, 3),
        index_dtype=st.sampled_from([torch.int, torch.long]),
        jagged_tensor_dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
                torch.int,
                torch.long,
            ]  # Disable torch.bfloat16 due to large error bound
        ),
        has_weights=st.just(False),  # Not supports weights
        check_non_contiguous=st.booleans(),
        use_selected_lengths_sum=st.booleans(),
    )
    @settings(max_examples=20, deadline=None)
    def test_keyed_jagged_index_select_dim1(
        self,
        max_seq_length: int,
        input_batch_size: int,
        output_batch_size: int,
        num_batches: int,
        index_dtype: torch.dtype,
        jagged_tensor_dtype: torch.dtype,
        has_weights: bool,
        check_non_contiguous: bool,
        use_selected_lengths_sum: bool,
    ) -> None:
        is_float = jagged_tensor_dtype in [torch.float, torch.half, torch.bfloat16]
        lengths = torch.randint(
            low=0,
            high=max_seq_length,
            size=(input_batch_size * num_batches,),
            dtype=index_dtype,
            device="npu",
        )
        offsets = torch.concat(
            [torch.zeros(1, dtype=torch.long, device="npu"), lengths.cumsum(0)]
        )
        indices = torch.randint(
            low=0,
            high=input_batch_size,
            size=(output_batch_size,),
            dtype=index_dtype,
            device="npu",
        )

        # If check_non_contiguous=True, create a tensor that is twice as big
        # and then select only odd indices to make it non contiguous
        values_numel = int(offsets[-1].item())
        values_numel = values_numel * 2 if check_non_contiguous else values_numel

        if is_float:
            values = torch.rand(
                values_numel,
                dtype=jagged_tensor_dtype,
                device="npu",
            )
        else:
            values = torch.randint(
                2**16,
                (values_numel,),
                dtype=jagged_tensor_dtype,
                device="npu",
            )
        values_ref = values.detach().clone()

        if check_non_contiguous:
            values = values[1::2]
            values_ref = values_ref[1::2]

        if has_weights:
            weights = torch.rand(
                int(offsets[-1].item()),
                dtype=random.choice([torch.float, torch.half]),
                device="npu",
            )
        else:
            weights = None

        if use_selected_lengths_sum:
            length_indices = torch.cat(
                [indices + i * input_batch_size for i in range(num_batches)]
            )
            selected_lengths_sum = (
                torch.index_select(lengths, 0, length_indices).sum().item()
            )
        else:
            selected_lengths_sum = None

        # Only float tensors can require grad
        if is_float:
            values.requires_grad = True
            values_ref.requires_grad = True

        index_select_output = torch.ops.fbgemm.keyed_jagged_index_select_dim1(
            values,
            lengths,
            offsets,
            indices,
            input_batch_size,
            weights,
            selected_lengths_sum,
        )
        output = index_select_output[0]
        if has_weights:
            output_weights = index_select_output[2]

        output_ref = []
        output_weight_ref = []
        for k in range(num_batches):
            key_lengths = lengths[k * input_batch_size : (k + 1) * input_batch_size]
            start_offset = offsets[k * input_batch_size]
            end_offset = offsets[(k + 1) * input_batch_size]
            key_values = values_ref[start_offset:end_offset].view(-1, 1)
            output_ref.append(
                torch.ops.fbgemm.jagged_index_select(key_values, key_lengths, indices)[
                    0
                ].view(-1)
            )
            if has_weights:
                # pyre-ignore[16]
                key_weights = weights[start_offset:end_offset].view(-1, 1)
                output_weight_ref.append(
                    torch.ops.fbgemm.jagged_index_select(
                        key_weights, key_lengths, indices
                    )[0].view(-1)
                )

        output_ref = torch.concat(output_ref)
        assert torch.equal(output, output_ref)

        if has_weights:
            output_weight_ref = torch.concat(output_weight_ref)
            assert torch.equal(output_weights, output_weight_ref)


if __name__ == "__main__":
    unittest.main()
