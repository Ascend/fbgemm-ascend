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
from typing import Optional

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given, settings, Verbosity

import fbgemm_gpu
import fbgemm_ascend


def npu_available() -> bool:
    return (
            hasattr(torch, "npu")
            and torch.npu.is_available()
            and torch.npu.device_count() > 0
    )


if npu_available():
    torch.npu.set_device("npu:0")


class CacheTest(unittest.TestCase):
    def _get_unique_indices_reference(
            self,
            linear_indices: torch.Tensor,
            max_indices: int,
            compute_count: bool,
            compute_inverse_indices: bool,
    ) -> tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        N = linear_indices.numel()

        indices_list = linear_indices.tolist()
        unique_vals_list = sorted(set(indices_list))
        num_unique = len(unique_vals_list)

        unique_indices = torch.empty_like(linear_indices)
        if num_unique > 0:
            unique_indices[:num_unique] = torch.tensor(
                unique_vals_list, dtype=linear_indices.dtype
            )

        unique_indices_length = torch.tensor([num_unique], dtype=torch.int32)

        unique_indices_count = None
        if compute_count:
            count_dict = {}
            for val in indices_list:
                count_dict[val] = count_dict.get(val, 0) + 1

            counts_list = [count_dict[val] for val in unique_vals_list]

            unique_indices_count = torch.empty(N, dtype=torch.int32)
            if num_unique > 0:
                unique_indices_count[:num_unique] = torch.tensor(
                    counts_list, dtype=torch.int32
                )

        linear_index_positions_sorted = None
        if compute_inverse_indices:
            indexed_list = [(val, idx) for idx, val in enumerate(indices_list)]
            sorted_indexed = sorted(indexed_list, key=lambda x: x[0])
            positions_list = [pos for val, pos in sorted_indexed]
            linear_index_positions_sorted = torch.tensor(
                positions_list, dtype=torch.int32
            )

        return (
            unique_indices,
            unique_indices_length,
            unique_indices_count,
            linear_index_positions_sorted,
        )

    @given(
        N=st.integers(min_value=0, max_value=1000),
        max_indices=st.integers(min_value=100, max_value=10000),
        compute_count=st.booleans(),
        compute_inverse_indices=st.booleans(),
        dtype=st.sampled_from([torch.int, torch.long]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=40, deadline=None)
    def test_get_unique_indices(
            self,
            N: int,
            max_indices: int,
            compute_count: bool,
            compute_inverse_indices: bool,
            dtype: torch.dtype,
    ) -> None:
        linear_indices = torch.randint(0, max_indices, (N,), dtype=dtype)
        expected = self._get_unique_indices_reference(
            linear_indices.cpu(),
            max_indices,
            compute_count,
            compute_inverse_indices,
        )
        expected_unique, expected_length, expected_count, expected_inverse = expected

        if compute_inverse_indices:
            unique_cpu, length_cpu, count_cpu, inverse_cpu = (
                torch.ops.fbgemm.get_unique_indices_with_inverse(
                    linear_indices,
                    max_indices,
                    compute_count,
                    compute_inverse_indices,
                )
            )
        else:
            unique_cpu, length_cpu, count_cpu = torch.ops.fbgemm.get_unique_indices(
                linear_indices,
                max_indices,
                compute_count,
            )
            inverse_cpu = None

        def compare_output(
                input_indices: torch.Tensor,
                annotate1: str,
                annotate2: str,
                length1: int,
                length2: int,
                unique1: torch.Tensor,
                unique2: torch.Tensor,
                compute_count: bool,
                compute_inverse_indices: bool,
                count1: Optional[torch.Tensor] = None,
                count2: Optional[torch.Tensor] = None,
                positions1: Optional[torch.Tensor] = None,
                positions2: Optional[torch.Tensor] = None,
        ) -> None:
            self.assertEqual(
                length1,
                length2,
                f"{annotate1} unique indices length mismatch with {annotate2}",
            )

            torch.testing.assert_close(
                unique1[:length1].cpu(),
                unique2[:length2].cpu(),
                msg=f"{annotate1} unique indices mismatch with {annotate2}",
            )

            if compute_count:
                self.assertIsNotNone(count1, f"{annotate1} count should not be None")
                self.assertIsNotNone(count2, f"{annotate2} count should not be None")
                torch.testing.assert_close(
                    count1[:length1].cpu(),
                    count2[:length2].cpu(),
                    msg=f"{annotate1} unique indices count mismatch with {annotate2}",
                )

            if compute_inverse_indices:
                self.assertIsNotNone(
                    positions1, f"{annotate1} positions should not be None"
                )
                self.assertIsNotNone(
                    positions2, f"{annotate2} positions should not be None"
                )
                torch.testing.assert_close(
                    positions1.cpu(),
                    positions2.cpu(),
                    msg=f"{annotate1} unique indices position mismatch with {annotate2}",
                )
                reordered1 = input_indices.gather(
                    0, positions1.long().to(input_indices.device)
                )
                reordered2 = input_indices.gather(
                    0, positions2.long().to(input_indices.device)
                )
                torch.testing.assert_close(
                    reordered1.cpu(),
                    reordered2.cpu(),
                    msg=f"{annotate1} reordered indices mismatch with {annotate2}",
                )

        compare_output(
            linear_indices,
            "CPU",
            "ref implementation",
            length_cpu.item(),
            int(expected_length.item()),
            unique_cpu,
            expected_unique,
            compute_count,
            compute_inverse_indices,
            count_cpu,
            expected_count,
            inverse_cpu,
            expected_inverse,
        )

        if npu_available():
            linear_indices_npu = linear_indices.npu()
            if compute_inverse_indices:
                unique_npu, length_npu, count_npu, inverse_npu = (
                    torch.ops.fbgemm.get_unique_indices_with_inverse(
                        linear_indices_npu,
                        max_indices,
                        compute_count,
                        compute_inverse_indices,
                    )
                )
            else:
                unique_npu, length_npu, count_npu = torch.ops.fbgemm.get_unique_indices(
                    linear_indices_npu,
                    max_indices,
                    compute_count,
                )
                inverse_npu = None

            compare_output(
                linear_indices,
                "CPU",
                "NPU",
                length_cpu.item(),
                length_npu.item(),
                unique_cpu,
                unique_npu,
                compute_count,
                compute_inverse_indices,
                count_cpu,
                count_npu,
                inverse_cpu,
                inverse_npu,
            )


if __name__ == "__main__":
    unittest.main()
