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
from itertools import accumulate
from typing import Optional

import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity, HealthCheck

from common import (
    npu_available, 
    npu_unavailable
)

suppressed_list: list[HealthCheck] = (
    [HealthCheck.differing_executors]
    if getattr(HealthCheck, "differing_executors", False)
    else []
)


@settings(suppress_health_check=suppressed_list)
def permute_indices_ref_(
        lengths: torch.Tensor,
        indices: torch.Tensor,
        weights: Optional[torch.Tensor],
        permute: torch.LongTensor,
        is_1D: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    T = lengths.size(0)
    B = lengths.size(1)
    if T == 0 or B == 0:
        if is_1D:
            lengths = lengths.view(-1)
        return lengths, indices, weights

    if is_1D:
        permuted_lengths = torch.index_select(lengths.view(-1), 0, permute).view(-1)
        original_segment_lengths = lengths.view(-1)
        original_segment_start = [0] + list(accumulate(lengths.view(-1)))

        permuted_indices = []
        permuted_weights = []
        for i in range(permute.numel()):
            start = original_segment_start[permute[i]]
            end = start + original_segment_lengths[permute[i]]
            permuted_indices.append(indices[start:end])
            if weights is not None:
                permuted_weights.append(weights[start:end])

        permuted_indices = torch.cat(permuted_indices, dim=0).flatten()

        if weights is None:
            permuted_weights = None
        else:
            permuted_weights = torch.cat(permuted_weights, dim=0).flatten()
    else:
        permuted_lengths = torch.index_select(lengths.view(T, -1), 0, permute)
        original_segment_lengths = lengths.view(T, -1).sum(dim=1, dtype=torch.int32)
        original_segment_start = [0] + list(
            accumulate(original_segment_lengths.view(-1))
        )

        permuted_indices = []
        permuted_weights = []
        for i in range(permute.size(0)):
            start = original_segment_start[permute[i]]
            end = start + original_segment_lengths[permute[i]]
            permuted_indices.append(indices[start:end])
            if weights is not None:
                permuted_weights.append(weights[start:end])

        permuted_indices = torch.cat(permuted_indices, dim=0).flatten()

        if weights is None:
            permuted_weights = None
        else:
            permuted_weights = torch.cat(permuted_weights, dim=0).flatten()

    return permuted_lengths, permuted_indices, permuted_weights


@torch.jit.script
def permute_scripted(
        permute: torch.Tensor, lengths: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    (
        permuted_lengths_cpu,
        permuted_indices_cpu,
        permuted_weights_cpu,
    ) = torch.ops.fbgemm.permute_2D_sparse_data(permute, lengths, indices, None, None)
    return (
        permuted_lengths_cpu,
        permuted_indices_cpu,
        permuted_weights_cpu,
    )


class PermuteIndicesTest(unittest.TestCase):
    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        long_index=st.booleans(),
        has_weight=st.booleans(),
        is_1D=st.booleans(),
        W=st.integers(min_value=4, max_value=8),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_permute_indices(
        self,
        B: int,
        T: int,
        L: int,
        long_index: bool,
        has_weight: bool,
        is_1D: bool,
        W: int,
    ) -> None:
        index_dtype = torch.int64 if long_index else torch.int32
        length_splits: Optional[list[torch.Tensor]] = None
        if is_1D:
            if B == 0:
                batch_sizes = [0] * W
            else:
                batch_sizes = [random.randint(a=1, b=B) for i in range(W)]
            length_splits = [
                torch.randint(low=1, high=L, size=(T, batch_sizes[i])).type(index_dtype)
                for i in range(W)
            ]
            lengths = torch.cat(length_splits, dim=1)
        else:
            lengths = torch.randint(low=1, high=L, size=(T, B)).type(index_dtype)
        #  typing.Tuple[int, ...]]` but got `Union[bool, float, int]`.
        weights = torch.rand(lengths.sum().item()).float() if has_weight else None
        indices = torch.randint(
            low=1,
            high=int(1e5),
            #  param but got `Tuple[typing.Union[float, int]]`.
            size=(lengths.sum().item(),),
        ).type(index_dtype)
        if is_1D:
            permute_list = []
            offset_w = [0] + list(
                accumulate([length_split.numel() for length_split in length_splits])
            )
            for t in range(T):
                for w in range(W):
                    for b in range(batch_sizes[w]):
                        permute_list.append(offset_w[w] + t * batch_sizes[w] + b)
        else:
            permute_list = list(range(T))
            random.shuffle(permute_list)

        permute = torch.IntTensor(permute_list)

        if is_1D:
            (
                permuted_lengths_cpu,
                permuted_indices_cpu,
                permuted_weights_cpu,
            ) = torch.ops.fbgemm.permute_1D_sparse_data(
                permute, lengths, indices, weights, None
            )
        else:
            (
                permuted_lengths_cpu,
                permuted_indices_cpu,
                permuted_weights_cpu,
            ) = torch.ops.fbgemm.permute_2D_sparse_data(
                permute, lengths, indices, weights, None
            )
        (
            permuted_lengths_ref,
            permuted_indices_ref,
            permuted_weights_ref,
        ) = permute_indices_ref_(lengths, indices, weights, permute.long(), is_1D)
        torch.testing.assert_close(permuted_indices_cpu, permuted_indices_ref)
        torch.testing.assert_close(permuted_lengths_cpu, permuted_lengths_ref)
        if has_weight:
            torch.testing.assert_close(permuted_weights_cpu, permuted_weights_ref)
        else:
            assert permuted_weights_cpu is None and permuted_weights_ref is None

        if npu_available():
            weights_npu = (
                weights.npu() if (has_weight and weights is not None) else None
            )
            if is_1D:
                (
                    permuted_lengths_npu,
                    permuted_indices_npu,
                    permuted_weights_npu,
                ) = torch.ops.fbgemm.permute_1D_sparse_data(
                    permute.npu(),
                    lengths.flatten().npu(),
                    indices.npu(),
                    weights_npu,
                    None,
                )
            else:
                (
                    permuted_lengths_npu,
                    permuted_indices_npu,
                    permuted_weights_npu,
                ) = torch.ops.fbgemm.permute_2D_sparse_data(
                    permute.npu(),
                    lengths.npu(),
                    indices.npu(),
                    weights_npu,
                    None,
                )
            torch.testing.assert_close(permuted_indices_npu.cpu(), permuted_indices_cpu)
            torch.testing.assert_close(permuted_lengths_npu.cpu(), permuted_lengths_cpu)
            if has_weight:
                torch.testing.assert_close(
                    permuted_weights_npu.cpu(), permuted_weights_cpu
                )
            else:
                self.assertIsNone(permuted_weights_npu)

    @given(
        B=st.integers(min_value=2, max_value=20),
        T=st.integers(min_value=2, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        long_index=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    @unittest.skipIf(*npu_unavailable)
    def test_permute_indices_non_contiguous(
        self,
        B: int,
        T: int,
        L: int,
        long_index: bool,
    ) -> None:
        index_dtype = torch.int64 if long_index else torch.int32
        lengths = torch.randint(low=1, high=L, size=(T, B)).type(index_dtype)

        indices = torch.randint(
            low=1,
            high=int(1e5),
            #  param but got `Tuple[typing.Union[float, int]]`.
            size=(lengths.sum().item(),),
        ).type(index_dtype)

        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        def create_non_contiguous(x: torch.Tensor) -> torch.Tensor:
            # Create a diluted tensor with 2x elements, and then take every other element
            # with the value from the original tensor. For example, if x = [1, 2, 3, 4],
            # then the diluted tensor is [1, 0, 2, 0, 3, 0, 4, 0].
            diluted = x.new_zeros(x.numel() * 2).flatten()
            diluted[::2] = x.flatten()
            # Returns the sliced tensor, which is non-contiguous.
            return diluted[::2].view(x.shape)

        (
            permuted_lengths_ref,
            permuted_indices_ref,
            permuted_weights_ref
        ) = permute_indices_ref_(lengths, indices, None, permute.long())

        permute_npu = create_non_contiguous(permute.npu())
        lengths_npu = create_non_contiguous(lengths.npu())
        indices_npu = create_non_contiguous(indices.npu())
        self.assertFalse(permute_npu.is_contiguous())
        self.assertFalse(lengths_npu.is_contiguous())
        self.assertFalse(indices_npu.is_contiguous())

        (
            permuted_lengths_npu,
            permuted_indices_npu,
            permuted_weights_npu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(
            permute_npu,
            lengths_npu,
            indices_npu,
            None,
            None,
        )
        torch.testing.assert_close(permuted_indices_npu.cpu(), permuted_indices_ref)
        torch.testing.assert_close(permuted_lengths_npu.cpu(), permuted_lengths_ref)
        self.assertIsNone(permuted_weights_npu)

    # TorchScript has different behaviors than eager mode. We can see undefined
    # models returned. So we need to add a unittest to ensure the op return
    # real None, not an undefined tensor.
    @unittest.skipIf(*npu_unavailable)
    def test_permute_indices_scripted_with_none_weights(
        self,
    ) -> None:
        index_dtype = torch.int32
        lengths = torch.randint(low=1, high=2, size=(1, 1)).type(index_dtype)
        weights = None
        indices = torch.randint(
            low=1,
            high=int(1e5),
            #  param but got `Tuple[typing.Union[float, int]]`.
            size=(lengths.sum().item(),),
        ).type(index_dtype)
        permute_list = list(range(1))
        random.shuffle(permute_list)

        permute = torch.IntTensor(permute_list)

        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = permute_scripted(permute, lengths, indices)
        (
            permuted_lengths_ref,
            permuted_indices_ref,
            permuted_weights_ref,
        ) = permute_indices_ref_(lengths, indices, weights, permute.long(), False)
        self.assertTrue(torch.equal(permuted_indices_cpu, permuted_indices_ref))
        self.assertTrue(torch.equal(permuted_lengths_cpu, permuted_lengths_ref))
        self.assertEqual(permuted_weights_cpu, None)
        self.assertEqual(permuted_weights_ref, None)

    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        long_index=st.booleans(),
        has_weight=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_permute_indices_with_repeats(
        self, B: int, T: int, L: int, long_index: bool, has_weight: bool
    ) -> None:
        index_dtype = torch.int64 if long_index else torch.int32
        lengths = torch.randint(low=1, high=L, size=(T, B)).type(index_dtype)
        #  typing.Tuple[int, ...]]` but got `Union[bool, float, int]`.
        weights = torch.rand(lengths.sum().item()).float() if has_weight else None
        indices = torch.randint(
            low=1,
            high=int(1e5),
            #  param but got `Tuple[typing.Union[float, int]]`.
            size=(lengths.sum().item(),),
        ).type(index_dtype)
        permute_list = list(range(T))

        num_repeats = random.randint(0, T)
        for _ in range(num_repeats):
            permute_list.append(random.randint(0, T - 1))

        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = torch.ops.fbgemm.permute_2D_sparse_data(permute, lengths, indices, weights)
        (
            permuted_lengths_ref,
            permuted_indices_ref,
            permuted_weights_ref,
        ) = permute_indices_ref_(lengths, indices, weights, permute.long())
        torch.testing.assert_close(permuted_indices_cpu, permuted_indices_ref)
        torch.testing.assert_close(permuted_lengths_cpu, permuted_lengths_ref)
        if has_weight:
            torch.testing.assert_close(permuted_weights_cpu, permuted_weights_ref)
        else:
            assert permuted_weights_cpu is None and permuted_weights_ref is None

        if npu_available():
            weights_npu = (
                weights.npu() if (has_weight and weights is not None) else None
            )
            (
                permuted_lengths_npu,
                permuted_indices_npu,
                permuted_weights_npu,
            ) = torch.ops.fbgemm.permute_2D_sparse_data(
                permute.npu(),
                lengths.npu(),
                indices.npu(),
                weights_npu,
            )
            torch.testing.assert_close(permuted_indices_npu.cpu(), permuted_indices_cpu)
            torch.testing.assert_close(permuted_lengths_npu.cpu(), permuted_lengths_cpu)
            if has_weight:
                torch.testing.assert_close(
                    permuted_weights_npu.cpu(), permuted_weights_cpu
                )
            else:
                assert permuted_weights_cpu is None

    @given(
        num_segments=st.integers(min_value=20, max_value=100),
        max_segment_length=st.integers(min_value=100, max_value=1000),
        index_dtype=st.sampled_from([torch.int32, torch.int64, torch.float32]),
        has_weight=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=100, deadline=None)
    @unittest.skipIf(*npu_unavailable)
    def test_permute_1D_sparse_data_vec(
        self,
        num_segments: int,
        max_segment_length: int,
        index_dtype: torch.dtype,
        has_weight: bool,
    ) -> None:
        """
        Test vectorized permute_1D_sparse_data kernel with vec4 optimization.

        Validates:
        - Correctness for various segment lengths (tests vec4 path and remainder handling)
        - Alignment-based vectorization (vec4 when aligned, scalar fallback when misaligned)
        - With and without weights (tests weights_vec4_aligned short-circuit logic)
        - Different index types (float vs int64)
        - Edge cases: segment lengths at vec4 boundaries (1, 3, 4, 5, 8, 15, 16, etc.)
        """

        # Generate variable-length segments to test vectorization
        lengths = torch.randint(
            low=max_segment_length // 2,
            high=max_segment_length,
            size=(num_segments,),
            dtype=torch.int32,
        )
        total_indices = int(lengths.sum().item())
        # Generate indices
        if index_dtype == torch.float32:
            indices = torch.rand(total_indices, dtype=index_dtype)
        else:
            indices = torch.randint(
                low=0,
                high=2**31 - 1,
                size=(total_indices,),
                dtype=index_dtype,
            )

        # Generate optional weights
        weights = torch.rand(total_indices, dtype=torch.float32) if has_weight else None

        # Generate random permutation
        permute_list = list(range(num_segments))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        # CPU reference (uses scalar kernel)
        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = torch.ops.fbgemm.permute_1D_sparse_data(
            permute, lengths, indices, weights, None
        )

        # NPU vectorized kernel (uses vec4 when aligned)
        (
            permuted_lengths_npu,
            permuted_indices_npu,
            permuted_weights_npu,
        ) = torch.ops.fbgemm.permute_1D_sparse_data(
            permute.npu(),
            lengths.npu(),
            indices.npu(),
            weights.npu() if has_weight and weights is not None else None,
            None,
        )

        # Validate correctness
        torch.testing.assert_close(
            permuted_lengths_npu.cpu(),
            permuted_lengths_cpu,
        )
        torch.testing.assert_close(
            permuted_indices_npu.cpu(),
            permuted_indices_cpu,
        )

        if has_weight:
            torch.testing.assert_close(
                permuted_weights_npu.cpu(),
                permuted_weights_cpu,
            )
        else:
            self.assertIsNone(permuted_weights_npu)
            self.assertIsNone(permuted_weights_cpu)

        # Test edge cases with specific segment lengths at vec4 boundaries
        # This validates remainder handling (segment_length % 4 = 0, 1, 2, 3)
        edge_case_lengths = [1, 3, 4, 5, 15, 16, 17, 63, 64, 127, 128]
        for segment_length in edge_case_lengths:
            lengths_edge = torch.tensor([segment_length], dtype=torch.int32)
            if index_dtype == torch.float32:
                indices_edge = torch.rand(segment_length, dtype=index_dtype)
            else:
                indices_edge = torch.randint(
                    0, 2**31 - 1, size=(segment_length,), dtype=index_dtype
                )
            weights_edge = (
                torch.rand(segment_length, dtype=torch.float32) if has_weight else None
            )
            permute_edge = torch.IntTensor([0])

            (
                permuted_lengths_cpu_edge,
                permuted_indices_cpu_edge,
                permuted_weights_cpu_edge,
            ) = torch.ops.fbgemm.permute_1D_sparse_data(
                permute_edge, lengths_edge, indices_edge, weights_edge, None
            )

            weights_edge_npu = (
                weights_edge.npu()
                if (has_weight and weights_edge is not None)
                else None
            )
            (
                permuted_lengths_npu_edge,
                permuted_indices_npu_edge,
                permuted_weights_npu_edge,
            ) = torch.ops.fbgemm.permute_1D_sparse_data(
                permute_edge.npu(),
                lengths_edge.npu(),
                indices_edge.npu(),
                weights_edge_npu,
                None,
            )
            torch.testing.assert_close(
                permuted_lengths_npu_edge.cpu(),
                permuted_lengths_cpu_edge,
            )
            torch.testing.assert_close(
                permuted_indices_npu_edge.cpu(),
                permuted_indices_cpu_edge,
            )

            if has_weight:
                torch.testing.assert_close(
                    permuted_weights_npu_edge.cpu(),
                    permuted_weights_cpu_edge,
                )


if __name__ == "__main__":
    unittest.main()
