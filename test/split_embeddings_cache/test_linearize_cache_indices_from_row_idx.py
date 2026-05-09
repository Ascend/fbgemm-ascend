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

import torch

# NPU 环境需要保留下列 import（勿删）。
from torch_npu.contrib import transfer_to_npu
import fbgemm_ascend

DEVICE = "npu:0"


class LinearizeCacheIndicesFromRowIdxTest(unittest.TestCase):
    def test_linearize_cache_indices_from_row_idx(self) -> None:
        update_row_indices = torch.tensor(
            [10, 2, 3, 7, 1, 4, 5, 9, 2, 7, 6, 8, 5, 1, 0, 4],
            dtype=torch.int,
            device=DEVICE,
        )
        update_table_indices = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            dtype=torch.int,
            device=DEVICE,
        )
        varying_update_table_indices = torch.tensor(
            [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3],
            dtype=torch.int,
            device=DEVICE,
        )

        # Testing equal sized tables.
        cache_hash_size_cumsum_0 = torch.tensor([0, 12, 24, 36, 48]).to(DEVICE)
        linear_cache_indices_0 = torch.ops.fbgemm.linearize_cache_indices_from_row_idx(
            cache_hash_size_cumsum_0,
            update_table_indices,
            update_row_indices,
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_0.cpu(),
                torch.tensor(
                    [10, 2, 3, 7, 13, 16, 17, 21, 26, 31, 30, 32, 41, 37, 36, 40],
                    dtype=torch.int,
                ),
            )
        )

        # Testing partially cached tables.
        cache_hash_size_cumsum_1 = torch.tensor([0, 12, -1, 24, 36]).to(DEVICE)
        linear_cache_indices_1 = torch.ops.fbgemm.linearize_cache_indices_from_row_idx(
            cache_hash_size_cumsum_1,
            update_table_indices,
            update_row_indices,
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_1.cpu(),
                torch.tensor(
                    [10, 2, 3, 7, 13, 16, 17, 21, 36, 36, 36, 36, 29, 25, 24, 28],
                    dtype=torch.int,
                ),
            )
        )

        # Testing batched with varying pooling factor.
        cache_hash_size_cumsum_2 = torch.tensor([0, 12, -1, 24, 36]).to(DEVICE)
        linear_cache_indices_2 = torch.ops.fbgemm.linearize_cache_indices_from_row_idx(
            cache_hash_size_cumsum_2,
            varying_update_table_indices,
            update_row_indices,
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_2.cpu(),
                torch.tensor(
                    [10, 2, 3, 19, 13, 16, 17, 21, 36, 36, 36, 36, 36, 36, 24, 28],
                    dtype=torch.int,
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
