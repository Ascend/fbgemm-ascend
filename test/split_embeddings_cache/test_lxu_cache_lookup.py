#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import torch
import hypothesis.strategies as st
from hypothesis import given, settings
import torch_npu
import fbgemm_ascend  # noqa: F401


DEFAULT_ASSOC = 32

DEVICE = "npu:0"


# 检查 NPU 是否可用
def npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available() and torch.npu.device_count() > 0


if npu_available():
    torch_npu.npu.set_device(DEVICE)

# Used for `@unittest.skipIf`
npu_unavailable: tuple[bool, str] = (
    not npu_available(),
    "NPU is not available or no NPUs detected",
)


class LXUCacheLookupTest(unittest.TestCase):
    @unittest.skipIf(*npu_unavailable)
    @given(
        associativity=st.sampled_from([1, DEFAULT_ASSOC]),
    )
    @settings(deadline=None)
    def test_lxu_cache_lookup(self, associativity: int):
        """Test: lxu_cache_lookup with different associativity values.

        与 GPU 保持一致：使用 hypothesis 参数化，测试 associativity=1 和 32。
        """
        max_index = 8000

        # Use single cache set to avoid dealing with cache set hash algorithm.
        lxu_cache_state = torch.arange(associativity, dtype=torch.int64, device=DEVICE).unsqueeze(0)

        # Testing all miss (与 GPU 保持一致：根据 associativity 选择不同索引)
        linear_cache_indices_0 = (
            torch.tensor([32, 33, 34, 35, 36, 100, 1000, 1725], dtype=torch.int64, device=DEVICE)
            if associativity <= 32
            else torch.tensor([64, 65, 66, 67, 68, 100, 1000, 1725], dtype=torch.int64, device=DEVICE)
        )
        lxu_locations = torch.ops.fbgemm.lxu_cache_lookup(linear_cache_indices_0, lxu_cache_state, max_index)
        torch.testing.assert_close(
            lxu_locations,
            torch.full_like(lxu_locations, -1),
        )

        # Testing all hits (与 GPU 保持一致)
        cache_indices_1 = torch.randint(0, associativity, (associativity,))
        cache_indices_1_npu = cache_indices_1.to(DEVICE)
        lxu_locations = torch.ops.fbgemm.lxu_cache_lookup(cache_indices_1_npu, lxu_cache_state, max_index)
        torch.testing.assert_close(
            lxu_locations.cpu(),
            cache_indices_1.int(),
        )

        # Testing mixture (与 GPU 保持一致)
        miss_cache_indices_0 = torch.randint(associativity, max_index // 2, (10,))
        hit_cache_indices_0 = torch.randint(0, associativity, (8,))
        miss_cache_indices_1 = torch.randint(max_index // 2, max_index, (16,))
        hit_cache_indices_1 = torch.randint(0, associativity, (8,))
        linear_cache_indices_2 = torch.cat(
            [
                miss_cache_indices_0,
                hit_cache_indices_0,
                miss_cache_indices_1,
                hit_cache_indices_1,
            ]
        ).to(DEVICE)
        lxu_locations = torch.ops.fbgemm.lxu_cache_lookup(linear_cache_indices_2, lxu_cache_state, max_index)
        expected_result = torch.cat(
            [
                torch.full_like(miss_cache_indices_0, -1),
                hit_cache_indices_0,
                torch.full_like(miss_cache_indices_1, -1),
                hit_cache_indices_1,
            ]
        ).int()
        torch.testing.assert_close(
            lxu_locations.cpu(),
            expected_result,
        )


if __name__ == "__main__":
    unittest.main()
