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
from hypothesis import given, settings, Verbosity

import fbgemm_gpu
import fbgemm_ascend
from common import npu_available


class MiscOpsTest(unittest.TestCase):
    @given(
        N=st.integers(min_value=1, max_value=20),
        offsets_type=st.sampled_from([torch.int32, torch.int64]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_offsets_range(
            self,
            offsets_type: torch.dtype,
            N: int,
    ) -> None:
        lengths = np.array([np.random.randint(low=0, high=20) for _ in range(N)])
        offsets = np.cumsum(np.concatenate([[0], lengths]))[:-1]
        range_ref = torch.from_numpy(
            np.concatenate([np.arange(size) for size in lengths])
        )
        output_size = np.sum(lengths)

        offsets_cpu = torch.tensor(offsets, dtype=offsets_type)
        range_cpu = torch.ops.fbgemm.offsets_range(offsets_cpu, output_size)
        range_ref = range_ref.to(range_cpu.dtype)
        torch.testing.assert_close(range_cpu, range_ref, rtol=0, atol=0)

        if npu_available():
            range_npu = torch.ops.fbgemm.offsets_range(offsets_cpu.npu(), output_size)
            range_ref = range_ref.to(range_npu.dtype)
            torch.testing.assert_close(range_npu.cpu(), range_ref, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
