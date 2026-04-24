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
from common import cpu_and_maybe_npu


def _is_npu(device: torch.device) -> bool:
    return str(device).startswith("npu")


class CumSumTest(unittest.TestCase):
    @given(
        n=st.integers(min_value=0, max_value=10),
        index_types=st.sampled_from(
            [
                (torch.int64, np.int64),
                (torch.int32, np.int32),
                (torch.float32, np.float32),
            ]
        ),
        device=cpu_and_maybe_npu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_cumsum(
            self,
            n: int,
            index_types: tuple[type[object], type[object]],
            device: torch.device,
    ) -> None:
        (pt_index_dtype, np_index_dtype) = index_types

        # The CPU variants of asynchronous_*_cumsum support floats, since some
        # downstream tests appear to be relying on this behavior.  As such, the
        # test is disabled for NPU + float test cases.
        if _is_npu(device) and pt_index_dtype is torch.float32:
            return

        x = torch.randint(low=0, high=100, size=(n,)).type(pt_index_dtype).to(device)
        ze = torch.ops.fbgemm.asynchronous_exclusive_cumsum(x)
        zi = torch.ops.fbgemm.asynchronous_inclusive_cumsum(x)
        zc = torch.ops.fbgemm.asynchronous_complete_cumsum(x)

        torch.testing.assert_close(
            torch.from_numpy(np.cumsum(x.cpu().numpy()).astype(np_index_dtype)),
            zi.cpu(),
        )
        torch.testing.assert_close(
            torch.from_numpy(
                (np.cumsum([0] + x.cpu().numpy().tolist())[:-1]).astype(np_index_dtype)
            ),
            ze.cpu(),
        )
        torch.testing.assert_close(
            torch.from_numpy(
                (np.cumsum([0] + x.cpu().numpy().tolist())).astype(np_index_dtype)
            ),
            zc.cpu(),
        )

        mx = torch.randint(low=0, high=100, size=(n,)).type(pt_index_dtype).to("meta")

        mze = torch.ops.fbgemm.asynchronous_exclusive_cumsum(mx)
        self.assertEqual(ze.size(), mze.size())

        mzi = torch.ops.fbgemm.asynchronous_inclusive_cumsum(mx)
        self.assertEqual(zi.size(), mzi.size())

        mzc = torch.ops.fbgemm.asynchronous_complete_cumsum(mx)
        self.assertEqual(zc.size(), mzc.size())

    @given(
        n=st.integers(min_value=0, max_value=60),
        b=st.integers(min_value=0, max_value=10),
        index_types=st.sampled_from(
            [
                (torch.int64, np.int64),
                (torch.int32, np.int32),
                (torch.float32, np.float32),
            ]
        ),
        device=cpu_and_maybe_npu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_asynchronous_complete_cumsum_2d(
            self,
            n: int,
            b: int,
            index_types: tuple[type[object], type[object]],
            device: torch.device,
    ) -> None:
        (pt_index_dtype, np_index_dtype) = index_types

        # The CPU variants of asynchronous_*_cumsum support floats, since some
        # downstream tests appear to be relying on this behavior.  As such, the
        # test is disabled for NPU + float test cases.
        if _is_npu(device) and pt_index_dtype is torch.float32:
            return

        # NPU impl currently only supports 1D input;
        if _is_npu(device):
            return

        x = torch.randint(low=0, high=100, size=(b, n)).type(pt_index_dtype).to(device)

        zc = torch.ops.fbgemm.asynchronous_complete_cumsum(x)
        zeros = torch.zeros(b, 1)
        torch.testing.assert_close(
            torch.from_numpy(
                np.cumsum(torch.concat([zeros, x.cpu()], dim=1).numpy(), axis=1).astype(
                    np_index_dtype
                )
            ),
            zc.cpu(),
        )


if __name__ == "__main__":
    unittest.main()
