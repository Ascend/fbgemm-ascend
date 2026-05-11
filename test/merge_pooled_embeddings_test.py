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

import fbgemm_gpu
import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given, settings, Verbosity

import torch_npu
import fbgemm_ascend

from test_utils import npu_unavailable

typed_npu_unavailable: tuple[bool, str] = npu_unavailable
bytes2bit = 8

def make_pitched_tensor(
    height: int,
    width: int,
    dtype: torch.dtype,
    # pyre-fixme[2]: Parameter must be annotated.
    device,
    alignment: int = 256,
) -> torch.Tensor:
    elem_size = (
        torch.finfo(dtype).bits // bytes2bit
        if dtype.is_floating_point
        else torch.iinfo(dtype).bits // bytes2bit
    )
    width_bytes = width * elem_size
    pitch_bytes = int(np.ceil(width_bytes / alignment) * alignment)
    pitch_elems = pitch_bytes // elem_size
    storage = torch.randn((height, pitch_elems), dtype=dtype, device=device)
    view = storage[:, :width]  # logical shape
    return view.contiguous() if alignment == 0 else view  # return pitched view

@unittest.skipIf(*typed_npu_unavailable)
class MergePooledEmbeddingsTest(unittest.TestCase):
    @given(
        num_inputs=st.integers(min_value=1, max_value=10),
        num_npus=st.integers(min_value=1, max_value=torch.npu.device_count()),
        r=st.randoms(use_true_random=False),
        use_pitched=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=40, deadline=None)
    def test_all_to_one_device(
        self,
        num_inputs: int,
        num_npus: int,
        # pyre-fixme[2]: Parameter must be annotated.
        r,
        use_pitched: bool,
    ) -> None:
        dst_device = torch.device(f"npu:{r.randint(0, num_npus - 1)}")
        with torch.npu.device(dst_device):
            if use_pitched:
                inputs = [
                    make_pitched_tensor(10, 20, torch.float32, "cpu", alignment=256)
                    for _ in range(num_inputs)
                ]
            else:
                inputs = [torch.randn(10, 20) for _ in range(num_inputs)]

            npu_inputs = [
                input.to(f"npu:{i % num_npus}")
                for i, input in enumerate(inputs)
            ]
            npu_outputs = torch.ops.fbgemm.all_to_one_device(npu_inputs, dst_device)
            for i, o in zip(inputs, npu_outputs):
                self.assertEqual(o.device, dst_device)
                torch.testing.assert_close(o.cpu(), i)
