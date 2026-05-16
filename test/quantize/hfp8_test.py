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
import torch
from hypothesis import given, HealthCheck, settings
from torch import Tensor

import fbgemm_gpu  # noqa: F401
import fbgemm_ascend  # noqa: F401
from common import npu_available


class TestHFP8QuantizationConversion(unittest.TestCase):
    # min_pos is the minimal of denormal numbers
    # min_normal_pos is the minimal of normal numbers
    def _get_hfp8_dynamic_range(self, ebits: int, mbits: int, bias: int) -> tuple[float, float, float]:
        max_pos = (1 << ((1 << ebits) - 2 - bias)) * (2 - 2 ** (-mbits))
        min_pos = 2 ** (1 - bias - mbits)
        min_normal_pos = 2 ** (1 - bias)
        return min_pos, max_pos, min_normal_pos

    def _get_hfp8_config(
        self,
    ) -> tuple[int, int, dict[int, float], dict[int, float], dict[int, float]]:
        ebits = 4
        mbits = 3
        max_pos_dict = {}
        min_pos_dict = {}
        min_normal_pos_dict = {}
        for bias in [4, 5, 6, 7]:
            min_pos, max_pos, min_normal_pos = self._get_hfp8_dynamic_range(ebits, mbits, bias)
            min_pos_dict[bias] = min_pos
            max_pos_dict[bias] = max_pos
            min_normal_pos_dict[bias] = min_normal_pos

        return ebits, mbits, min_pos_dict, max_pos_dict, min_normal_pos_dict

    def _test_conversion(
        self,
        input_data: Tensor,
        reference_data: Tensor,
        ebits: int,
        exponent_bias: int,
        max_pos: float,
        atol: float = 0.0,
        rtol: float = 1e-7,
    ) -> None:
        quantized_data_cpu = torch.ops.fbgemm.FloatToHFP8Quantized(input_data, ebits, exponent_bias, max_pos)
        dequantized_data_cpu = torch.ops.fbgemm.HFP8QuantizedToFloat(quantized_data_cpu, ebits, exponent_bias)
        torch.testing.assert_close(dequantized_data_cpu, reference_data, rtol=rtol, atol=atol)

        if npu_available():
            quantized_data_npu = quantized_data_cpu.npu()
            dequantized_data_npu = torch.ops.fbgemm.HFP8QuantizedToFloat(quantized_data_npu, ebits, exponent_bias)
            torch.testing.assert_close(dequantized_data_npu.cpu(), reference_data, rtol=rtol, atol=atol)

    # pyre-ignore [56]
    @given(
        nrows=st.integers(min_value=1, max_value=100),
        ncols=st.integers(min_value=1, max_value=100),
        exponent_bias=st.integers(min_value=4, max_value=7),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_and_dequantize_op(self, nrows: int, ncols: int, exponent_bias: int) -> None:
        ebits, mbits, min_pos, max_pos, min_normal_pos = self._get_hfp8_config()
        # test positive normal range
        input_data = torch.FloatTensor(nrows, ncols).uniform_(min_normal_pos[exponent_bias], max_pos[exponent_bias])

        self._test_conversion(
            input_data,
            input_data,
            ebits,
            exponent_bias,
            max_pos[exponent_bias],
            rtol=(2 ** (-mbits - 1)),
            atol=0,
        )

        # test positive denormal range
        input_data = torch.FloatTensor(nrows, ncols).uniform_(min_pos[exponent_bias], min_normal_pos[exponent_bias])
        self._test_conversion(
            input_data,
            input_data,
            ebits,
            exponent_bias,
            max_pos[exponent_bias],
            rtol=0.0,
            atol=(2 ** (1 - exponent_bias - mbits)),
        )

        # test negative normal range
        input_data = torch.FloatTensor(nrows, ncols).uniform_(-max_pos[exponent_bias], -min_normal_pos[exponent_bias])
        self._test_conversion(
            input_data,
            input_data,
            ebits,
            exponent_bias,
            max_pos[exponent_bias],
            rtol=(2 ** (-mbits - 1)),
            atol=0,
        )

        # test negative denormal range
        input_data = torch.FloatTensor(nrows, ncols).uniform_(-min_normal_pos[exponent_bias], -min_pos[exponent_bias])
        self._test_conversion(
            input_data,
            input_data,
            ebits,
            exponent_bias,
            max_pos[exponent_bias],
            rtol=0.0,
            atol=(2 ** (1 - exponent_bias - mbits)),
        )

        # test positive underflow
        input_data = torch.FloatTensor(nrows, ncols).uniform_(0, 0.5 * min_pos[exponent_bias])
        self._test_conversion(
            input_data,
            input_data.new_full(input_data.shape, 0),
            ebits,
            exponent_bias,
            max_pos[exponent_bias],
        )

        # test negative underflow
        input_data = torch.FloatTensor(nrows, ncols).uniform_(-0.5 * min_pos[exponent_bias], 0)
        self._test_conversion(
            input_data,
            input_data.new_full(input_data.shape, 0),
            ebits,
            exponent_bias,
            max_pos[exponent_bias],
        )

        # test positive overflow
        input_data = torch.FloatTensor(nrows, ncols).uniform_(max_pos[exponent_bias], max_pos[exponent_bias] * 2)
        self._test_conversion(
            input_data,
            input_data.new_full(input_data.shape, max_pos[exponent_bias]),
            ebits,
            exponent_bias,
            max_pos[exponent_bias],
        )

        # test negative overflow
        input_data = torch.FloatTensor(nrows, ncols).uniform_(-max_pos[exponent_bias] * 2, -max_pos[exponent_bias])
        self._test_conversion(
            input_data,
            input_data.new_full(input_data.shape, -max_pos[exponent_bias]),
            ebits,
            exponent_bias,
            max_pos[exponent_bias],
        )


if __name__ == "__main__":
    unittest.main()
