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
from hypothesis import assume, given, HealthCheck, settings

import fbgemm_gpu
import fbgemm_ascend
from common import npu_available, bytes_to_half_floats, fused_rowwise_nbit_quantize_reference


class TestFusedNBitRowwiseQuantizationConversion(unittest.TestCase):
    @given(
        nrows=st.integers(min_value=0, max_value=100),
        ncols=st.integers(min_value=0, max_value=100),
        bit_rate=st.sampled_from([2, 4]),
        is_half=st.booleans(),
        test_float_or_half_op=st.booleans(),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_op(
            self,
            nrows: int,
            ncols: int,
            bit_rate: int,
            is_half: bool,
            test_float_or_half_op: bool,
    ) -> None:
        assert 8 % bit_rate == 0
        num_elem_per_byte = 8 // bit_rate
        assume(ncols % (2 * num_elem_per_byte) == 0)

        input_data = torch.rand(nrows, ncols).float()
        if is_half:
            input_data = input_data.half()

        if test_float_or_half_op:
            quantized_data = (
                torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
                    input_data, bit_rate
                )
            )
        else:
            if not is_half:
                quantized_data = (
                    torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                        input_data, bit_rate
                    )
                )
            else:
                quantized_data = torch.ops.fbgemm.HalfToFusedNBitRowwiseQuantizedSBHalf(
                    input_data, bit_rate
                )
        if nrows == 0 or ncols == 0:
            assert quantized_data.numel() == nrows * (
                    (ncols + bit_rate - 1) // bit_rate + 4
            )
            return

        quantized_data = quantized_data.numpy()

        reference = fused_rowwise_nbit_quantize_reference(
            input_data.float().numpy(), bit_rate
        )

        interleaved_dim = ncols // num_elem_per_byte
        # compare quantized data
        np.testing.assert_array_equal(
            quantized_data[:, :interleaved_dim], reference[:, :interleaved_dim]
        )
        # compare scales
        np.testing.assert_array_almost_equal(
            bytes_to_half_floats(
                quantized_data[:, interleaved_dim: interleaved_dim + 2]
            ),
            bytes_to_half_floats(reference[:, interleaved_dim: interleaved_dim + 2]),
        )
        # compare zero points (bias)
        np.testing.assert_array_equal(
            quantized_data[:, interleaved_dim + 2], reference[:, interleaved_dim + 2]
        )

        if npu_available():
            input_data_npu = input_data.npu()
            if test_float_or_half_op:
                quantized_data_npu = (
                    torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
                        input_data_npu, bit_rate
                    )
                )
            else:
                if not is_half:
                    quantized_data_npu = (
                        torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                            input_data_npu, bit_rate
                        )
                    )
                else:
                    quantized_data_npu = (
                        torch.ops.fbgemm.HalfToFusedNBitRowwiseQuantizedSBHalf(
                            input_data_npu, bit_rate
                        )
                    )
            quantized_data_numpy = quantized_data_npu.cpu().numpy()
            # compare quantized data
            np.testing.assert_array_equal(
                quantized_data_numpy[:, :interleaved_dim], reference[:, :interleaved_dim]
            )
            # compare scales
            np.testing.assert_array_almost_equal(
                bytes_to_half_floats(
                    quantized_data_numpy[:, interleaved_dim: interleaved_dim + 2]
                ),
                bytes_to_half_floats(reference[:, interleaved_dim: interleaved_dim + 2]),
            )
            # compare zero points (bias)
            np.testing.assert_array_equal(
                quantized_data_numpy[:, interleaved_dim + 2], reference[:, interleaved_dim + 2]
            )

    @unittest.skipIf(not npu_available(), "Skip when NPU is not available")
    def test_quantize_op_npu_large_nrows(self) -> None:
        ncols = 256
        bit_rate = 4
        nrows = 65540

        num_elem_per_byte = 8 // bit_rate
        input_data = torch.rand(nrows, ncols).float()
        # ncols=256 is always divisible by 2*num_elem_per_byte for bit_rate=4

        reference = fused_rowwise_nbit_quantize_reference(
            input_data.numpy(), bit_rate
        )

        input_data_npu = input_data.npu()
        quantized_data_npu = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
            input_data_npu, bit_rate
        )
        quantized_data_numpy = quantized_data_npu.cpu().numpy()

        interleaved_dim = ncols // num_elem_per_byte
        np.testing.assert_allclose(
            quantized_data_numpy[:, :interleaved_dim],
            reference[:, :interleaved_dim],
            atol=1,
            rtol=0,
        )
        np.testing.assert_array_almost_equal(
            bytes_to_half_floats(
                quantized_data_numpy[:, interleaved_dim: interleaved_dim + 2]
            ),
            bytes_to_half_floats(reference[:, interleaved_dim: interleaved_dim + 2]),
        )
        np.testing.assert_array_equal(
            quantized_data_numpy[:, interleaved_dim + 2], reference[:, interleaved_dim + 2]
        )


if __name__ == "__main__":
    unittest.main()
