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
from hypothesis import given, HealthCheck, settings

import fbgemm_gpu  # noqa: F401
import fbgemm_ascend  # noqa: F401
from common import (
    npu_available,
    fused_rowwise_8bit_quantize_reference,
    fused_rowwise_8bit_dequantize_reference,
    fused_rowwise_8bit_dequantize_2bytes_padding_scale_bias_first_reference,
)


class TestFused8BitRowwiseQuantizationConversion(unittest.TestCase):
    @given(
        nrows=st.integers(min_value=0, max_value=100),
        ncols=st.integers(min_value=0, max_value=100),
        is_half=st.booleans(),
        test_float_or_half_op=st.booleans(),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_op(self, nrows, ncols, is_half, test_float_or_half_op):
        input_data = torch.rand(nrows, ncols).float()
        if is_half:
            input_data = torch.rand(nrows, ncols).half()

        if test_float_or_half_op:
            quantized_data = torch.ops.fbgemm.FloatOrHalfToFused8BitRowwiseQuantized(input_data)
        else:
            if not is_half:
                quantized_data = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_data)
            else:
                quantized_data = torch.ops.fbgemm.HalfToFused8BitRowwiseQuantized(input_data)

        if nrows == 0 or ncols == 0:
            assert quantized_data.numel() == nrows * ((ncols + 3) // 4 * 4 + 8)
            return

        reference = fused_rowwise_8bit_quantize_reference(input_data.float().numpy())
        np.testing.assert_array_almost_equal(quantized_data.numpy(), reference)

        if npu_available():
            input_data_npu = input_data.npu()
            if test_float_or_half_op:
                quantized_data_npu = torch.ops.fbgemm.FloatOrHalfToFused8BitRowwiseQuantized(input_data_npu)
            else:
                if not is_half:
                    quantized_data_npu = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_data_npu)
                else:
                    quantized_data_npu = torch.ops.fbgemm.HalfToFused8BitRowwiseQuantized(input_data_npu)
            quantized_data_numpy = quantized_data_npu.cpu().numpy()
            ncols_aligned = (ncols + 4 - 1) // 4 * 4
            # compare quantized data
            np.testing.assert_allclose(
                quantized_data_numpy[:, :ncols],
                reference[:, :ncols],
                atol=1,
            )
            # compare scales (allow 1-ulp float diff on NPU)
            np.testing.assert_allclose(
                quantized_data_numpy[:, ncols_aligned : ncols_aligned + 4],
                reference[:, ncols : ncols + 4],
                atol=1,
            )
            # compare zero points (bias) (allow 1-ulp float diff on NPU)
            np.testing.assert_allclose(
                quantized_data_numpy[:, ncols_aligned + 4 : ncols_aligned + 8],
                reference[:, ncols + 4 : ncols + 8],
                atol=1,
            )

    def quantize_and_dequantize_op_test_helper(
        self,
        nrows,
        ncols,
        output_dtype,
        quant_padding_float_type,
        test_generic_op,
        test_npu,
    ):
        input_data = torch.rand(nrows, ncols).float()
        if output_dtype == 1:
            input_data = input_data.half()
        elif output_dtype == 5:
            input_data = input_data.bfloat16()

        if not test_npu:
            # cpu path only supports bf16 dequantization
            if output_dtype == 5:
                input_data = input_data.float()
            if not test_generic_op and not quant_padding_float_type:
                return
            if not quant_padding_float_type and output_dtype == 0:
                return
            if test_generic_op:
                quantized_data_ref = torch.ops.fbgemm.FloatOrHalfToFused8BitRowwiseQuantized(input_data)
                # fbgemm weight 2byte storages are scale_bias first layout
                if quant_padding_float_type is False:
                    scale_bias_last = False
                    quant_pad = quantized_data_ref[:, -8:]
                    quant_data = quantized_data_ref[:, :-8]
                    quantized_data = torch.cat(
                        [
                            quant_pad.view(torch.float).to(torch.half).view(torch.uint8),
                            quant_data,
                        ],
                        dim=1,
                    )
                else:
                    scale_bias_last = True
                    quantized_data = quantized_data_ref
                dequantized_data = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloatOrHalf(
                    quantized_data,
                    output_dtype,
                    quant_padding_float_type=quant_padding_float_type,
                    scale_bias_last=scale_bias_last,
                )
            else:
                if output_dtype == 0:
                    quantized_data = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_data)
                    dequantized_data = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(quantized_data)
                elif output_dtype == 1:
                    quantized_data = torch.ops.fbgemm.HalfToFused8BitRowwiseQuantized(input_data)
                    dequantized_data = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(quantized_data)
                elif output_dtype == 5:
                    quantized_data = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_data)
                    dequantized_data = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToBfloat16(quantized_data)
                else:
                    raise NotImplementedError("Unsupported dtype")

            if nrows == 0 or ncols == 0:
                assert dequantized_data.numel() == 0
                return

            quantize_data_numpy = quantized_data.numpy()
            if quant_padding_float_type:
                reference = torch.from_numpy(fused_rowwise_8bit_dequantize_reference(quantize_data_numpy))
            else:
                reference = torch.from_numpy(
                    fused_rowwise_8bit_dequantize_2bytes_padding_scale_bias_first_reference(quantize_data_numpy)
                )
            if output_dtype == 0:
                torch.testing.assert_close(dequantized_data.float(), reference.float())
            elif output_dtype == 1:
                torch.testing.assert_close(dequantized_data.half(), reference.half())
            elif output_dtype == 5:
                torch.testing.assert_close(dequantized_data.bfloat16(), reference.bfloat16())

        if test_npu and npu_available():
            if nrows == 0 or ncols == 0:
                return
            input_data_npu = input_data.npu()
            if not test_generic_op and not quant_padding_float_type:
                return
            if not quant_padding_float_type and output_dtype == 0:
                return
            if test_generic_op:
                quantized_data_npu_ref = torch.ops.fbgemm.FloatOrHalfToFused8BitRowwiseQuantized(input_data_npu)
                # fbgemm weight 2byte storages are scale_bias first layout
                if quant_padding_float_type is False:
                    scale_bias_last = False
                    quant_pad = quantized_data_npu_ref[:, -8:]
                    quant_data = quantized_data_npu_ref[:, :-8]
                    quantized_data_npu = torch.cat(
                        [
                            quant_pad.view(torch.float).to(torch.half).view(torch.uint8),
                            quant_data,
                        ],
                        dim=1,
                    )
                else:
                    scale_bias_last = True
                    quantized_data_npu = quantized_data_npu_ref
                dequantized_data_npu = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloatOrHalf(
                    quantized_data_npu,
                    output_dtype,
                    quant_padding_float_type=quant_padding_float_type,
                    scale_bias_last=scale_bias_last,
                )
            else:
                # legacy path does not support bf16
                if output_dtype == 5:
                    return
                if output_dtype == 0:
                    quantized_data_npu = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_data_npu)
                    dequantized_data_npu = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(quantized_data_npu)
                elif output_dtype == 1:
                    quantized_data_npu = torch.ops.fbgemm.HalfToFused8BitRowwiseQuantized(input_data_npu)
                    dequantized_data_npu = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(quantized_data_npu)
                else:
                    raise NotImplementedError("Unsupported dtype")

            dequantized_data_trimmed = dequantized_data_npu[:, :ncols].cpu()
            quantize_data_numpy = quantized_data_npu.cpu().numpy()
            if quant_padding_float_type:
                reference = torch.from_numpy(fused_rowwise_8bit_dequantize_reference(quantize_data_numpy)[:, :ncols])
            else:
                reference = torch.from_numpy(
                    fused_rowwise_8bit_dequantize_2bytes_padding_scale_bias_first_reference(quantize_data_numpy)[
                        :, :ncols
                    ]
                )
            if output_dtype == 0:
                torch.testing.assert_close(dequantized_data_trimmed.float(), reference.float())
            elif output_dtype == 1:
                torch.testing.assert_close(dequantized_data_trimmed.half(), reference.half())
            elif output_dtype == 5:
                torch.testing.assert_close(dequantized_data_trimmed.bfloat16(), reference.bfloat16())

    @given(
        nrows=st.integers(min_value=0, max_value=100),
        ncols=st.sampled_from([32, 128, 256, 384, 512, 1024]),
        output_dtype=st.sampled_from([0, 1, 5]),
        quant_padding_float_type=st.sampled_from([True, False]),
        test_generic_op=st.booleans(),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_and_dequantize_op_cpu(
        self, nrows, ncols, output_dtype, quant_padding_float_type, test_generic_op
    ):
        self.quantize_and_dequantize_op_test_helper(
            nrows, ncols, output_dtype, quant_padding_float_type, test_generic_op, False
        )

    @unittest.skipIf(not npu_available(), "Skip when NPU is not available")
    @given(
        nrows=st.integers(min_value=0, max_value=100),
        ncols=st.sampled_from([32, 128, 256, 384, 512, 1024]),
        output_dtype=st.sampled_from([0, 1, 5]),
        quant_padding_float_type=st.sampled_from([True, False]),
        test_generic_op=st.booleans(),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_and_dequantize_op_npu(
        self, nrows, ncols, output_dtype, quant_padding_float_type, test_generic_op
    ):
        self.quantize_and_dequantize_op_test_helper(
            nrows, ncols, output_dtype, quant_padding_float_type, test_generic_op, True
        )

    def test_quantize_and_dequantize_op_npu_large_nrows(self):
        ncols = 256
        nrows = 65540

        input_data = torch.rand(nrows, ncols).float()
        quantized_data = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_data)

        reference = torch.from_numpy(fused_rowwise_8bit_dequantize_reference(quantized_data.numpy()))

        if npu_available():
            input_data_npu = input_data.npu()
            quantized_data_npu = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_data_npu)
            dequantized_data_npu = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(quantized_data_npu)
            reference = torch.from_numpy(fused_rowwise_8bit_dequantize_reference(quantized_data_npu.cpu().numpy()))
            torch.testing.assert_close(dequantized_data_npu.cpu(), reference)


if __name__ == "__main__":
    unittest.main()
