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
from ctypes import c_float, c_int32, cast, POINTER, pointer

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given, HealthCheck, settings

import fbgemm_gpu
import fbgemm_ascend
from common import npu_available, bfloat_quantize


class SparseNNOperatorsGPUTest(unittest.TestCase):
    @settings(deadline=None)  # 关闭deadline检查
    @given(
        precision=st.just("BF16"),
        batch_size=st.integers(min_value=1, max_value=256),
        k=st.integers(min_value=2, max_value=2),
        n=st.integers(min_value=2, max_value=2),
    )
    def test_dense_mlp_quantize_ops(
        self, precision: str, batch_size: int, k: int, n: int
    ) -> None:
        if precision == "BF16":
            input_data = torch.rand((n, k), dtype=torch.float32)
            quantized_data = torch.ops.fbgemm.FloatToBfloat16Quantized(input_data)
            if npu_available():
                quantized_npu = torch.ops.fbgemm.FloatToBfloat16Quantized(
                    input_data.npu()
                )
                torch.testing.assert_close(quantized_npu.cpu(), quantized_data.view(torch.bfloat16))


class TestBfloat16QuantizationConversion(unittest.TestCase):
    @given(
        nrows=st.integers(min_value=0, max_value=100),
        ncols=st.integers(min_value=0, max_value=100),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_op(self, nrows: int, ncols: int) -> None:
        input_data = torch.rand(nrows, ncols).float()
        quantized_data = torch.ops.fbgemm.FloatToBfloat16Quantized(input_data)
        if nrows == 0 or ncols == 0:
            assert quantized_data.numel() == 0
            return
        f = np.vectorize(lambda x: bfloat_quantize(x))
        reference = f(input_data.numpy())
        quantized_data_uint16 = quantized_data.numpy()
        quantized_data_uint16.dtype = np.uint16
        np.testing.assert_array_almost_equal(quantized_data_uint16, reference)

        if npu_available():
            input_data_npu = input_data.npu()
            quantized_data_npu = torch.ops.fbgemm.FloatToBfloat16Quantized(
                input_data_npu
            )
            quantized_data_numpy = quantized_data_npu.cpu().view(torch.uint16).numpy()
            quantized_data_numpy.dtype = np.uint16
            np.testing.assert_allclose(quantized_data_numpy, reference)

    @unittest.skipIf(not npu_available(), "Skip when NPU is not available")
    @given(
        ncols_nrows=st.sampled_from([(65540, 256), (256, 65540)]),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_op_npu_large_nrows_bf16(
        self, ncols_nrows: tuple[int, int]
    ) -> None:
        ncols, nrows = ncols_nrows
        input_data = torch.rand(nrows, ncols).float()
        quantized_data_cpu = torch.ops.fbgemm.FloatToBfloat16Quantized(input_data)
        input_data_npu = input_data.npu()
        quantized_data_npu = torch.ops.fbgemm.FloatToBfloat16Quantized(input_data_npu)
        torch.testing.assert_close(quantized_data_npu.cpu(), quantized_data_cpu.view(torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
