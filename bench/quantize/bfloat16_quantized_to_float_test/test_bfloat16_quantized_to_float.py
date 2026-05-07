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
import pytest
import torch
import torch_npu
import fbgemm_gpu
import fbgemm_ascend

torch.npu.config.allow_internal_format = False

DEVICE = "npu:0"

LARGE_SIZES = [2 ** 16, 2 ** 18, 2 ** 20, 2 ** 22, 2 ** 24]
LARGE_COLS = [128, 256, 512, 1024]


def generate_input_data(device: str, rows: int, cols: int) -> torch.Tensor:
    """
    生成随机 float32 输入数据，值域在 [-5, 5]，shape 为 [rows, cols]
    """
    input_data = torch.rand(rows, cols, device=device, dtype=torch.float32) * 10.0 - 5.0
    return input_data


def _run_bfloat16_quantized_to_float_test(input_data: torch.Tensor) -> None:
    """公共测试逻辑：CPU golden 与 NPU 输出比对"""
    bf16_data = torch.ops.fbgemm.FloatToBfloat16Quantized(input_data)
    golden_output = torch.ops.fbgemm.Bfloat16QuantizedToFloat(bf16_data)
    npu_output = torch.ops.fbgemm.Bfloat16QuantizedToFloat(
        bf16_data.view(torch.bfloat16).to(DEVICE)
    )
    assert npu_output.dtype == torch.float32
    assert npu_output.shape == input_data.shape
    # bf16 -> fp32 is bit-exact (left-shift by 16 bits)
    torch.testing.assert_close(
        npu_output.cpu(), golden_output, rtol=2 ** (-11), atol=2 ** (-11), equal_nan=True
    )


def test_bfloat16_quantized_to_float_small():
    """小 shape 功能验证测试"""
    torch.npu.set_device(DEVICE)
    input_data = generate_input_data(device="cpu", rows=32, cols=32)
    _run_bfloat16_quantized_to_float_test(input_data)


@pytest.mark.parametrize("total_elements", LARGE_SIZES)
@pytest.mark.parametrize("cols", LARGE_COLS)
def test_bfloat16_quantized_to_float_large(total_elements, cols):
    """大 shape 2D 张量性能/精度测试"""
    torch.npu.set_device(DEVICE)
    rows = total_elements // cols
    input_data = generate_input_data(device="cpu", rows=rows, cols=cols)
    _run_bfloat16_quantized_to_float_test(input_data)


@pytest.mark.parametrize("rows", [2 ** n for n in range(4, 12)])
@pytest.mark.parametrize("cols", [2 ** n for n in range(4, 10)])
def test_bfloat16_quantized_to_float_bench(rows, cols):
    """多维 shape 功能/性能测试"""
    torch.npu.set_device(DEVICE)
    input_data = generate_input_data(device="cpu", rows=rows, cols=cols)
    _run_bfloat16_quantized_to_float_test(input_data)


def test_bfloat16_quantized_to_float_special_values():
    """特殊值测试：0, -0, 正负值, inf, -inf, nan, 极小值"""
    torch.npu.set_device(DEVICE)
    input_data = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, float('inf'), float('-inf'), float('nan'), 1e-6, -1e-6],
        dtype=torch.float32,
        device="cpu",
    ).view(1, -1)
    _run_bfloat16_quantized_to_float_test(input_data)


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
