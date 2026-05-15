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
import fbgemm_gpu  # noqa: F401
import fbgemm_ascend  # noqa: F401

torch.npu.config.allow_internal_format = False

DEVICE = "npu:0"

# 常见 HFP8 配置: (ebits, exponent_bias, max_pos)
HFP8_CONFIGS = [
    (4, 7, 448.0),  # E4M3-like
    (5, 15, 57344.0),  # E5M2-like
]

LARGE_SIZES = [2**16, 2**18, 2**20, 2**22, 2**24]
LARGE_COLS = [128, 256, 512, 1024]


def generate_input_data(device: str, rows: int, cols: int) -> torch.Tensor:
    """
    生成随机 float32 输入数据，值域在 [-5, 5]，shape 为 [rows, cols]
    """
    input_data = torch.rand(rows, cols, device=device, dtype=torch.float32) * 10.0 - 5.0
    return input_data


def test_float_to_hfp8_quantized_small():
    """小 shape 功能验证测试"""
    torch.npu.set_device(DEVICE)
    for ebits, exponent_bias, max_pos in HFP8_CONFIGS:
        input_data = generate_input_data(device="cpu", rows=32, cols=32)
        golden_output = torch.ops.fbgemm.FloatToHFP8Quantized(input_data, ebits, exponent_bias, max_pos)
        npu_output = torch.ops.fbgemm.FloatToHFP8Quantized(input_data.to(DEVICE), ebits, exponent_bias, max_pos)
        assert npu_output.dtype == torch.uint8
        assert npu_output.shape == input_data.shape
        torch.testing.assert_close(npu_output.cpu(), golden_output, rtol=1, atol=1)


@pytest.mark.parametrize("ebits,exponent_bias,max_pos", HFP8_CONFIGS)
@pytest.mark.parametrize("total_elements", LARGE_SIZES)
@pytest.mark.parametrize("cols", LARGE_COLS)
def test_float_to_hfp8_quantized_large(ebits, exponent_bias, max_pos, total_elements, cols):
    """大 shape 2D 张量性能/精度测试"""
    torch.npu.set_device(DEVICE)
    rows = total_elements // cols
    input_data = generate_input_data(device="cpu", rows=rows, cols=cols)
    golden_output = torch.ops.fbgemm.FloatToHFP8Quantized(input_data, ebits, exponent_bias, max_pos)
    npu_output = torch.ops.fbgemm.FloatToHFP8Quantized(input_data.to(DEVICE), ebits, exponent_bias, max_pos)
    assert npu_output.dtype == torch.uint8
    assert npu_output.shape == input_data.shape
    torch.testing.assert_close(npu_output.cpu(), golden_output, rtol=1, atol=1)


@pytest.mark.parametrize("ebits,exponent_bias,max_pos", HFP8_CONFIGS)
@pytest.mark.parametrize("rows", [2**n for n in range(4, 12)])
@pytest.mark.parametrize("cols", [2**n for n in range(4, 10)])
def test_float_to_hfp8_quantized_bench(ebits, exponent_bias, max_pos, rows, cols):
    """多维 shape 功能/性能测试"""
    torch.npu.set_device(DEVICE)
    input_data = generate_input_data(device="cpu", rows=rows, cols=cols)

    golden_output = torch.ops.fbgemm.FloatToHFP8Quantized(input_data, ebits, exponent_bias, max_pos)
    npu_output = torch.ops.fbgemm.FloatToHFP8Quantized(input_data.to(DEVICE), ebits, exponent_bias, max_pos)
    assert npu_output.dtype == torch.uint8
    assert npu_output.shape == input_data.shape
    torch.testing.assert_close(npu_output.cpu(), golden_output, rtol=1, atol=1)


@pytest.mark.parametrize("ebits,exponent_bias,max_pos", HFP8_CONFIGS)
def test_float_to_hfp8_quantized_special_values(ebits, exponent_bias, max_pos):
    """特殊值测试：0, 正负大值, 小值"""
    torch.npu.set_device(DEVICE)
    input_data = torch.tensor(
        [0.0, 1.0, -1.0, max_pos, -max_pos, max_pos * 2, -max_pos * 2, 1e-6, -1e-6],
        dtype=torch.float32,
        device="cpu",
    ).view(1, -1)

    golden_output = torch.ops.fbgemm.FloatToHFP8Quantized(input_data, ebits, exponent_bias, max_pos)
    npu_output = torch.ops.fbgemm.FloatToHFP8Quantized(input_data.to(DEVICE), ebits, exponent_bias, max_pos)
    torch.testing.assert_close(npu_output.cpu(), golden_output, rtol=1, atol=1)


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
