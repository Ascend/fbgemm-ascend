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
LARGE_SIZES = [2**16, 2**18, 2**20, 2**22, 2**24]

DATATYPE_TO_THRESHOLD = {torch.float32: 2 ** (-9), torch.float16: 2 ** (-7), torch.bfloat16: 2 ** (-6)}


def padding_for_cpu_result(input_data: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    assert input_data.dim() == 2, "Input must be 2D tensor"
    ncols_aligned = ((cols + 4 - 1) // 4) * 4
    output_cols = ncols_aligned + 8
    output = torch.zeros((rows, output_cols), dtype=torch.uint8, device=input_data.device)
    output[:, :cols] = input_data[:, :cols]
    output[:, ncols_aligned:] = input_data[:, cols:]
    return output


def bf16_to_fused8bit_rowwise_quantized(input_data: torch.Tensor) -> torch.Tensor:
    assert input_data.dim() == 2, "Input must be 2D tensor"
    rows, cols = input_data.shape

    output = torch.empty((rows, cols + 8), dtype=torch.uint8, device=input_data.device)

    input_f32 = input_data.float()
    row_min = input_f32.min(dim=1, keepdim=True).values
    row_max = input_f32.max(dim=1, keepdim=True).values

    range_val = row_max - row_min
    scale = range_val / 255.0
    bias = row_min

    eps = 1e-20
    inverse_scale = 255.0 / (range_val + eps)

    quantized = torch.round((input_f32 - bias) * inverse_scale).clamp(0, 255).to(torch.uint8)
    output[:, :cols] = quantized

    output[:, cols : cols + 4] = scale.view(torch.uint8).view(rows, 4)
    output[:, cols + 4 : cols + 8] = bias.view(torch.uint8).view(rows, 4)

    return output


def get_golden(quantized_data: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.float32:
        return torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(quantized_data)
    elif dtype == torch.float16:
        return torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(quantized_data)
    elif dtype == torch.bfloat16:
        return torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloatOrHalf(quantized_data, 5)
    else:
        return torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(quantized_data)


def get_op(quantized_data: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.float32:
        return torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(quantized_data)
    elif dtype == torch.float16:
        return torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(quantized_data)
    elif dtype == torch.bfloat16:
        return torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloatOrHalf(quantized_data, 5)
    else:
        return torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloatOrHalf(quantized_data)


def generate_input_data(device: str, dtype: torch.dtype, rows: int, cols: int) -> torch.Tensor:
    input_data = torch.rand(rows, cols, device=device, dtype=dtype) * 10 - 5
    return input_data


def get_quantized_data(input_data: torch.Tensor) -> torch.Tensor:
    rows, cols = input_data.shape
    if input_data.dtype in (torch.float32, torch.float16):
        dequantized_data = torch.ops.fbgemm.FloatOrHalfToFused8BitRowwiseQuantized(input_data)
    else:
        dequantized_data = bf16_to_fused8bit_rowwise_quantized(input_data)
    dequantized_data = padding_for_cpu_result(dequantized_data, rows, cols)
    return dequantized_data


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("total_elements", LARGE_SIZES)
@pytest.mark.parametrize("cols", [256, 512])
def test_fused8_bit_rowwise_quantized_to_float_or_half_large(dtype, total_elements, cols):
    torch.npu.set_device(DEVICE)
    input_1d = generate_input_data(device="cpu", dtype=dtype, rows=1, cols=total_elements)
    rows = total_elements // cols
    input_data = input_1d.view(rows, cols)
    quantized_data = get_quantized_data(input_data)
    golden_output = get_golden(quantized_data, dtype)
    npu_output = get_op(quantized_data.to(DEVICE), dtype)
    torch.testing.assert_close(
        npu_output.cpu(), golden_output, rtol=DATATYPE_TO_THRESHOLD[dtype], atol=DATATYPE_TO_THRESHOLD[dtype]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [2**n for n in range(4, 10)])
@pytest.mark.parametrize("cols", [2**n for n in range(4, 10)])
def test_fused8_bit_rowwise_quantized_to_float_or_half_bench(dtype, rows, cols):
    torch.npu.set_device(DEVICE)
    input_data = generate_input_data(device="cpu", dtype=dtype, rows=rows, cols=cols)
    quantized_data = get_quantized_data(input_data)
    golden_output = get_golden(quantized_data, dtype)
    npu_output = get_op(quantized_data.to(DEVICE), dtype)
    torch.testing.assert_close(
        npu_output.cpu(), golden_output, rtol=DATATYPE_TO_THRESHOLD[dtype], atol=DATATYPE_TO_THRESHOLD[dtype]
    )


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
