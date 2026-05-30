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
import struct
import pytest
import torch
import fbgemm_gpu  # noqa: F401
import fbgemm_ascend  # noqa: F401

torch.npu.config.allow_internal_format = False

DEVICE = "npu:0"
LARGE_SIZES = [2**16, 2**18, 2**20, 2**22, 2**24]


def bf16_to_fused8bit_rowwise_quantized(input_data: torch.Tensor) -> torch.Tensor:
    """
    将 bf16 输入量化为 8-bit 行级量化格式

    输出格式: [quantized_data (ncols_aligned)] [scale (4 bytes)] [bias (4 bytes)]

    Args:
        input_data: dim >= 2 的 bfloat16 tensor

    Returns:
        uint8 tensor, shape [..., ncols_aligned + 8]
    """
    assert input_data.dim() >= 2, "Input must have >= 2 dimensions"
    original_shape = list(input_data.shape)
    cols = original_shape[-1]
    rows = 1
    for s in original_shape[:-1]:
        rows *= s

    if rows == 0 or cols == 0:
        ncols_aligned = (cols + 4 - 1) // 4 * 4
        output_cols = ncols_aligned + 8
        out_shape = original_shape[:-1] + [output_cols]
        return torch.zeros(out_shape, dtype=torch.uint8, device=input_data.device)

    # 输出列数: 对齐后的数据 + scale (4) + bias (4)
    output_cols = cols + 8
    # 初始化输出 (uint8)
    output = torch.zeros((rows, output_cols), dtype=torch.uint8, device=input_data.device)
    # 转换为 float32 计算（与 fbgemm 一致）
    input_f32 = input_data.float().view(rows, cols)
    # 每行计算 min/max
    row_min = input_f32.min(dim=1, keepdim=True).values  # [rows, 1]
    row_max = input_f32.max(dim=1, keepdim=True).values  # [rows, 1]
    # 量化参数
    range_val = row_max - row_min
    scale = range_val / 255.0
    bias = row_min
    # 避免除零
    eps = 1e-20
    inverse_scale = 255.0 / (range_val + eps)
    # 量化: round((x - bias) * inverse_scale)
    quantized = torch.round((input_f32 - bias) * inverse_scale)
    # 钳制到 [0, 255]
    quantized = torch.clamp(quantized, 0, 255).to(torch.uint8)
    # 写入输出
    output[:, :cols] = quantized
    # 写入 scale 和 bias (float32 -> 4 bytes each, little-endian)
    for i in range(rows):
        scale_bytes = struct.pack('<f', scale[i].item())
        bias_bytes = struct.pack('<f', bias[i].item())
        output[i, cols : cols + 4] = torch.tensor(list(scale_bytes), dtype=torch.uint8)
        output[i, cols + 4 : cols + 8] = torch.tensor(list(bias_bytes), dtype=torch.uint8)

    out_shape = original_shape[:-1] + [output_cols]
    return output.view(out_shape)


def get_golden(input_data: torch.Tensor) -> torch.Tensor:
    """
    :param input_data: 需要量化的数据
    :return: 执行cpu算子量化后的数据
    """
    # 在fbgemm中，FloatToFused8BitRowwiseQuantized的cpu实现和cuda实现不一样，尤其是输出shape中的列数的计算是不一样。在cpu计算里，
    # 输出shape的output_columns是ncols+2 * sizeof(float)，其中ncols是输入的列数。但是在cuda实现里，输出shape的output_columns分成
    # 两步计算，分别是：
    # 1.ncols_aligned = (ncols + 4 - 1) / 4 * 4
    # 2.output_columns = ncols_aligned + 2 * sizeof(float)
    # 在npu实现里，也和cuda实现的输出shape保持一致，但是这个量化算子在fbgemm的cpu实现中不支持bfloat16，所以输入类型是bfloat16的时候需要
    # 使用torch来实现。
    if input_data.dtype == torch.float32:
        y = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_data)
    elif input_data.dtype == torch.float16:
        y = torch.ops.fbgemm.HalfToFused8BitRowwiseQuantized(input_data)
    elif input_data.dtype == torch.bfloat16:
        y = bf16_to_fused8bit_rowwise_quantized(input_data)
    else:
        raise TypeError(f"Unsupported dtype: {input_data.dtype}, only float32, float16 and bfloat16 are supported.")
    return y


def get_op(input_data: torch.Tensor) -> torch.Tensor:
    """
    :param input_data: 需要量化的数据
    :return: 执行npu算子量化后的数据
    """
    if input_data.dtype == torch.float32:
        y = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_data)
    elif input_data.dtype == torch.float16:
        y = torch.ops.fbgemm.HalfToFused8BitRowwiseQuantized(input_data)
    elif input_data.dtype == torch.bfloat16:
        y = torch.ops.fbgemm.FloatOrHalfToFused8BitRowwiseQuantized(input_data)
    else:
        raise TypeError(f"Unsupported dtype: {input_data.dtype}, only float32, float16 and bfloat16 are supported.")
    return y


def generate_input_data(device: str, dtype: torch.dtype, rows: int, cols: int) -> torch.Tensor:
    """
    :param device: cpu/cuda/npu
    :param dtype: 数据类型
    :param rows: 输入数据的行数
    :param cols: 输入数据的列数
    :return: 随机生成的输入数据，符合均匀分布，值域在[-5, 5]，shape是[rows, cols]，数据类型是dtype
    """
    input_data = torch.rand(rows, cols, device=device, dtype=dtype) * 10 - 5
    return input_data


def compare_result(actual: torch.Tensor, golden: torch.Tensor, cols: int):
    """
    :param golden: 标杆数据
    :param actual: npu结果
    :param cols: 输入数据实际的列数
    :return:
    """
    # 由于FloatToFused8BitRowwiseQuantized算子的量化结果是uint8类型，所以这里需要设置rtol=1, atol=1
    # 由于fbgemm中量化算子的cpu没有补齐这一步，所以比较精度的时候分成量化部分的比较和量化参数(scales, bias)的比较
    cols_aligned = (cols + 4 - 1) // 4 * 4
    # 使用 ... 支持多维输入（除最后一维外任意维度）
    # 1.比较量化数据部分[..., :cols]
    torch.testing.assert_close(actual.cpu()[..., :cols], golden[..., :cols], rtol=1, atol=1)
    # 2.比较scales
    torch.testing.assert_close(
        actual.cpu()[..., cols_aligned : cols_aligned + 4], golden[..., cols : cols + 4], rtol=1, atol=1
    )
    # 3.比较bias
    torch.testing.assert_close(
        actual.cpu()[..., cols_aligned + 4 : cols_aligned + 8], golden[..., cols + 4 : cols + 8], rtol=1, atol=1
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("total_elements", LARGE_SIZES)
@pytest.mark.parametrize("cols", [256, 512, 1024])
def test_float_or_half_to_fused8_bit_rowwise_quantized_large(dtype, total_elements, cols):
    torch.npu.set_device(DEVICE)
    input_1d = generate_input_data(device="cpu", dtype=dtype, rows=1, cols=total_elements)
    rows = total_elements // cols
    input_data = input_1d.view(rows, cols)
    output_cpu = get_golden(input_data)
    output_npu = get_op(input_data.to(DEVICE))
    compare_result(output_npu.cpu(), output_cpu, cols)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [torch.randint(1, 101, (1,)).item()])
@pytest.mark.parametrize("cols", [5, 7, 10, 15, 17, 31, 33, 63, 65, 100, 127, 129, 255, 257])
def test_float_or_half_to_fused8_bit_rowwise_quantized_unaligned(dtype, rows, cols):
    torch.npu.set_device(DEVICE)
    input_data = generate_input_data(device="cpu", dtype=dtype, rows=rows, cols=cols)
    output_cpu = get_golden(input_data)
    output_npu = get_op(input_data.to(DEVICE))
    compare_result(output_npu.cpu(), output_cpu, cols)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [2**n for n in range(4, 10)])
@pytest.mark.parametrize("cols", [2**n for n in range(4, 11)])
def test_float_or_half_to_fused8_bit_rowwise_quantized_bench(dtype, rows, cols):
    torch.npu.set_device(DEVICE)
    input_data = generate_input_data(device="cpu", dtype=dtype, rows=rows, cols=cols)
    output_cpu = get_golden(input_data)
    output_npu = get_op(input_data.to(DEVICE))
    compare_result(output_npu.cpu(), output_cpu, cols)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (0, 16),  # nrows = 0
        (16, 0),  # ncols = 0
        (0, 0),  # both zero
    ],
)
def test_float_or_half_to_fused8_bit_rowwise_quantized_empty(dtype, shape):
    """Test empty tensor support: should return empty output without error."""
    torch.npu.set_device(DEVICE)
    rows, cols = shape
    input_data = generate_input_data(device="cpu", dtype=dtype, rows=rows, cols=cols)
    output_npu = get_op(input_data.to(DEVICE))

    ncols_aligned = (cols + 4 - 1) // 4 * 4
    output_cols = ncols_aligned + 8
    expected_shape = (rows, output_cols)
    assert output_npu.shape == expected_shape, f"Expected shape {expected_shape}, got {output_npu.shape}"
    assert output_npu.numel() == 0 or output_npu.dtype == torch.uint8


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 3, 16),  # 3D
        (2, 3, 32),  # 3D
        (2, 2, 3, 16),  # 4D
        (2, 2, 3, 32),  # 4D
    ],
)
def test_float_or_half_to_fused8_bit_rowwise_quantized_multidim(dtype, shape):
    """Test multi-dimensional input (dim >= 2): should flatten inner dims and preserve outer shape."""
    torch.npu.set_device(DEVICE)
    cols = shape[-1]
    # Resize to target shape while preserving random distribution
    input_data = torch.rand(shape, device="cpu", dtype=dtype) * 10 - 5
    output_cpu = get_golden(input_data)
    output_npu = get_op(input_data.to(DEVICE))
    compare_result(output_npu.cpu(), output_cpu, cols)


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
