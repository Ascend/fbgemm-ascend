# FloatOrHalfToFused8BitRowwiseQuantized

本算子仅支持 NPU 调用。

## 目录结构

```text
float_or_half_to_fused8_bit_rowwise_quantized
|-- float_or_half_to_fused8_bit_rowwise_quantized.cpp    # PTA 适配层
|-- README.md
|-- c310/
|   |-- float_or_half_to_fused8_bit_rowwise_quantized.json
|   |-- op_host/
|   |   `-- float_or_half_to_fused8_bit_rowwise_quantized.cpp    # Tiling + InferShape
|   |-- op_kernel/
|   |   `-- float_or_half_to_fused8_bit_rowwise_quantized.cpp    # AscendC SIMT Kernel
|   |-- run.sh
```

## 硬件支持情况

| 实现目录 | 典型硬件 |
| --- | --- |
| `c310/` | Atlas A5 训练系列 |

## 接口定义

本算子暴露三个 PyTorch 接口，分别对应不同输入数据类型：

```python
# fp32 输入
torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input) -> Tensor[uint8]

# fp16 输入
torch.ops.fbgemm.HalfToFused8BitRowwiseQuantized(input) -> Tensor[uint8]

# 通用入口（fp32 / fp16 / bf16）
torch.ops.fbgemm.FloatOrHalfToFused8BitRowwiseQuantized(input) -> Tensor[uint8]
```

## 功能说明

将浮点型输入张量按**行**进行 8-bit 均匀量化，并输出 fused rowwise 格式：

```
每行输出 = [quantized_data (ncols_aligned bytes)] [scale (4 bytes)] [bias (4 bytes)]
```

其中 `ncols_aligned = (ncols + 4 - 1) / 4 * 4`，即按 4 字节对齐。

- 每行独立计算 min/max，得到 scale 和 bias
- 量化公式：`q = round((x - bias) / scale)`，结果 clamp 到 `[0, 255]`
- scale/bias 以 float32 小端序存储在每行尾部

### 伪代码

```python
def quantize_rowwise(input):
    nrows = prod(input.shape[:-1])
    ncols = input.shape[-1]
    ncols_aligned = (ncols + 3) // 4 * 4
    output_cols = ncols_aligned + 8
    output = zeros(input.shape[:-1] + [output_cols], dtype=uint8)

    for row in range(nrows):
        row_data = input[row, :]
        bias = row_data.min()
        max_val = row_data.max()
        scale = (max_val - bias) / 255.0
        quantized = clamp(round((row_data - bias) / scale), 0, 255)
        output[row, :ncols] = quantized
        output[row, ncols_aligned:ncols_aligned+4] = pack_float32(scale)
        output[row, ncols_aligned+4:ncols_aligned+8] = pack_float32(bias)
    return output
```

## 参数说明

| 名称 | 输入/输出 | 类型 | 数据格式/形状 | 说明 |
| --- | --- | --- | --- | --- |
| `inputData` | 输入 | Tensor | `dim >= 2`，最后一维为 `ncols` | 输入浮点数据，支持 fp32/fp16/bf16 |
| `y` | 输出 | Tensor[uint8] | 输入 shape 最后一维替换为 `ncols_aligned + 8` | 量化后的 fused 8-bit rowwise 数据 |

### 参数约束

- 输入维度 `dim >= 2`。除最后一维外，其余维度会被展平为 `nrows`。
- 支持空张量：`nrows == 0` 或 `ncols == 0` 时返回对应 shape 的空输出。
- 输入数据类型仅支持 `float32`、`float16`、`bfloat16`。
- 非连续（non-contiguous）输入会被自动拷贝为连续张量。
- 量化舍入方式与 FBGEMM CUDA 一致，采用 round-to-nearest（`q + 0.5f` 后 truncate）。

## 调用示例

```python
import torch
import torch_npu
import fbgemm_ascend

# fp32 示例
input_fp32 = torch.rand(128, 256).float()
output = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_fp32.npu())
print(output.shape)  # [128, 264]  (256 aligned to 256 + 8 = 264)

# bf16 示例
input_bf16 = torch.rand(64, 128).bfloat16()
output = torch.ops.fbgemm.FloatOrHalfToFused8BitRowwiseQuantized(input_bf16.npu())
print(output.shape)  # [64, 136]  (128 aligned to 128 + 8 = 136)

# 多维输入示例 (3D)
input_3d = torch.rand(2, 4, 64).float()
output = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_3d.npu())
print(output.shape)  # [2, 4, 72]  (64 aligned to 64 + 8 = 72)
```

## 编译与测试

- Ascend C 算子编译与适配层编译参考仓库根目录 [README.md](../../../README.md)
- 测试示例参考：`bench/quantize/float_or_half_to_fused8_bit_rowwise_quantized_test/test_float_or_half_to_fused8_bit_rowwise_quantized.py`
