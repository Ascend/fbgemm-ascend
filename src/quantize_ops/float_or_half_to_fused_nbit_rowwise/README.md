# `FloatOrHalfToFusedNBitRowwise`

本算子仅支持 NPU 调用。

## 目录结构

```text
float_or_half_to_fused_nbit_rowwise
|-- float_or_half_to_fused_nbit_rowwise.cpp
|-- README.md
|-- c310/
|   |-- float_or_half_to_fused_nbit_rowwise.json
|   |-- op_host/
|   |-- op_kernel/
|   `-- run.sh
```

## 硬件支持情况

| 实现目录 | 典型硬件 |
| --- | --- |
| `c310/` | Atlas A5 训练系列 |

## 接口定义

```python
torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
    Tensor input,
    int bit_rate,
) -> Tensor

torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
    Tensor input,
    int bit_rate,
) -> Tensor

torch.ops.fbgemm.HalfToFusedNBitRowwiseQuantizedSBHalf(
    Tensor input,
    int bit_rate,
) -> Tensor
```

三个接口共享同一 kernel 实现：
- `FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf`：接受 float32 或 float16
- `FloatToFusedNBitRowwiseQuantizedSBHalf`：仅接受 float32
- `HalfToFusedNBitRowwiseQuantizedSBHalf`：仅接受 float16

## 功能说明

对输入张量逐行进行 N-bit 线性量化，并将每行的 `scale` 和 `bias`（均为 fp16）追加到行尾：

1. 对每行求 `min` 和 `max`，计算 `range = max - min`
2. `scale = range / ((1 << bit_rate) - 1)`，以 fp16 存储
3. `bias = min`，以 fp16 存储
4. 量化值：`quantized = round((x - min) / scale)`，clamp 到 `[0, (1 << bit_rate) - 1]`
5. 打包：每 `numElemPerByte` 个量化值按低 bit 在前的方式打包为 1 字节
6. 每行末尾追加 4 字节的 `scale` + `bias`

### 仿真/伪代码

```python
def quantize_row(row, bit_rate):
    num_elem_per_byte = 8 // bit_rate
    x_min = row.min()
    x_max = row.max()
    scale = (x_max - x_min) / ((1 << bit_rate) - 1)
    quantized = np.round((row - x_min) / scale).clip(0, (1 << bit_rate) - 1)
    packed = pack_bits(quantized, num_elem_per_byte)  # bit-packing
    scale_bias = np.array([scale, x_min], dtype=np.float16).view(np.uint8)
    return np.concatenate([packed, scale_bias])
```

## 参数说明

| 名称 | 输入/输出 | 类型 | 数据格式/形状 | 说明 |
| --- | --- | --- | --- | --- |
| input | 输入 | Tensor[float32/float16] | `[nrows, ncols]` | 待量化的 2D 输入张量 |
| bit_rate | 属性 | int64 | NA | 量化位宽，取值为 1、2、4 或 8 |
| output | 输出 | Tensor[uint8] | `[nrows, ceil(ncols / numElemPerByte) + 4]` | 量化结果，每行末尾含 4 字节 scale+bias |

其中 `numElemPerByte = 8 / bit_rate`。

### 参数约束

- `input.dim() == 2`
- `ncols % (2 * numElemPerByte) == 0`
- `bit_rate` 取值建议为 2 或 4；`bit_rate = 8` 有专门的优化算子
- `input` 为 float32 或 float16

## 调用示例

```python
import torch
import torch_npu
import fbgemm_ascend

torch.npu.set_device("npu:0")

nrows, ncols = 4, 16
input_data = torch.randn(nrows, ncols, dtype=torch.float32, device="npu:0")
bit_rate = 4

output = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(input_data, bit_rate)

assert output.dtype == torch.uint8
# 输出列数 = ncols / (8/4) + 4 = 16/2 + 4 = 12
assert output.shape == (nrows, ncols // (8 // bit_rate) + 4)
```

CPU/NPU 一致性验证：

```python
input_cpu = torch.randn(nrows, ncols, dtype=torch.float32)
input_npu = input_cpu.to("npu:0")

output_cpu = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(input_cpu, bit_rate)
output_npu = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(input_npu, bit_rate)

# NPU 输出与 CPU golden 逐字节一致
torch.testing.assert_close(output_npu.cpu(), output_cpu)
```

## 编译与测试

- Ascend C 算子编译与适配层编译参考仓库根目录 [README.md](../../../README.md)
- 测试示例参考：`test/quantize/fused_nbit_rowwise_test.py`或`bench/quantize/float_or_half_to_fused_nbit_rowwise_test/test_float_or_half_to_fused_nbit_rowwise.py`
