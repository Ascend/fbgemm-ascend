# `FloatToHFP8Quantized`

本算子仅支持 NPU 调用。

## 目录结构

```text
float_to_hfp8_quantized
|-- float_to_hfp8_quantized.cpp
|-- README.md
|-- c310/
|   |-- float_to_hfp8_quantized.json
|   |-- op_host/
|   |-- op_kernel/
|   |-- run.sh
```

## 硬件支持情况

| 实现目录 | 典型硬件 |
| --- | --- |
| `c310/` | Atlas A5 训练系列 |

## 接口定义

```python
torch.ops.fbgemm.FloatToHFP8Quantized(
    Tensor input,
    int ebits,
    int exponent_bias,
    float max_pos,
) -> Tensor
```

## 功能说明

将 float32 张量逐元素量化为 HFP8（8-bit floating point）格式，输出为 uint8 张量。

- 支持任意输入维度，内部按总元素数进行一维 grid-stride loop 处理
- 对 normal 和 denormal（subnormal）值分别走不同的量化路径
- 采用 **Bouncer 技术** 实现 Round-to-Nearest-Even 舍入，数值结果与 FBGEMM GPU 实现逐位对齐
- Kernel 侧根据数据规模动态选择 blockDim（256/512/1024）和启用的 AI Core 数量

### 仿真/伪代码

```python
def float_to_hfp8(val_fp, ebits, exponent_bias, max_pos):
    mbits = 7 - ebits
    sign_bit = float_as_bits(val_fp) & 0x80000000
    abs_val = min(abs(val_fp), max_pos)
    smallest_normal = 2 ** (1 - exponent_bias)

    if abs_val >= smallest_normal:
        # Normal path: Bouncer round-to-nearest-even
        bouncer = (float_as_bits(abs_val) & 0xFF800000) + ((23 - mbits) << 23)
        rounded = float_from_bits(float_as_bits(bouncer) + float_as_bits(abs_val) - float_as_bits(bouncer))
        hfp8 = ((float_as_bits(rounded) - ((127 - exponent_bias) << 23)) << (8 - ebits)) >> 24
    else:
        # Denormal path
        bouncer = (127 + 23 + (1 - exponent_bias - mbits)) << 23
        rounded = float_from_bits(bouncer + float_as_bits(abs_val))
        hfp8 = float_as_bits(rounded) | (sign_bit >> 24)

    return (hfp8 | (sign_bit >> 24)) & 0xFF
```

## 参数说明

| 名称 | 输入/输出 | 类型 | 数据格式/形状 | 说明 |
| --- | --- | --- | --- | --- |
| input | 输入 | Tensor[float32] | 任意形状 | 待量化的浮点输入张量 |
| ebits | 属性 | int64 | NA | 指数位宽，常见取值为 4（E4M3）或 5（E5M2） |
| exponent_bias | 属性 | int64 | NA | 指数偏移量，如 E4M3 典型值为 7 |
| max_pos | 属性 | float | NA | 最大可表示正值，超出部分将被 clamp 到该值 |
| output | 输出 | Tensor[uint8] | 与 input 同 shape | HFP8 量化结果 |

### 参数约束

- `input.dtype()` 必须为 `torch.float32`
- `ebits > 0`，典型取值为 4 或 5
- `exponent_bias > 0`，典型取值为 7 或 15
- `max_pos > 0`
- 输出形状与输入形状完全一致
- 输入为空张量（`numel() == 0`）时，输出也为空张量

## 调用示例

```python
import torch
import torch_npu
import fbgemm_ascend

torch.npu.set_device("npu:0")

# 构造 float32 输入
input_fp32 = torch.randn(4, 16, dtype=torch.float32, device="npu:0")

# E4M3 量化（与 FBGEMM GPU 一致）
output_hfp8 = torch.ops.fbgemm.FloatToHFP8Quantized(
    input_fp32,
    ebits=4,
    exponent_bias=7,
    max_pos=448.0,
)

assert output_hfp8.dtype == torch.uint8
assert output_hfp8.shape == input_fp32.shape
```

CPU/NPU 一致性验证：

```python
input_cpu = torch.randn(4, 16, dtype=torch.float32)
input_npu = input_cpu.to("npu:0")

output_cpu = torch.ops.fbgemm.FloatToHFP8Quantized(
    input_cpu, ebits=4, exponent_bias=7, max_pos=448.0
)
output_npu = torch.ops.fbgemm.FloatToHFP8Quantized(
    input_npu, ebits=4, exponent_bias=7, max_pos=448.0
)

torch.testing.assert_close(output_npu.cpu(), output_cpu)
```

## 编译与测试

- Ascend C 算子编译与适配层编译参考仓库根目录 [README.md](../../../README.md)
- 测试示例参考：`bench/quantize/float_to_hfp8_quantized_test/test_float_to_hfp8_quantized.py`
