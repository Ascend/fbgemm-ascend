# `HFP8QuantizedToFloat`

本算子仅支持 NPU 调用。

## 目录结构

```text
hfp8_quantized_to_float
|-- hfp8_quantized_to_float.cpp
|-- README.md
|-- c310/
|   |-- hfp8_quantized_to_float.json
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
torch.ops.fbgemm.HFP8QuantizedToFloat(
    Tensor input,
    int ebits,
    int exponent_bias,
) -> Tensor
```

## 功能说明

将 HFP8（8-bit floating point）格式的 uint8 张量反量化为 float32 张量。

- 支持任意输入维度，内部按总元素数 `totalElems` 进行一维 grid-stride loop 处理
- 对 HFP8 normal 和 denormal（subnormal）值均通过浮点乘法正确解码
- Kernel 侧根据数据规模动态选择 blockDim（256/512/1024）和启用的 AI Core 数量，减少小 shape 时空转线程的调度开销

### 仿真/伪代码

```python
def hfp8_quantized_to_float(input, ebits, exponent_bias):
    mbits = 7 - ebits
    output = torch.empty_like(input, dtype=torch.float32)
    multiplier = 2.0 ** (127 - exponent_bias)

    for i in range(input.numel()):
        h = int(input.flat[i])
        sign = (h & 0x80) << 24
        val = (h & 0x7F) << (24 - 8 + ebits)   # 构造 float32 尾数/指数低位
        val_f = float_from_bits(val) * multiplier
        output.flat[i] = float_from_bits(sign | float_as_bits(val_f))

    return output
```

## 参数说明

| 名称 | 输入/输出 | 类型 | 数据格式/形状 | 说明 |
| --- | --- | --- | --- | --- |
| input | 输入 | Tensor[uint8] | 任意形状 | HFP8 量化后的输入张量 |
| ebits | 属性 | int64 | NA | HFP8 指数位宽（如 4 或 5） |
| exponent_bias | 属性 | int64 | NA | HFP8 指数偏移量（如 7 或 15） |
| output | 输出 | Tensor[float32] | 与 input 同 shape | 反量化后的浮点张量 |

### 参数约束

- `input.dtype()` 必须为 `torch.uint8`
- `ebits > 0`，典型取值为 4 或 5
- `exponent_bias > 0`，典型取值为 7 或 15
- 输出形状与输入形状完全一致
- 输入为空张量（`numel() == 0`）时，输出也为空张量

## 调用示例

```python
import torch
import torch_npu
import fbgemm_ascend

torch.npu.set_device("npu:0")

# 构造 HFP8 量化数据（ebits=4, exponent_bias=7）
input_fp32 = torch.randn(4, 16, dtype=torch.float32, device="npu:0")
hfp8_data = torch.ops.fbgemm.FloatToHFP8Quantized(
    input_fp32, ebits=4, exponent_bias=7, max_pos=448.0
)

# HFP8 反量化
output_fp32 = torch.ops.fbgemm.HFP8QuantizedToFloat(
    hfp8_data, ebits=4, exponent_bias=7
)

assert output_fp32.dtype == torch.float32
assert output_fp32.shape == input_fp32.shape
```

CPU/NPU 一致性验证：

```python
input_cpu = torch.randn(4, 16, dtype=torch.float32)
hfp8_cpu = torch.ops.fbgemm.FloatToHFP8Quantized(
    input_cpu, ebits=4, exponent_bias=7, max_pos=448.0
)

# CPU golden
output_cpu = torch.ops.fbgemm.HFP8QuantizedToFloat(hfp8_cpu, ebits=4, exponent_bias=7)
# NPU output
output_npu = torch.ops.fbgemm.HFP8QuantizedToFloat(hfp8_cpu.to("npu:0"), ebits=4, exponent_bias=7)

torch.testing.assert_close(output_npu.cpu(), output_cpu)
```

## 编译与测试

- Ascend C 算子编译与适配层编译参考仓库根目录 [README.md](../../../README.md)
- 测试示例参考：`bench/quantize/hfp8_quantized_to_float_test/test_hfp8_quantized_to_float.py`
