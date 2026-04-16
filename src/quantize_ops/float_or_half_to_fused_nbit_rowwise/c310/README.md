# FloatOrHalfToFusedNBitRowwise

本算子仅支持NPU调用。

## 产品支持情况

支持 Atlas A5 系列产品（Atlas 350 加速卡，Ascend 950PR/Ascend 950DT）。

不支持更早的硬件型号。

## 目录层级

```shell
-- float_or_half_to_fused_nbit_rowwise
   |-- c310
      |-- op_host                                         # 算子host侧实现
      |-- op_kernel                                       # 算子kernel侧实现
      |-- float_or_half_to_fused_nbit_rowwise.json        # 算子原型配置
      |-- README.md                                       # 算子说明文档
      |-- run.sh                                          # 算子编译部署脚本
```

## 功能

对二维 float32 或 float16 输入张量进行逐行 N-bit 线性量化，输出紧凑的 uint8 张量。每行输出由量化后的 N-bit 打包数据与末尾追加的 2 个 float16（scale 和 bias，共 4 字节）拼接而成。

算子注册为以下接口，功能相同：

- `torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf`（接受 float32 或 float16）
- `torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf`（接受 float32）
- `torch.ops.fbgemm.HalfToFusedNBitRowwiseQuantizedSBHalf`（接受 float16）

Python 示意代码：

```python
def fused_nbit_rowwise_quantize(input, bit_rate):
    """
    Args:
        input: 2D tensor, shape [nrows, ncols], dtype float32 or float16
        bit_rate: 量化位宽, 取值 1/2/4/8
    Returns:
        output: uint8 tensor, shape [nrows, ceil(ncols / num_elem_per_byte) + 4]
    """
    nrows, ncols = input.shape
    num_elem_per_byte = 8 // bit_rate
    emb_bytes = (ncols + num_elem_per_byte - 1) // num_elem_per_byte
    max_quant = (1 << bit_rate) - 1
    output = zeros_uint8([nrows, emb_bytes + 4])

    for row in range(nrows):
        x = input[row]
        min_half = float16(min(x))
        min_f = float32(min_half)
        range_val = max(x) - min_f

        if range_val == 0:
            scale_half = float16(1.0)
        else:
            scale_half = float16(range_val / max_quant)

        scale = float32(scale_half)
        if scale == 0:
            scale_half = float16(1.0)
            scale = 1.0

        inv_scale = 1.0 / scale
        if isinf(inv_scale):
            scale_half = float16(1.0)
            inv_scale = 1.0

        # 量化
        quantized = clamp(round((x - min_f) * inv_scale), 0, max_quant)

        # 打包: 每 num_elem_per_byte 个值打包为 1 字节 (低 bit 在前)
        for col in range(ncols):
            byte_idx = col // num_elem_per_byte
            bit_offset = (col % num_elem_per_byte) * bit_rate
            if bit_offset == 0:
                output[row][byte_idx] = quantized[col]
            else:
                output[row][byte_idx] |= quantized[col] << bit_offset

        # 行末追加 scale (fp16) 和 bias (fp16)
        output[row][-4:-2] = scale_half.to_bytes()
        output[row][-2:] = min_half.to_bytes()

    return output
```

## 算子输入与输出

| 名称 | 输入/输出/属性 | 数据类型 | 数据格式 | 说明 |
| ---- | ---- | ---- | ---- | ---- |
| input | 输入 | float32, float16 | ND | 待量化的2D输入张量，shape为[nrows, ncols] |
| bitRate | 属性 | int | NA | 量化位宽，取值为1、2、4或8 |
| output | 输出 | uint8 | ND | 量化结果，shape为[nrows, ncols/numElemPerByte + 4] |

## 约束说明

- ncols % (2 * numElemPerByte) == 0，其中 numElemPerByte = 8 / bitRate。
- bitRate 取值建议为 2 或 4。bitRate 为 8 的情况有专门的算子。

## 算子编译部署

参考[README.md文档](../../../../README.md)。

运行测试：

```shell
cd bench/quantize/float_or_half_to_fused_nbit_rowwise_test
pytest test_float_or_half_to_fused_nbit_rowwise.py
```
