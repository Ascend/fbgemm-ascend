**使用 PyTorch 框架调用方式调用 fused8_bit_rowwise_quantized_to_float_or_half 算子**

# PyTorch 框架对外接口原型

通过 **fbgemm** 已注册的 schema 挂载 NPU 实现（不在本库重复 `m.def`）：

```python
# 快捷入口：固定输出为 float32
torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
    Tensor input,
) -> Tensor

# 快捷入口：固定输出为 float16
torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(
    Tensor input,
) -> Tensor

# 通用入口：支持 FP32 / FP16 / BF16 三种输出
torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloatOrHalf(
    Tensor input,
    int output_dtype = 0,
    bool scale_bias_last = True,
    bool quant_padding_float_type = True,
) -> Tensor
```

实现上通过 `EXEC_NPU_CMD(aclnnFused8BitRowwiseQuantizedToFloatOrHalf, ...)` 调用 AscendC 自定义算子 `Fused8BitRowwiseQuantizedToFloatOrHalf`，行为对齐 FBGEMM CUDA 入口（`fbgemm_gpu/src/quantize_ops/quantize_fused_8bit_rowwise.cu`）。

# 参数说明

| 名称 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | 说明 |
|------|----------|---------|---------|---------|------|
| input | 输入 | Tensor | uint8 | ND | fused 8-bit rowwise 量化数据，最后 2×quantPaddingSize 字节为 scale/bias |
| output_dtype | 输入 | int64 | - | - | 输出类型：`0=FP32`, `1=FP16`, `5=BF16` |
| scale_bias_last | 输入 | bool | - | - | scale/bias 是否放在每行末尾；`False` 时放在行首 |
| quant_padding_float_type | 输入 | bool | - | - | scale/bias 存储类型：`True=float32`, `False=half` |
| y | 输出 | Tensor | FP32/FP16/BF16 | ND | 反量化后的浮点张量，shape 为 input 前 N-1 维 + (cols - 2×quantPaddingSize) |

注：当前算子的实现中 `output_dtype` 的取值与 FBGEMM `SparseType` 枚举对齐（`FP32=0, FP16=1, BF16=5`）。

# 运行算子样例

## 算子编译与部署

AscendC 子算子编译部署请参考项目根目录 [README.md](../../../README.md) 中「源码编译与安装」章节。进入本目录执行：

```bash
cd src/quantize_ops/fused8_bit_rowwise_quantized_to_float_or_half/c310
bash run.sh
```

集成在 `fbgemm_ascend` whl 包中时，通常由包内路径自动加载，无需单独加载本目录产物。

## PyTorch 编译

PyTorch 适配层编译请参考项目根目录 [README.md](../../../README.md) 中「源码编译与安装」章节。执行：

```bash
pip install . --no-build-isolation
```

或：

```bash
bash build_whl.sh
pip install dist/fbgemm_ascend-*.whl
```

## 算子调用示例

以下示例展示调用 **fbgemm** 算子（需已安装 `fbgemm_gpu` 并完成 schema 注册；设备为 NPU）：

```python
import torch
import torch_npu
import fbgemm_gpu
import fbgemm_ascend

torch.npu.set_device("npu:0")

# 构造输入：先通过 FloatToFused8BitRowwiseQuantized 生成量化张量
input_fp32 = torch.randn(1024, 256, dtype=torch.float32, device="cpu")
quantized = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_fp32)

# 调用算子：反量化为 float32
output = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
    quantized.to("npu:0")
)

assert output.dtype == torch.float32
assert output.shape == torch.Size([1024, 248])  # 256 - 8 = 248
```

## 精度验证示例

```python
import torch
import torch_npu
import fbgemm_ascend

torch.npu.set_device("npu:0")

rows, cols = 1024, 256
input_fp32 = torch.rand(rows, cols - 8, device="cpu", dtype=torch.float32) * 10.0 - 5.0
quantized = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_fp32)

golden_output = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(quantized)
npu_output = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
    quantized.to("npu:0")
)

# 反量化存在 8-bit 量化误差，容差需覆盖量化步长
quantization_error = 2 * (input_fp32.max() - input_fp32.min()) / 255.0
torch.testing.assert_close(
    npu_output.cpu(),
    golden_output,
    rtol=1e-3,
    atol=quantization_error.item(),
    equal_nan=True,
)
```

注：上述用例为接口形态说明；完整精度与多场景测试请参考 [test_fused8_bit_rowwise_quantized_to_float_or_half.py](../../../bench/quantize/fused8_bit_rowwise_quantized_to_float_or_half_test/test_fused8_bit_rowwise_quantized_to_float_or_half.py)。

# aclnn 底层说明

适配层通过 `EXEC_NPU_CMD(aclnnFused8BitRowwiseQuantizedToFloatOrHalf, ...)` 调用 `libopapi.so` 中的 `aclnnFused8BitRowwiseQuantizedToFloatOrHalf` / `aclnnFused8BitRowwiseQuantizedToFloatOrHalfGetWorkspaceSize`；张量与标量顺序须与 CANN 定义一致。

当前底层调用顺序为：

1. `inputData`（uint8）
2. `outputDtype`（int64）
3. `scaleBiasLast`（bool）
4. `quantPaddingFloatType`（bool）
5. `y`（FP32 / FP16 / BF16）

实现特性：

- **SIMT 编程模型**：采用 CUDA-like SIMT 单指令多线程模式，每个 AIV Core 作为一个 block，线程内完成多行数据的反量化。
- **向量化内存访问**：
  - FP32 输出路径：`uint32_t` 向量读取 4 个 uint8，`float4`（128bit）向量写入 4 个 fp32。
  - FP16/BF16 输出路径：`uint32_t` 向量读取 4 个 uint8，`uint64_t` 向量写入 4 个 fp16/bf16。
- **多线程行内并行**：每行由 `threadsPerRow` 个线程协作处理，通过位运算（`& / >>`）快速计算线程在行内偏移。
- **动态 Tiling 策略**：Host 侧根据输入 shape 动态选择 `threadsPerRow`（避免空闲线程）和 `rowsPerBlock`（保证 block 内至少 128 线程），并通过 grid-stride loop 遍历全局数据。
- **三输出类型支持**：`FP32=0 / FP16=1 / BF16=5`，与 FBGEMM `SparseType` 枚举对齐，编译期通过 `if constexpr` 展开为独立代码路径。
- **尾循环处理**：向量化主路径处理 4 字节对齐数据，标量尾循环处理剩余 0~3 列。
