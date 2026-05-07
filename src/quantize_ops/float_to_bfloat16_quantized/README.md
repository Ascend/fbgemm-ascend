**使用 PyTorch 框架调用方式调用 float_to_bfloat16_quantized 算子**

# PyTorch 框架对外接口原型

通过 **fbgemm** 已注册的 schema 挂载 NPU 实现（不在本库重复 `m.def`）：

```python
torch.ops.fbgemm.FloatToBfloat16Quantized(
    Tensor input,
) -> Tensor
```

实现上通过 `EXEC_NPU_CMD(aclnnFloatToBfloat16Quantized, ...)` 调用 AscendC 自定义算子 `FloatToBfloat16Quantized`，行为对齐 FBGEMM CUDA 入口 `float_to_bfloat16_quantized_cuda`（`fbgemm_gpu/src/quantize_ops/quantize_bfloat16.cu`）。

# 参数说明

| 名称 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | 说明 |
|------|----------|---------|---------|---------|------|
| input | 输入 | Tensor | float32 | ND | 任意 shape 的 fp32 张量 |
| output | 输出 | Tensor | bfloat16 | ND | 与 input 同 shape 的 bf16 张量 |

注：当前FloatToBfloat16Quantized算子的[fbgemm_gpu实现](https://github.com/pytorch/FBGEMM/blob/v1.5.0/fbgemm_gpu/src/quantize_ops/quantize_bfloat16.cu)里暂时不支持bfloat16类型，计算时使用half来代替bfloat16,；但是
Ascendc是支持bfloat16类型的，因此，为了保证算子功能与描述一致，当前Ascendc实现返回的数据类型是bfloat16，而非half，不与fbgemm_gpu
保持一致。

# 运行算子样例

## 算子编译与部署

AscendC 子算子编译部署请参考项目根目录 [README.md](../../../README.md) 中「源码编译与安装」章节。进入本目录执行：

```bash
cd src/quantize_ops/float_to_bfloat16_quantized/c310
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
import fbgemm_ascend

torch.npu.set_device("npu:0")

# 构造输入：任意 shape 的 float32 张量
input_data = torch.randn(1024, 1024, dtype=torch.float32, device="npu:0")

# 调用算子：float32 -> bfloat16
output = torch.ops.fbgemm.FloatToBfloat16Quantized(input_data)

assert output.dtype == torch.bfloat16
assert output.shape == input_data.shape
```

精度验证示例（与 GPU golden 对比）：

```python
import torch
import torch_npu
import fbgemm_ascend

torch.npu.set_device("npu:0")

input_data = torch.rand(1024, 1024, device="cpu", dtype=torch.float32) * 10.0 - 5.0

golden_output = torch.ops.fbgemm.FloatToBfloat16Quantized(input_data)
npu_output = torch.ops.fbgemm.FloatToBfloat16Quantized(input_data.to("npu:0"))

# fbgemm CUDA returns half as a 16-bit container for bfloat16 bit-pattern.
# Use .view() to reinterpret the container as bfloat16 without format conversion.
torch.testing.assert_close(
    npu_output.cpu(),
    golden_output.view(torch.bfloat16),
    rtol=2 ** (-7),
    atol=2 ** (-7),
)
```

注：上述用例为接口形态说明；完整精度与多场景测试请参考 [test_float_to_bfloat16_quantized.py](../../../bench/quantize/float_to_bfloat16_quantized_test/test_float_to_bfloat16_quantized.py)。

# aclnn 底层说明

适配层通过 `EXEC_NPU_CMD(aclnnFloatToBfloat16Quantized, ...)` 调用 `libopapi.so` 中的 `aclnnFloatToBfloat16Quantized` / `aclnnFloatToBfloat16QuantizedGetWorkspaceSize`；张量与标量顺序须与 CANN 定义一致。

当前底层调用顺序为：

1. `input`（float32）
2. `output`（bfloat16）

实现特性：

- **SIMT 编程模型**：采用 CUDA-like SIMT 单指令多线程模式，简化控制流开发。
- **向量化内存访问**：使用 `float4`（128bit）向量读取 4 个 fp32，使用 `uint64_t`（64bit）向量写入 4 个 bf16，减少内存指令数。
- **双路径处理**：向量化主路径处理 4 字节对齐数据，标量尾循环处理剩余 1~3 个元素。
- **动态资源调度**：Host 侧根据输入规模动态选择核心数与线程数（小数据用少核心降开销，大数据用满核心饱和带宽）。
- **数值一致性**：舍入策略 `(I + 0x8000) >> 16` 与 FBGEMM GPU 实现完全一致，确保 NPU 输出与 GPU golden 逐位一致。
