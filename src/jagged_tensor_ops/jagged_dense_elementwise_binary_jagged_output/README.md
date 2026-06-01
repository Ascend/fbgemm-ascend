# `jagged_dense_elementwise_binary_jagged_output`

本算子仅支持 NPU 调用。

## 目录结构

```text
jagged_dense_elementwise_binary_jagged_output
|-- jagged_dense_elementwise_binary_jagged_output.cpp
|-- README.md
|-- c310/
    |-- jagged_dense_elementwise_binary_jagged_output.json
    |-- op_host/
    |   |-- jagged_dense_elementwise_binary_jagged_output.cpp
    |   |-- jagged_dense_elementwise_binary_jagged_output_tiling.h
    |-- op_kernel/
    |   |-- jagged_dense_elementwise_binary_jagged_output.cpp
    |-- run.sh
```

## 硬件支持情况

| 实现目录 | 典型硬件 |
| --- | --- |
| `c310/` | Atlas A5 训练系列|

## 接口定义

```python
torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
    x_values: torch.Tensor,
    x_offsets: List[torch.Tensor],
    y: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]

torch.ops.fbgemm.jagged_dense_elementwise_mul(
    x_values: torch.Tensor,
    x_offsets: List[torch.Tensor],
    y: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]

torch.ops.fbgemm.jagged_dense_elementwise_mul_forward(
    x_values: torch.Tensor,
    x_offsets: List[torch.Tensor],
    y: torch.Tensor,
) -> torch.Tensor

torch.ops.fbgemm.jagged_dense_elementwise_mul_backward(
    grad_output: torch.Tensor,
    x_offsets: List[torch.Tensor],
    y: torch.Tensor,
    x_values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]
```

底层自定义算子为 `JaggedDenseElementwiseBinaryJaggedOutput`，通过 `elementwise_mode` 属性复用同一套 host 和 kernel 实现：

| elementwise_mode | FBGEMM 接口 | 功能 |
| --- | --- | --- |
| `0` | `jagged_dense_elementwise_add_jagged_output` | `x_values + dense_to_jagged(y, x_offsets)` |
| `1` | `jagged_dense_elementwise_mul` | `x_values * dense_to_jagged(y, x_offsets)` |

## 功能说明

本算子用于将 dense tensor 按 jagged offsets 抽取成与 `x_values` 对齐的 jagged values，然后与 `x_values` 做逐元素二元运算，输出仍为 jagged values 形态，同时 offsets 原样返回。

数学语义如下：

```python
y_jagged = dense_to_jagged(y, x_offsets)

if elementwise_mode == 0:
    out_values = x_values + y_jagged
elif elementwise_mode == 1:
    out_values = x_values * y_jagged

out_offsets = x_offsets
```

其中 `x_values` 支持：

```text
[total_L, D]    # vector jagged values
```

`y` 的形态与 `x_offsets` 数量对应：

```text
[B, max_L0, ..., max_Ln, D]    # vector jagged values 对应的 dense
```

## 参数说明

| 名称 | 输入/输出 | 类型 | 数据格式/形状 | 说明 |
| --- | --- | --- | --- | --- |
| `x_values` | 输入 | Tensor[fp32/fp16/bf16] | `[total_L, D]` | jagged tensor 的 values |
| `x_offsets` / `offsets` | 输入 | ListTensor[int32/int64] | 每个 offset 为 1D Tensor | jagged tensor 每个维度的偏移 |
| `y` / `dense` | 输入 | Tensor[fp32/fp16/bf16] | `[B, *max_lengths, D]` | padded dense tensor |
| `jagged_dim0` | 属性 | int | NA | `x_values` 第 0 维大小，即 `total_L` |
| `elementwise_mode` | 属性 | int | NA | `0` 表示 add，`1` 表示 mul |
| `out` / `out_values` | 输出 | 同 `x_values` | 同 `x_values` | 计算后的 jagged values |
| `out_offsets` | 输出 | 同 `x_offsets` | 同 `x_offsets` | 原样返回输入 offsets |

### 参数约束
1. `1 <= len(x_offsets) <= 5`。
2. `x_offsets[i].dim() == 1`。
3. `x_offsets[i][0] == 0`，算子中不完整校验，需用户保证。
4. `x_offsets[i][j + 1] >= x_offsets[i][j]`，offsets 必须单调非递减，算子中不完整校验，需用户保证。
5. 多层 jagged 场景下应满足 `x_offsets[i][-1] == x_offsets[i + 1].size(0) - 1`，算子中不完整校验，需用户保证。
6. `x_offsets[0].size(0) - 1 == y.size(0)`。
7. `x_values.dim() == 2`，`y.dim() == len(x_offsets) + 2`，且 `x_values.size(-1) == y.size(-1)`。
8. `elementwise_mode` 仅支持 `0` 和 `1`。
9. `x_values`、`y`、`x_offsets` 必须位于 NPU 设备，且设备一致。
10. `x_values` 和 `y` 支持 `fp32`、`fp16`、`bf16`；`x_offsets` 支持 `int32`、`int64`。
11. 当某个 jagged 坐标超过 `y` 的 padded 长度时，该位置 dense 值等价于 padding value `0`，因此 add 模式输出 `x_values`，mul 模式输出 `0`。

## 调用示例

### add_jagged_output 示例

```python
import torch
import torch_npu
import fbgemm_ascend

DEVICE = "npu:0"
torch.npu.set_device(DEVICE)

x_values = torch.tensor(
    [
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
    ],
    dtype=torch.float32,
    device=DEVICE,
)
x_offsets = [torch.tensor([0, 2, 3], dtype=torch.int64, device=DEVICE)]
y = torch.tensor(
    [
        [[100.0, 1000.0], [200.0, 2000.0], [999.0, 9999.0]],
        [[300.0, 3000.0], [888.0, 8888.0], [777.0, 7777.0]],
    ],
    dtype=torch.float32,
    device=DEVICE,
)

out_values, out_offsets = torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
    x_values,
    x_offsets,
    y,
)

print(out_values)
# tensor([[ 101., 1010.],
#         [ 202., 2020.],
#         [ 303., 3030.]], device='npu:0')
```

### mul 示例

```python
out_values, out_offsets = torch.ops.fbgemm.jagged_dense_elementwise_mul(
    x_values,
    x_offsets,
    y,
)

print(out_values)
# tensor([[  100., 10000.],
#         [  400., 40000.],
#         [  900., 90000.]], device='npu:0')
```

## 编译与测试

算子编译请参考仓库根目录 [README.md](../../../README.md) 中的源码编译与安装章节。

c310 目录也提供单算子编译入口：

```bash
cd src/jagged_tensor_ops/jagged_dense_elementwise_binary_jagged_output/c310
bash run.sh
bash run.sh ai_core-Ascend310P3
```

算子测试用例请参考：

[test/jagged/elementwise_binary_test.py](../../../test/jagged/elementwise_binary_test.py)

[bench/jagged/jagged_dense_elementwise_op/test_jagged_dense_elementwise_add.py](../../../bench/jagged/jagged_dense_elementwise_op/test_jagged_dense_elementwise_add.py)
