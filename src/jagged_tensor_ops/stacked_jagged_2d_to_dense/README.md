# `stacked_jagged_2d_to_dense`

本算子仅支持 NPU 调用。

## 目录结构

```text
stacked_jagged_2d_to_dense
|-- stacked_jagged_2d_to_dense.cpp
|-- README.md
```

## 硬件支持情况

| 实现目录 | 典型硬件 |
| --- | --- |
| `stacked_jagged_2d_to_dense.cpp` | Atlas A5 训练系列 |

## 接口定义

```python
torch.ops.fbgemm.stacked_jagged_2d_to_dense(
    values: torch.Tensor,
    lengths: torch.Tensor,
    offset_per_key: list[int],
    max_lengths_per_key: list[int],
    padding_value: int = 0,
) -> list[torch.Tensor]

torch.ops.fbgemm.stacked_jagged_2d_to_dense_forward(
    values: torch.Tensor,
    lengths: torch.Tensor,
    offset_per_key: list[int],
    max_lengths_per_key: list[int],
    padding_value: int = 0,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]

torch.ops.fbgemm.stacked_jagged_2d_to_dense_backward(
    B: int,
    D: int,
    total_L: int,
    grad_padded_values_per_key: list[torch.Tensor],
    offsets_tensor_per_key: list[torch.Tensor],
    offset_per_key: list[int],
) -> torch.Tensor
```

## 功能说明

本算子用于将按 key/table 堆叠存储的二维 jagged tensor 转换为多个 padded dense tensor。

`values` 是所有 key 的 jagged values 按 key 顺序拼接后的二维张量，形状为 `[total_L, D]`。`lengths` 的形状为 `[T, B]`，其中 `T` 表示 key/table 数量，`B` 表示 batch size。`offset_per_key[t]` 和 `offset_per_key[t + 1]` 表示第 `t` 个 key 在 `values` 中的起止位置。

返回值为长度为 `T` 的 tensor 列表，第 `t` 个输出形状为 `[B, max_lengths_per_key[t], D]`。

数学语义如下：

```python
def stacked_jagged_2d_to_dense_forward(
    values: torch.Tensor,
    lengths: torch.Tensor,
    offset_per_key: list[int],
    max_lengths_per_key: list[int],
    padding_value: int = 0,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    T = lengths.size(0)
    outputs = []
    offsets_tensor_per_key = []

    for t in range(T):
        begin = offset_per_key[t]
        end = offset_per_key[t + 1]
        key_values = values[begin:end]
        key_lengths = lengths[t]
        key_offsets = torch.cat((
            torch.zeros((1,), dtype=key_lengths.dtype, device=key_lengths.device),
            torch.cumsum(key_lengths, dim=0),
        ))
        offsets_tensor_per_key.append(key_offsets)
        output = torch.ops.fbgemm.jagged_to_padded_dense(
            key_values,
            [key_offsets],
            [max_lengths_per_key[t]],
            float(padding_value),
        )
        outputs.append(output)

    return outputs, offsets_tensor_per_key
```

反向接口用于将每个 key 的 dense 梯度根据对应 jagged offsets 还原为堆叠后的 `values` 梯度：

```python
def stacked_jagged_2d_to_dense_backward(
    B: int,
    D: int,
    total_L: int,
    grad_padded_values_per_key: list[torch.Tensor],
    offsets_tensor_per_key: list[torch.Tensor],
    offset_per_key: list[int],
) -> torch.Tensor:
    T = len(grad_padded_values_per_key)
    grad_values = []

    for t in range(T):
        key_offsets = offsets_tensor_per_key[t]
        key_total_l = offset_per_key[t + 1] - offset_per_key[t]
        grad_values.append(torch.ops.fbgemm.jagged_to_padded_dense_backward(
            grad_padded_values_per_key[t],
            [key_offsets],
            key_total_l,
        ))

    grad_values = torch.cat(grad_values, dim=0)
    assert grad_values.shape == (total_L, D)
    return grad_values
```

## 参数说明

| 名称 | 输入/输出 | 类型 | 数据格式/形状 | 说明 |
| --- | --- | --- | --- | --- |
| `values` | 输入 | Tensor[fp32/fp16/bf16] | `[total_L, D]` | 所有 key 的二维 jagged values 按 key 顺序拼接后的结果 |
| `lengths` | 输入 | Tensor[int32/int64] | `[T, B]` | 每个 key、每个 batch 对应的 jagged 长度 |
| `offset_per_key` | 属性 | list[int] | `[T+1]` | 每个 key 在 `values` 中的起止偏移 |
| `max_lengths_per_key` | 属性 | list[int] | `[T]` | 每个 key 输出 dense tensor 的最大长度 |
| `padding_value` | 属性 | int | NA | 填充值 |
| `output` | 输出 | list[Tensor] | T 个 `[B, max_lengths_per_key[t], D]` | 每个 key 对应的 dense tensor |
| `offsets_tensor_per_key` | 输出/输入 | list[Tensor[int32/int64]] | T 个 `[B+1]` | forward 输出的每个 key 对应 offsets，backward 作为输入复用 |
| `B` | 属性 | int | NA | batch size |
| `D` | 属性 | int | NA | values 的 dense 维度 |
| `total_L` | 属性 | int | NA | values 第 0 维大小 |
| `grad_padded_values_per_key` | 输入 | list[Tensor] | T 个 `[B, max_lengths_per_key[t], D]` | 每个 key 对应的 dense 输出梯度 |
| `grad_values` | 输出 | Tensor | `[total_L, D]` | 堆叠后的 values 梯度 |

### 参数约束

1. `values.dim() == 2`。
2. `lengths.dim() == 2`。
3. `offset_per_key.size() == T + 1`，其中 `T = lengths.size(0)`。
4. `max_lengths_per_key.size() == T`。
5. `offset_per_key[t + 1] >= offset_per_key[t]`，`offset_per_key` 必须单调非递减。
6. `offset_per_key[t + 1] - offset_per_key[t] == lengths[t].sum()`。
7. `grad_padded_values_per_key.size() == T`。
8. `offsets_tensor_per_key.size() == T`，且每个 offsets tensor 的长度为 `B + 1`。
9. `grad_padded_values_per_key[t].shape == [B, max_lengths_per_key[t], D]`。
10. `total_L == offset_per_key[T]`。
11. `values`、`lengths`、`grad_padded_values_per_key`、`offsets_tensor_per_key` 必须位于 NPU 设备，且设备一致。
12. 本实现只注册正向和反向算子，不注册 meta 和 autograd。

## 调用示例

```python
import torch
import torch_npu
import fbgemm_ascend

DEVICE = "npu:0"
torch.npu.set_device(DEVICE)

values = torch.tensor(
    [
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0],
        [6.0, 60.0],
        [7.0, 70.0],
    ],
    dtype=torch.float32,
    device=DEVICE,
)
lengths = torch.tensor([[2, 1], [1, 3]], dtype=torch.int64, device=DEVICE)
offset_per_key = [0, 3, 7]
max_lengths_per_key = [3, 4]
padding_value = 0

outputs, offsets_tensor_per_key = torch.ops.fbgemm.stacked_jagged_2d_to_dense_forward(
    values,
    lengths,
    offset_per_key,
    max_lengths_per_key,
    padding_value,
)

print(len(outputs))
# > 2

print(outputs[0].shape)
# > torch.Size([2, 3, 2])

print(outputs[0])
# tensor([[[ 1., 10.],
#          [ 2., 20.],
#          [ 0.,  0.]],
#
#         [[ 3., 30.],
#          [ 0.,  0.],
#          [ 0.,  0.]]], device='npu:0')

print(outputs[1])
# tensor([[[ 4., 40.],
#          [ 0.,  0.],
#          [ 0.,  0.],
#          [ 0.,  0.]],
#
#         [[ 5., 50.],
#          [ 6., 60.],
#          [ 7., 70.],
#          [ 0.,  0.]]], device='npu:0')
```

## 编译与测试

算子编译请参考仓库根目录 [README.md](../../../README.md) 中的源码编译与安装章节。

算子测试用例请参考：

[bench/jagged/stacked_jagged_2d_to_dense/test_stacked_jagged_2d_to_dense.py](../../../bench/jagged/stacked_jagged_2d_to_dense/test_stacked_jagged_2d_to_dense.py)
