# `stacked_jagged_1d_to_dense`

本算子仅支持 NPU 调用。

## 硬件支持情况

| 实现目录 | 典型硬件 |
| --- | --- |
| `stacked_jagged_1d_to_dense` | Atlas A5 训练系列 |

## 接口定义

```python
torch.ops.fbgemm.stacked_jagged_1d_to_dense(
    values: torch.Tensor,
    lengths: torch.Tensor,
    offset_per_key: list[int],
    max_lengths_per_key: list[int],
    padding_value: int = 0,
) -> list[torch.Tensor]
```

## 功能说明

- 用于将按 key 堆叠存储的一维 jagged tensor 转换为多个 dense tensor。
- `lengths` 的形状为 `[T, B]`，其中 `T` 表示 key/table 数量，`B` 表示 batch size。
- `values` 是所有 key 的 jagged values 按 key 顺序拼接后的一维张量。
- `offset_per_key[t]` 和 `offset_per_key[t + 1]` 表示第 `t` 个 key 在 `values` 中的起止位置。
- 返回长度为 `T` 的 tensor 列表，第 `t` 个输出形状为 `[B, max_lengths_per_key[t]]`。

### 仿真/伪代码

```python
def stacked_jagged_1d_to_dense(
    values: torch.Tensor,
    lengths: torch.Tensor,
    offset_per_key: list[int],
    max_lengths_per_key: list[int],
    padding_value: int = 0,
) -> list[torch.Tensor]:
    T = lengths.size(0)
    outputs = []

    for t in range(T):
        begin = offset_per_key[t]
        end = offset_per_key[t + 1]
        key_values = values[begin:end]
        key_lengths = lengths[t]
        key_offsets = torch.cat((
            torch.zeros((1,), dtype=key_lengths.dtype, device=key_lengths.device),
            torch.cumsum(key_lengths, dim=0),
        ))
        output = torch.ops.fbgemm.jagged_to_padded_dense(
            key_values,
            [key_offsets],
            [max_lengths_per_key[t]],
            float(padding_value),
        )
        outputs.append(output)

    return outputs
```

## 参数说明

| 名称 | 输入/输出 | 类型 | 数据格式/形状 | 说明 |
| --- | --- | --- | --- | --- |
| values | 输入 | Tensor[fp32/fp16/bf16/int32/int64] | [total_L] | 所有 key 的 jagged tensor 原始数据按 key 顺序拼接后的结果 |
| lengths | 输入 | Tensor[int32/int64] | [T, B] | 每个 key、每个 batch 对应的 jagged 长度 |
| offset_per_key | 属性 | list[int] | [T+1] | 每个 key 在 values 中的起止偏移 |
| max_lengths_per_key | 属性 | list[int] | [T] | 每个 key 输出 dense tensor 的最大长度 |
| padding_value | 属性 | int | NA | 填充值 |
| output | 输出 | list[Tensor] | T 个 [B, max_lengths_per_key[t]] | 每个 key 对应的 dense tensor |

### 参数约束

1. values.dim() == 1
values 参数必须是一维张量。
2. lengths.dim() == 2
lengths 参数必须是二维张量。
3. offset_per_key.size() == T + 1
其中 T 为 lengths.size(0)。
4. max_lengths_per_key.size() == T
每个 key 必须提供一个最大输出长度。
5. offset_per_key[t + 1] >= offset_per_key[t]
offset_per_key 参数必须单调递增。
6. offset_per_key[t + 1] - offset_per_key[t] == lengths[t].sum()
每个 key 的 values 切片长度必须等于对应 lengths 行的总和。
7. values 和 lengths 必须位于 NPU 设备。

## 调用示例

```python
import torch
import torch_npu
import fbgemm_ascend
DEVICE = "npu:0"
torch.npu.set_device(DEVICE)

values = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.int64, device=DEVICE)
lengths = torch.tensor([[2, 1], [1, 3]], dtype=torch.int64, device=DEVICE)
offset_per_key = [0, 3, 7]
max_lengths_per_key = [3, 4]
padding_value = 0

outputs = torch.ops.fbgemm.stacked_jagged_1d_to_dense(
    values,
    lengths,
    offset_per_key,
    max_lengths_per_key,
    padding_value,
)

print(len(outputs))
# > 2

print(outputs[0].shape)
# > torch.Size([2, 3])

print(outputs[0])
# tensor([[1, 2, 0],
#         [3, 0, 0]], device='npu:0')

print(outputs[1])
# tensor([[4, 0, 0, 0],
#         [5, 6, 7, 0]], device='npu:0')
```

## 编译与测试

算子编译请参考[README.md](../../../README.md)中"源码编译与安装"章节。

算子测试用例请参考:

[test_stacked_jagged_1d_to_dense.py](../../../bench/jagged/stacked_jagged_1d_to_dense/test_stacked_jagged_1d_to_dense.py)
