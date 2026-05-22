# `jagged_to_padded_dense_v2`

本算子仅支持 NPU 调用。

## 目录结构

```text
jagged_to_padded_dense_v2
|-- jagged_to_padded_dense_v2.cpp
|-- README.md
|-- c310/
|   |-- run.sh
|`-- v220/
    |-- jagged_to_padded_dense_v2.json
    |-- op_host/
    |-- op_kernel/
    `-- run.sh
```

## 硬件支持情况

| 实现目录 | 典型硬件 |
| --- | --- |
| `c310/` | Atlas A5 训练系列 |
| `v220/` | Atlas A2 / A3 训练系列 |

## 接口定义

```python
torch.ops.fbgemm.jagged_to_padded_dense(
    values: torch.Tensor,
    offsets: List[torch.Tensor],
    max_lengths: List[torch.Tensor],
    padding_value: float = 0.0
) -> torch.Tensor
torch.ops.fbgemm.jagged_to_padded_dense.v2(
    values: torch.Tensor,
    offsets: List[torch.Tensor],
    max_lengths: List[torch.Tensor],
    padding_value: float = 0.0
) -> torch.Tensor
torch.ops.fbgemm.jagged_to_padded_dense_forward.v2(
    values: torch.Tensor,
    offsets: List[torch.Tensor],
    max_lengths: List[torch.Tensor],
    padding_value: float = 0.0
) -> torch.Tensor
```

## 功能说明

- 用于 jagged tensor 转 dense tensor

### 仿真/伪代码

```python
def to_padded_dense(
    values: torch.Tensor,
    offsets: list[torch.Tensor],
    max_lengths: np.typing.NDArray,
    padding_value: float = 0,
) -> torch.Tensor:
    outer_dense_size = len(offsets[0]) - 1
    # canonicalize by unsqueeze the last dim if the inner dense dimension
    # is 1 and folded.
    inner_dense_size = 1
    if values.ndim > 1:
        inner_dense_size = values.size(-1)
    dense = torch.empty(
        (outer_dense_size,) + tuple(max_lengths) + (inner_dense_size,),
        dtype=values.dtype,
        device=values.device,
    )
    for i in range(outer_dense_size):
        for jagged_coord in itertools.product(
                *(list(range(max_l)) for max_l in max_lengths)
        ):
            cur_offset = i
            is_zero = False
            for d in range(len(max_lengths)):
                begin = offsets[d][cur_offset].item()
                end = offsets[d][cur_offset + 1].item()
                if jagged_coord[d] >= end - begin:
                    is_zero = True
                    break
                cur_offset = begin + jagged_coord[d]
            dense[(i,) + jagged_coord] = (
                padding_value
                if is_zero
                else values[cur_offset]
            )
    return dense.squeeze(-1) if values.ndim == 1 else dense
```

## 参数说明

| 名称 | 输入/输出 | 类型 | 数据格式/形状 | 说明 |
| --- | --- | --- | --- | --- |
| values | 输入 | Tensor[fp32/fp16/bf16/int32/int64] | [total_L, D] | jagged tensor原始数据 |
| offsets | 输入 | ListTensor[int32/int64] | - | jagged tensor每个维度的偏移 |
| max_lengths | 属性 | List[int] | - | dense tensor每个维度的最大长度 |
| padding_value | 属性 | float | NA | 填充值 |
| output | 输出 | 同values | [len(offsets[0]) - 1, *max_lengths, D] | dense tensor |

### 参数约束

1. 1 <= len(offsets) <= 5
offsets参数作为一个List[Tensor]类型数据，其Tensor个数不超过5个，不少于1个。
2. offsets[i, 0] == 0
offsets参数每个Tensor的第一个元素必须为0。
3. offsets[i, j+1] >= offsets[i, j]
offsets参数每个Tensor必须单调递增。
4. offsets[i, -1] == offsets[i + 1].size(0) - 1
需保证前一个offset的最后一个元素等于后一个offset的元素个数减1。算子中不予校验，需用户自行保证输入的正确性。
5. len(max_lengths) == len(offsets)
max_lengths、offsets参数的长度必须相等。
6. D <= 8192
values参数的第2维大小不超过8192。

## 调用示例

```python
import torch
import torch_npu
import fbgemm_ascend
DEVICE = "npu:0"
torch.npu.set_device(DEVICE)

values = torch.range(0, 26, dtype=torch.float32, device=DEVICE).view(-1, 1)
offsets = [
    torch.tensor([0, 2, 5], device=DEVICE),
    torch.tensor([0, 8, 14, 18, 26, 27], device=DEVICE)
]
max_lengths = [2, 10]
padding_value = 0.0

output = torch.ops.fbgemm.jagged_to_padded_dense(
    values,
    offsets,
    max_lengths,
    padding_value,
)
print(output.shape)
# > torch.Size([2, 2, 10, 1])
# > (offsets[0].size(0) - 1, *max_lengths, values.size(1))

print(output.squeeze(-1))
# tensor([[[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  0.,  0.],
#          [ 8.,  9., 10., 11., 12., 13.,  0.,  0.,  0.,  0.]],

#         [[14., 15., 16., 17.,  0.,  0.,  0.,  0.,  0.,  0.],
#          [18., 19., 20., 21., 22., 23., 24., 25.,  0.,  0.]]])
```
## 编译与测试

算子编译请参考[README.md](../../../README.md)中"源码编译与安装"章节。

算子测试用例请参考:

[jagged_to_padded_dense_test.py](../../../test/jagged/jagged_to_padded_dense_test.py)

[test_jagged_to_padded_dense.py](../../../bench/jagged/jagged_to_padded_dense_test/test_jagged_to_padded_dense.py)
