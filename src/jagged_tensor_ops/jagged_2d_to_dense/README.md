# `jagged_2d_to_dense`

本算子仅支持 NPU 调用。

## 硬件支持情况

| 实现目录 | 典型硬件 |
| --- | --- |
| `c310/` | Atlas A5 训练系列 |
| `v220/` | Atlas A2 / A3 训练系列 |

## 接口定义

```python
torch.ops.fbgemm.jagged_2d_to_dense(
    values: torch.Tensor,
    offsets: torch.Tensor,
    max_lengths: int
) -> torch.Tensor
```

## 功能说明

- 用于二维 jagged tensor 转 dense tensor

### 仿真/伪代码

```python
def jagged_2d_to_dense(
    values: torch.Tensor,
    offsets: torch.Tensor,
    max_lengths: int,
) -> torch.Tensor:
    B = offsets.size(0) - 1
    D = values.size(-1)
    output = torch.zeros((B, max_lengths, D), dtype=values.dtype, device=values.device)

    for i in range(B):
        start = offsets[i].item()
        end = offsets[i + 1].item()
        length = end - start
        if length > 0:
            output[i, :length] = values[start:end]

    return output
```

## 参数说明

| 名称 | 输入/输出 | 类型 | 数据格式/形状 | 说明 |
| --- | --- | --- | --- | --- |
| values | 输入 | Tensor[fp32/fp16/bf16/int32/int64] | [total_L, D] | jagged tensor原始数据 |
| offsets | 输入 | Tensor[int32/int64] | [B+1] | jagged tensor的偏移 |
| max_lengths | 属性 | int | NA | dense tensor的最大长度 |
| output | 输出 | 同values | [B, max_lengths, D] | dense tensor |

### 参数约束

1. offsets.size(0) >= 2
offsets参数的长度必须至少为2。
2. offsets[0] == 0
offsets参数的第一个元素必须为0。
3. offsets[j+1] >= offsets[j]
offsets参数必须单调递增。
4. max_lengths >= 0
最大序列长度必须非负。
5. values.dim() == 2
values参数必须是二维张量。

## 调用示例

```python
import torch
import torch_npu
import fbgemm_ascend
DEVICE = "npu:0"
torch.npu.set_device(DEVICE)

values = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=torch.int64, device=DEVICE)
offsets = torch.tensor([0, 2, 5], device=DEVICE)
max_lengths = 3

output = torch.ops.fbgemm.jagged_2d_to_dense(
    values,
    offsets,
    max_lengths,
)
print(output.shape)
# > torch.Size([2, 3, 2])
# > (offsets.size(0) - 1, max_lengths, values.size(1))

print(output)
# tensor([[[ 1,  2],
#          [ 3,  4],
#          [ 0,  0]],
#
#         [[ 5,  6],
#          [ 7,  8],
#          [ 9, 10]]])
```
## 编译与测试

算子编译请参考[README.md](../../../README.md)中"源码编译与安装"章节。

算子测试用例请参考:

[2d_to_dense_test.py](../../../test/jagged/2d_to_dense_test.py)

[test_jagged_2d_to_dense.py](../../../bench/jagged/jagged_to_padded_dense_test/test_jagged_2d_to_dense.py)
