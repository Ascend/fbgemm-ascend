# `jagged_dense_elementwise_add`

本算子仅支持 NPU 调用。

## 目录结构

```text
jagged_dense_elementwise
|-- jagged_dense_elementwise.cpp
`-- README.md
```

## 硬件支持情况

| 实现目录                        | 典型硬件        |
| --------------------------- | ----------- |
| `jagged_dense_elementwise/` | Atlas A5 系列 |

## 接口定义

```python
torch.ops.fbgemm.jagged_dense_elementwise_add(
    jagged_values: torch.Tensor,
    offsets: List[torch.Tensor],
    dense_tensor: torch.Tensor,
) -> torch.Tensor
```

内部反向接口：

```python
torch.ops.fbgemm.jagged_dense_elementwise_add_backward(
    grad_output: torch.Tensor,
    offsets: List[torch.Tensor],
    total_L: int,
    jagged_values_is_1d: bool,
) -> torch.Tensor
```

## 功能说明

- 将 jagged tensor 按 offsets 转换成与 dense\_tensor 形状一致的 padded dense tensor，padding\_value 固定为 0.0。
- 对 padded jagged tensor 与 dense\_tensor 执行逐元素加法。
- 支持 AutogradPrivateUse1 反向传播，jagged\_values 的梯度会按照 offsets 从 dense 梯度映射回 jagged values，dense\_tensor 的梯度直接透传。

### 仿真/伪代码

```python
def jagged_dense_elementwise_add(
    jagged_values: torch.Tensor,
    offsets: list[torch.Tensor],
    dense_tensor: torch.Tensor,
) -> torch.Tensor:
    x_is_1d = jagged_values.ndim == 1
    max_lengths = list(dense_tensor.shape[1:] if x_is_1d else dense_tensor.shape[1:-1])
    padded_jagged = torch.ops.fbgemm.jagged_to_padded_dense(
        jagged_values.unsqueeze(-1) if x_is_1d else jagged_values,
        offsets,
        max_lengths,
        0.0,
    )
    if x_is_1d:
        padded_jagged = padded_jagged.squeeze(-1)
    return padded_jagged + dense_tensor
```

## 参数说明

| 名称             | 输入/输出 | 类型                       | 数据格式/形状                                        | 说明                                      |
| -------------- | ----- | ------------------------ | ---------------------------------------------- | --------------------------------------- |
| jagged\_values | 输入    | Tensor\[fp32/fp16/bf16]  | \[total\_L] 或 \[total\_L, D]                   | jagged tensor 原始数据                      |
| offsets        | 输入    | ListTensor\[int32/int64] | -                                              | jagged tensor 每个维度的偏移                   |
| dense\_tensor  | 输入    | 同 jagged\_values         | \[B, \*max\_lengths] 或 \[B, \*max\_lengths, D] | 与 padded jagged tensor 相加的 dense tensor |
| output         | 输出    | 同 jagged\_values         | 同 dense\_tensor                                | 逐元素相加后的 dense tensor                    |

### 参数约束

1. 1 <= len(offsets) <= 5
   offsets 参数作为 List\[Tensor] 类型数据，其 Tensor 个数不超过 5 个，不少于 1 个。
2. offsets 中每个 Tensor 必须是一维非空 Tensor。
3. jagged\_values、offsets、dense\_tensor 必须在同一个 NPU 设备上。
4. dense\_tensor.size(0) == offsets\[0].size() - 1
   ,dense\_tensor 的 batch 维必须与 offsets\[0] 推导出的 batch size 一致。
5. 当 jagged\_values 为二维 Tensor 时，jagged\_values.size(-1) == dense\_tensor.size(-1)。
6. dense\_tensor 的中间维度会作为每个 jagged 维度的 max\_lengths，维度个数必须与 len(offsets) 匹配。
7. offsets\[i, 0] == 0，offsets\[i, j+1] >= offsets\[i, j]，并且多级 offsets 之间的长度关系需由用户保证。

## 调用示例

```python
import torch
import torch_npu
import fbgemm_ascend
DEVICE = "npu:0"
torch.npu.set_device(DEVICE)

jagged_values = torch.tensor(
    [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
    dtype=torch.float32,
    device=DEVICE,
)
offsets = [torch.tensor([0, 2, 3], dtype=torch.int64, device=DEVICE)]
dense_tensor = torch.tensor(
    [
        [[100.0, 1000.0], [200.0, 2000.0]],
        [[300.0, 3000.0], [400.0, 4000.0]],
    ],
    dtype=torch.float32,
    device=DEVICE,
)

output = torch.ops.fbgemm.jagged_dense_elementwise_add(
    jagged_values,
    offsets,
    dense_tensor,
)
print(output.shape)
# > torch.Size([2, 2, 2])
# > (offsets[0].size(0) - 1, dense_tensor.size(1), jagged_values.size(1))

print(output)
# tensor([[[ 101., 1010.],
#          [ 202., 2020.]],
#
#         [[ 303., 3030.],
#          [ 400., 4000.]]], device='npu:0')
```

## 编译与测试

算子编译请参考[README.md](../../../README.md)中"源码编译与安装"章节。

算子测试用例请参考:

[test\_jagged\_dense\_elementwise\_add.py](../../../bench/jagged/jagged_dense_elementwise_op/test_jagged_dense_elementwise_add.py)
