# `jagged_dense_dense_elementwise_add_jagged_output`

本算子仅支持 NPU 调用。

## 目录结构

```text
jagged_dense_dense_elementwise_add_jagged_output
|-- jagged_dense_dense_elementwise_add_jagged_output.cpp
|-- README.md
|-- c310/
    |-- jagged_dense_dense_elementwise_add_jagged_output.json
    |-- op_host/
    |-- op_kernel/
    |-- run.sh
```

## 硬件支持情况

| 实现目录    | 典型硬件           |
| ------- | -------------- |
| `c310/` | Atlas 950 训练系列 |

## 接口定义

```python
torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output(
    xValues: torch.Tensor,
    offsets: List[torch.Tensor],
    y0: torch.Tensor,
    y1: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]

torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output_forward(
    xValues: torch.Tensor,
    offsets: List[torch.Tensor],
    y0: torch.Tensor,
    y1: torch.Tensor,
) -> torch.Tensor

torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output_backward(
    grad_output: torch.Tensor,
    xValues: torch.Tensor,
    offsets: List[torch.Tensor],
    y0: torch.Tensor,
    y1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

## 功能说明

- 将 jagged tensor `xValues` 按 `offsets` 映射到与 dense tensor `y0` / `y1` 对齐的 padded dense 空间。
- 对有效 jagged 位置执行 `xValues + y0 + y1`。
- 输出仍为 jagged values，并原样返回 `offsets`。
- 支持 AutogradPrivateUse1 反向传播，`xValues` 梯度直接透传，`y0` / `y1` 梯度按照 `offsets` 写回到 padded dense 形状，padding 区域梯度为 0。

### 仿真/伪代码

```python
def jagged_dense_dense_elementwise_add_jagged_output(
    xValues: torch.Tensor,
    offsets: list[torch.Tensor],
    y0: torch.Tensor,
    y1: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    max_lengths = list(y0.shape[1:-1])
    x_dense = torch.ops.fbgemm.jagged_to_padded_dense(
        xValues,
        offsets,
        max_lengths,
        0.0,
    )
    dense_sum = x_dense + y0 + y1
    out_values = dense_to_jagged(dense_sum, offsets)
    return out_values, offsets
```

其中 `dense_to_jagged` 表示按 `offsets` 仅采集 padded dense 中有效 jagged 位置的元素。

## 参数说明

| 名称      | 输入/输出 | 类型                       | 数据格式/形状                                        | 说明                            |
| ------- | ----- | ------------------------ | ---------------------------------------------- | ----------------------------- |
| xValues | 输入    | Tensor\[fp32/fp16/bf16]  | \[total\_L, D]                   | jagged tensor 原始数据            |
| offsets | 输入/输出 | ListTensor\[int32/int64] | -                                              | jagged tensor 每个维度的偏移，输出时原样返回 |
| y0      | 输入    | Tensor\[fp32/fp16/bf16]  | \[B, \*max\_lengths, D] | 第一个 dense 输入                  |
| y1      | 输入    | Tensor\[fp32/fp16/bf16]  | 与 y0 相同                                        | 第二个 dense 输入                  |
| output  | 输出    | 同 xValues                | \[total\_L, D]                   | jagged 输出 values              |

### 参数约束

1. 1 <= len(offsets) <= 5
   offsets 参数作为一个 List\[Tensor] 类型数据，其 Tensor 个数不超过 5 个，不少于 1 个。
2. offsets\[i] 必须为 1D Tensor。
3. offsets\[0].size() - 1 必须等于 y0.size(0)。
4. y0 和 y1 的 shape 必须完全一致。
6. y0 / y1 的维度必须为 len(offsets) + 2，形状为 \[B, \*max\_lengths, D]，且 xValues.size(-1) == y0.size(-1)。
7. offsets\[i, 0] == 0。offsets 参数每个 Tensor 的第一个元素必须为 0。
8. offsets\[i, j+1] >= offsets\[i, j]。offsets 参数每个 Tensor 必须单调递增。
9. offsets\[i, -1] == offsets\[i + 1].size(0) - 1。
   需保证前一个 offset 的最后一个元素等于后一个 offset 的元素个数减 1。

## 调用示例

```python
import torch
import torch_npu
import fbgemm_ascend

DEVICE = "npu:0"
torch.npu.set_device(DEVICE)

xValues = torch.tensor(
    [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
    dtype=torch.float32,
    device=DEVICE,
)
offsets = [torch.tensor([0, 2, 3], dtype=torch.int64, device=DEVICE)]
y0 = torch.tensor(
    [
        [[100.0, 1000.0], [200.0, 2000.0]],
        [[300.0, 3000.0], [400.0, 4000.0]],
    ],
    dtype=torch.float32,
    device=DEVICE,
)
y1 = torch.tensor(
    [
        [[10.0, 100.0], [20.0, 200.0]],
        [[30.0, 300.0], [40.0, 400.0]],
    ],
    dtype=torch.float32,
    device=DEVICE,
)

output, output_offsets = torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output(
    xValues,
    offsets,
    y0,
    y1,
)

print(output.shape)
# > torch.Size([3, 2])
# > (offsets[-1][-1], xValues.size(1))

print(output)
# tensor([[ 111., 1110.],
#         [ 222., 2220.],
#         [ 333., 3330.]], device='npu:0')

print(torch.equal(output_offsets[0], offsets[0]))
# > True
```

## 编译与测试

算子编译请参考[README.md](../../../README.md)中"源码编译与安装"章节。

算子测试用例请参考:

[test\_jagged\_dense\_dense\_elementwise\_add\_jagged\_output.py](../../../bench/jagged/jagged_dense_dense_elementwise_add_jagged_output/test_jagged_dense_dense_elementwise_add_jagged_output.py)
