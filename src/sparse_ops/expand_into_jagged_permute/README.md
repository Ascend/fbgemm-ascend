# expand_into_jagged_permute

本算子仅支持 NPU 调用，用于把将稀疏数据置换索引从表维度扩展到批次维度。

## 目录结构

```text
expand_into_jagged_permute
|-- expand_into_jagged_permute.cpp
|-- README.md
`-- c310/
    |-- expand_into_jagged_permute.json
    |-- op_host/
    |-- op_kernel/
    `-- run.sh
```

## 硬件支持

| 实现目录 | 典型硬件 |
| --- | --- |
| `c310/` | Atlas A5 训练系列 |

## PyTorch 接口原型

```python
torch.ops.mxrec.expand_into_jagged_permute(
    Tensor permute,
    Tensor input_offset,
    Tensor output_offset,
    SymInt output_size
) -> Tensor

torch.ops.fbgemm.expand_into_jagged_permute(
    Tensor permute,
    Tensor input_offset,
    Tensor output_offset,
    SymInt output_size
) -> Tensor
```

## 功能说明

输入 `permute` 描述表顺序重排，算子会把每张表对应的区间展开为逐元素索引。

```python
def expand_into_jagged_permute(permute, input_offsets, output_offsets, output_size):
    output = []
    for i in range(len(permute)):
        begin = input_offsets[permute[i]]
        end = input_offsets[permute[i] + 1]
        output.extend(range(begin, end))
    return output
```

## 参数与约束

| 名称 | 输入/输出 | 类型 | 说明 |
| --- | --- | --- | --- |
| `permute` | 输入 | `Tensor[int32/int64]` | 1D 表级置换索引 |
| `input_offset` | 输入 | `Tensor[int32/int64]` | 1D 原始 offsets，长度为 `permute.numel() + 1` |
| `output_offset` | 输入 | `Tensor[int32/int64]` | 1D 置换后 offsets，长度为 `permute.numel() + 1` |
| `output_size` | 输入属性 | `SymInt` | 输出长度，通常等于 `output_offset[-1]` |
| `output` | 输出 | `Tensor[int32/int64]` | 展开后的逐元素 permute |

- 所有输入必须位于同一块 NPU 上。
- `permute`、`input_offset`、`output_offset` 的 dtype 必须一致。
- `permute.numel() > 0`。
- `permute.numel() == input_offset.numel() - 1 == output_offset.numel() - 1`。

## 调用示例

```python
import sysconfig
import torch
import torch_npu
import fbgemm_ascend

permute = torch.tensor([2, 0, 1], dtype=torch.int32, device="npu")
input_offsets = torch.tensor([0, 10, 30, 45], dtype=torch.int32, device="npu")
output_offsets = torch.tensor([0, 15, 25, 45], dtype=torch.int32, device="npu")

output = torch.ops.fbgemm.expand_into_jagged_permute(
    permute, input_offsets, output_offsets, 45
)
```

## 编译与测试

- Ascend C 算子编译与适配层编译参考仓库根目录 [README.md](../../../README.md)。
- 测试示例参考 [bench/sparse/expand_into_jagged_permute/test_expand_into_jagged_permute.py](../../../bench/sparse/expand_into_jagged_permute/test_expand_into_jagged_permute.py)。

