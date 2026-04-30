# asynchronous_complete_cumsum

本算子仅支持 NPU 调用，适用于稀疏场景中的前缀和计算。

## 目录结构

```text
asynchronous_complete_cumsum
|-- asynchronous_complete_cumsum.cpp
|-- README.md
|-- c310/
|   |-- asynchronous_complete_cumsum.json
|   |-- op_host/
|   |-- op_kernel/
|   `-- run.sh
`-- v220/
    |-- asynchronous_complete_cumsum.json
    |-- op_host/
    |-- op_kernel/
    `-- run.sh
```

## 硬件支持

| 实现目录 | 典型硬件 |
| --- | --- |
| `c310/` | Atlas A5 训练系列 |
| `v220/` | Atlas A2 / A3 训练系列、Atlas 推理系列 |

## PyTorch 接口原型

```python
torch.ops.fbgemm.asynchronous_complete_cumsum(Tensor offset) -> Tensor
torch.ops.fbgemm.asynchronous_inclusive_cumsum(Tensor offset) -> Tensor
torch.ops.fbgemm.asynchronous_exclusive_cumsum(Tensor offset) -> Tensor

torch.ops.mxrec.asynchronous_complete_cumsum(Tensor offset) -> Tensor
torch.ops.mxrec.asynchronous_inclusive_cumsum(Tensor offset) -> Tensor
torch.ops.mxrec.asynchronous_exclusive_cumsum(Tensor offset) -> Tensor
```

## 功能说明

- `asynchronous_complete_cumsum` 返回带起点 `0` 和最终总和的前缀和。
- `asynchronous_inclusive_cumsum` 返回不带起点 `0` 但包含最终总和的前缀和。
- `asynchronous_exclusive_cumsum` 返回带起点 `0` 但不包含最终总和的前缀和。

示例输入 `offset = [1, 5, 6]` 时：

```python
complete = [0, 1, 6, 12]
inclusive = [1, 6, 12]
exclusive = [0, 1, 6]
```

## 参数与约束

| 名称 | 输入/输出 | 类型 | 说明                                                |
| --- | --- | --- |---------------------------------------------------|
| `offset` | 输入 | `Tensor[int32/int64]` | 常用为 1D；`asynchronous_complete_cumsum` 支持 1D/2D 输入 |
| `result` | 输出 | `Tensor[int32/int64]` | 输出 dtype 与输入一致                                    |

- 输入 tensor 必须位于 NPU 上。
- `asynchronous_complete_cumsum` 对 1D 输入返回长度 `N + 1`，对 2D 输入返回形状 `[B, N + 1]`。
- `asynchronous_inclusive_cumsum` / `asynchronous_exclusive_cumsum` 当前实现基于扁平化后计算，再 reshape 回原始形状，实际推荐按 1D offsets 使用。
- 需要用户自行保证累加结果不超出 `int32` / `int64` 范围。

## 调用示例

```python
import sysconfig
import torch
import torch_npu
import fbgemm_ascend

offset = torch.tensor([1, 5, 6], dtype=torch.int64, device="npu")

complete = torch.ops.fbgemm.asynchronous_complete_cumsum(offset)
inclusive = torch.ops.fbgemm.asynchronous_inclusive_cumsum(offset)
exclusive = torch.ops.fbgemm.asynchronous_exclusive_cumsum(offset)
```

## 编译与测试

- Ascend C 算子编译与适配层编译参考仓库根目录 [README.md](../../../README.md)。
- 测试示例参考 [bench/sparse/asynchronous_complete_cumsum_test/test_asynchronous_complete_cumsum.py](../../../bench/sparse/asynchronous_complete_cumsum_test/test_asynchronous_complete_cumsum.py)。
