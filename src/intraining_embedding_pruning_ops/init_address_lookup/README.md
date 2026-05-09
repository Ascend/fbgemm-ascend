# `InitAddressLookup`

本算子仅支持 NPU 调用。

## 目录结构

```text
init_address_lookup
|-- init_address_lookup.cpp
|-- README.md
|-- c310/
|   |-- init_address_lookup.json
|   |-- op_host/
|   |-- op_kernel/
|   `-- run.sh
```

## 硬件支持情况

| 实现目录 | 典型硬件 |
| --- | --- |
| `c310/` | Atlas A5 训练系列 |

## 接口定义

```python
torch.ops.fbgemm.init_address_lookup(
    Tensor address_lookups,
    Tensor buffer_offsets,
    Tensor emb_sizes,
) -> None
```

**注意**：本算子为 **inplace 写入**，返回值为 `None`，调用后结果直接写入 `address_lookups` 张量。

## 功能说明

用于训练中嵌入剪枝（In-training Embedding Pruning），为每个嵌入表建立初始的"恒等映射"地址关系：

- 在 `buffer_offsets[t]` 到 `buffer_offsets[t+1]` 范围内
- 如果行索引 `r < emb_sizes[t]`，则映射到自身：`address_lookups[idx] = r`
- 如果行索引 `r >= emb_sizes[t]`，则映射到 0：`address_lookups[idx] = 0`

### 仿真/伪代码

```python
def init_address_lookup(address_lookups, buffer_offsets, emb_sizes):
    for t in range(len(emb_sizes)):
        for r in range(buffer_offsets[t + 1] - buffer_offsets[t]):
            idx = buffer_offsets[t] + r
            if r < emb_sizes[t]:
                address_lookups[idx] = r
            else:
                address_lookups[idx] = 0
    return None  # inplace
```

## 参数说明

| 名称 | 输入/输出 | 类型 | 数据格式/形状 | 说明 |
| --- | --- | --- | --- | --- |
| address_lookups | 输入（inplace） | Tensor[int64/int32] | `[total_rows]` | 预分配的地址查找表缓冲区，写入后即为输出 |
| buffer_offsets | 输入 | Tensor[int64] | `[num_tables + 1]` | CSR 格式的行偏移，定义每个嵌入表的起始索引 |
| emb_sizes | 输入 | Tensor[int64/int32] | `[num_tables]` | 每个嵌入表的逻辑行数（有效数据区大小） |

### 参数约束

- `address_lookups.dim() == 1`，`buffer_offsets.dim() == 1`，`emb_sizes.dim() == 1`
- `buffer_offsets` 必须为 int64 类型
- `emb_sizes` 和 `address_lookups` 必须为相同类型（同为 int64 或同为 int32）
- `buffer_offsets.size(0) == emb_sizes.size(0) + 1`
- `address_lookups.numel() == buffer_offsets[-1]`
- `address_lookups.numel() == 0` 时直接返回，不触发 NPU kernel

## 调用示例

```python
import torch
import torch_npu
import fbgemm_ascend

torch.npu.set_device("npu:0")

# 2 个嵌入表，表0 缓冲区 5 行，表1 缓冲区 4 行
buffer_offsets = torch.tensor([0, 5, 9], dtype=torch.int64, device="npu:0")
emb_sizes = torch.tensor([3, 4], dtype=torch.int64, device="npu:0")

total_rows = buffer_offsets[-1].item()
address_lookups = torch.empty(total_rows, dtype=torch.int64, device="npu:0")

# inplace 调用
torch.ops.fbgemm.init_address_lookup(address_lookups, buffer_offsets, emb_sizes)

# 验证结果: [0, 1, 2, 0, 0, 0, 1, 2, 3]
expected = torch.tensor([0, 1, 2, 0, 0, 0, 1, 2, 3], dtype=torch.int64)
torch.testing.assert_close(address_lookups.cpu(), expected)
```

## 编译与测试

- Ascend C 算子编译与适配层编译参考仓库根目录 [README.md](../../../README.md)
- 测试示例参考：`bench/intraining_embedding_pruning/init_address_lookup_test/`
