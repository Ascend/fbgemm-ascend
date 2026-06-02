# lxu_cache_lookup

本算子仅支持 NPU 调用，用于 LXU 缓存查找操作。

## 目录结构

```text
lxu_cache_lookup
|-- lxu_cache_lookup.cpp
|-- README.md
|-- c310/
|   |-- op_host/
|   |-- op_kernel/
|   |-- run.sh
|   |-- lxu_cache_lookup.json

```

## 硬件支持

| 实现目录 | 典型硬件 |
| --- | --- |
| `c310/` | Atlas A5 训练系列 |

## PyTorch 接口原型

```python
torch.ops.fbgemm.lxu_cache_lookup(
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    int invalid_index,
    bool gather_cache_stats,
    Tensor? uvm_cache_stats = None,
    Tensor? num_uniq_cache_indices = None,
    Tensor? lxu_cache_locations_output = None
) -> Tensor
```

## 功能说明

`lxu_cache_lookup` 执行 LXU 缓存查找操作，根据给定的缓存索引从缓存状态中检索对应的缓存位置。该算子内部调用 Ascend NPU 的 `aclnnLxuCacheLookup` 算子实现。

### 主要功能

- 根据 `linear_cache_indices` 在 `lxu_cache_state` 中查找对应的缓存位置
- 支持统计缓存命中信息
- 支持唯一索引查找模式

### 参数说明

| 名称 | 输入/输出 | 类型 | 说明 |
| --- | --- | --- | --- |
| `linear_cache_indices` | 输入 | `Tensor[int32/int64]` | 线性缓存索引，1D 张量 |
| `lxu_cache_state` | 输入 | `Tensor` | 缓存状态，2D 张量 |
| `invalid_index` | 输入 | `int64_t` | 无效索引值 |
| `gather_cache_stats` | 输入 | `bool` | 是否收集缓存统计信息 |
| `uvm_cache_stats` | 输入 | `Tensor?` | UVM 缓存统计信息（可选） |
| `num_uniq_cache_indices` | 输入 | `Tensor?` | 唯一缓存索引数量（可选） |
| `lxu_cache_locations_output` | 输入 | `Tensor?` | 缓存位置输出张量（可选） |
| `lxu_cache_locations` | 输出 | `Tensor[int32]` | 缓存位置索引 |

## 参数约束

- `linear_cache_indices` 必须是 1D 张量
- `lxu_cache_state` 必须是 2D 张量
- 所有输入张量必须位于 NPU 设备上
- 当 `gather_cache_stats=true` 时，`uvm_cache_stats` 不能为空
- `uniq_lookup` 和 `gather_cache_stats` 不能同时为 true
- 如果打开gather_cache_stats，那么必须提供uvm_cache_stats
- gather_cache_stats当前仅支持false，目前暂不支持uvm相关功能。故gather_cache_stats和uvm_cache_stats当前为保留参数，暂不支持相关功能

## 调用示例

```python
import torch
import fbgemm_ascend

# 创建输入张量
linear_cache_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="npu:0")
lxu_cache_state = torch.randint(0, 100, (4, 10), dtype=torch.int64, device="npu:0")
invalid_index = -1

# 调用算子
lxu_cache_locations = torch.ops.fbgemm.lxu_cache_lookup(
    linear_cache_indices,
    lxu_cache_state,
    invalid_index,
    gather_cache_stats=False
)
```

## 编译与测试

- Ascend C 算子编译与适配层编译参考仓库根目录 [README.md](../../../README.md)。
- 测试示例参考 [bench/split_embeddings_cache/lxu_cache_lookup_test/test_lxu_cache_lookup.py](../../../bench/split_embeddings_cache/lxu_cache_lookup_test/test_lxu_cache_lookup.py)。
