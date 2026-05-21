# direct_mapped_lxu_cache_lookup

本算子仅支持 NPU 调用，适用于在NPU上的直接映射LRU/LFU缓存中执行查找操作。

## 目录结构

```text
direct_mapped_lxu_cache_lookup
|-- direct_mapped_lxu_cache_lookup.cpp
|-- README.md
|-- c310/
|   |-- direct_mapped_lxu_cache_lookup.json
|   |-- op_host/
|   |-- op_kernel/
|   `-- run.sh
```

## 硬件支持

| 实现目录 | 典型硬件 |
| --- | --- |
| `c310/` | Atlas A5 训练系列 |

## PyTorch 接口原型

```python
torch.ops.fbgemm.direct_mapped_lxu_cache_lookup(Tensor offset) -> Tensor
```

## 功能说明

- `direct_mapped_lxu_cache_lookup` 在直接映射的 LRU/LFU 缓存中查找指定索引，返回每个索引对应的缓存槽位位置，缺失时返回-1。

- 支持跳过index为哨兵值invalid_index时的查询。

- 支持可选的缓存统计信息收集，记录命中次数、未命中次数和总请求数。

示例输入：缓存状态lxu_cache_state存储key值[5, 10, 15, 20]分别位于 slot 0~3，查询索引indices = [5, 10, 99, 15]时：

```python
# 输出：每个索引命中的槽位位置，缺失返回 -1
result = [0, 1, -1, 2]
```

## 参数与约束

| 名称 | 输入/输出 | 类型 | 说明                                                |
| --- | --- | --- |---------------------------------------------------|
| `linear_cache_indices` | 输入 | `Tensor[int64/int32]` | 待查找的原始线性索引                                        |
| `lxu_cache_state` | 输入 | `Tensor[int64]` | 缓存状态表，记录当前缓存中存放的原始索引及其对应的槽位信息                     |
| `invalid_index` | 输入 | `int64` | 哨兵值，当出现该index时，跳过查询；默认为-1                         |
| `gather_cache_stats` | 输入 | `bool` | 是否收集缓存命中/未命中的统计信息；默认为false                        |
| `uvm_cache_stats` | 输入 | `Tensor[int32]` | 可选，当 gather_cache_stats=true 时提供，用于存放统计结果的 tensor |
| -- | 输出 | `Tensor[int64/int32]` | 每个输入索引对应的缓存槽位位置（slot）；若索引不在缓存中，则对应位置返回 invalid_index |


- 输入必须位于 NPU 上
- 如果打开gather_cache_stats，那么必须提供uvm_cache_stats
- gather_cache_stats当前仅支持false，目前暂不支持uvm相关功能。故gather_cache_stats和uvm_cache_stats当前为保留参数，暂不支持相关功能

## 调用示例

```python
import sysconfig
import torch
import torch_npu
import fbgemm_ascend

cache_state = torch.tensor([
    [0, 100],   # slot0: key=0, 元数据=100
    [1, 101],   # slot1: key=1, 元数据=101
    [2, 102],   # slot2: key=2, 元数据=102
    [3, 103],   # slot3: key=3, 元数据=103
], device='npu0')

indices = torch.tensor([0, 1, 2, 3], device='npu0')

result = torch.ops.fbgemm.direct_mapped_lxu_cache_lookup(
    linear_cache_indices=indices,
    lxu_cache_state=cache_state,
    invalid_index=-1,
    gather_cache_stats=False,
    uvm_cache_stats=None
)
```

## 编译与测试

- Ascend C 算子编译与适配层编译参考仓库根目录 [README.md](../../../README.md)。
- 测试示例参考 [bench/split_embeddings_cache/direct_mapped_lxu_cache_lookup_test/test_direct_mapped_lxu_cache_lookup.py](../../../bench/split_embeddings_cache/direct_mapped_lxu_cache_lookup_test/test_direct_mapped_lxu_cache_lookup.py)。
