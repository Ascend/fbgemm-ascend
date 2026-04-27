# 使用PyTorch框架调用pruned_hashmap_lookup算子

该样例基于 PyTorch 2.7.1 和 Python 3.11.0 运行。

## 算子调用示例

```python
import sysconfig
import numpy as np

import torch
import torch_npu
import fbgemm_gpu
import fbgemm_ascend


LOAD_FACTOR = 0.8
PRUNING_RATIO = 0.5

DEVICE = "npu:0"
np.random.seed(42)
table_num: int = 10
batch_num: int = 10
length: int = 100
param_types: list[torch.dtype] = [torch.int32, torch.int64]
current_device = torch.device(DEVICE)
indices_type, hash_table_offsets_type = param_types

# 稀疏索引的值的范围
sparse_idx_range = int(batch_num * length / (1.0 - PRUNING_RATIO))  
idx_type_max = torch.iinfo(indices_type).max
_assume(
    sparse_idx_range < idx_type_max,
    f"sparse_idx_range must be less than indices_type:{indices_type} max, "
    f"indices type max:{idx_type_max}, sparse_idx_range:{sparse_idx_range}.",
)

# 生成唯一的indices
indices = torch.empty(size=(table_num, batch_num, length), dtype=indices_type)
for t in range(table_num):
    np_table = np.random.choice(
        np.arange(sparse_idx_range, dtype=np.int64),
        size=(batch_num, length),
        replace=False,
    )
    indices[t] = torch.tensor(np_table, dtype=indices_type)
indices = indices.view(-1)

# 创建offsets
offsets = torch.tensor([length * b_t for b_t in range(batch_num * table_num + 1)]).to(dtype=indices_type)

# 生成致密索引
dense_idx_range = int(batch_num * length / (1.0 - PRUNING_RATIO + 0.2))  # 致密索引范围小于稀疏索引范围
dense_indices = (
    torch.randint(low=0, high=dense_idx_range, size=(table_num, batch_num, length)).view(-1).to(dtype=indices_type)
)

# 初始化hash_table和对应offsets
# hash_table 中每个致密索引表的大小
capacities = [int(batch_num * length / LOAD_FACTOR) for _ in range(table_num)]
hash_table = torch.full(
    (sum(capacities), 2),
    -1,  # 填充-1
    dtype=indices_type,
)
hash_table_offsets = torch.tensor([0] + np.cumsum(capacities).tolist()).to(dtype=hash_table_offsets_type)

# 将生成的dense_indices插入到hash_table中，调用fbgemm的CPU实现
torch.ops.fbgemm.pruned_hashmap_insert(
    indices, dense_indices, offsets, hash_table, hash_table_offsets
)

indices = indices.to(current_device)
dense_indices = dense_indices.to(current_device)
offsets = offsets.to(current_device)
hash_table = hash_table.to(current_device)
hash_table_offsets = hash_table_offsets.to(current_device)

# 查表致密索引
dense_indices_lookup = torch.ops.fbgemm.pruned_hashmap_lookup(indices, offsets, hash_table, hash_table_offsets)
```

## 编译与部署

算子编译与部署请参考 [README.md](../../../README.md) 中 "源码编译与安装" 章节。

> **提示**
> 以上示例仅展示基本用法，如需更全面的测试用例，请参考完整测试文件：[test](../../../bench/tbe_inference/pruned_hashmap_lookup/test_pruned_hashmap_lookup.py)。
