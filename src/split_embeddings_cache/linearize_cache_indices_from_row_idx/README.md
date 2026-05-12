# **linearize_cache_indices_from_row_idx**

本算子仅支持NPU调用。

## linearize_cache_indices_from_row_idx算子目录层级
```shell
-- linearize_cache_indices_from_row_idx
   |-- c310
      |-- op_host                                          # 算子host侧实现
      |-- op_kernel                                        # 算子kernel侧实现
      |-- linearize_cache_indices_from_row_idx.json        # 算子原型配置
      |-- README.md                                        # 算子说明文档
      |-- run.sh                                           # 算子编译部署脚本
```
## 硬件支持情况
| 实现目录              | 典型硬件                  |
| -------------------- | ------------------------ |
| c310/     | Atlas A5 训练系列     |

## 接口定义
```
torch.ops.fbgemm. linearize_cache_indices_from_row_idx(
           Tensor cache_hash_size_cumsum, 
           Tensor update_table_indices, 
           Tensor update_row_indices
           ) -> Tensor
```
## 功能说明

- 用于将稀疏嵌入表更新中的（表索引，行索引）二元组线性化为全局扁平化缓存中的绝对索引
- 支持合法索引映射（`cumsum[tableId] + rowId`）/ 非法索引哨兵值填充（`cumsum[-1]`）两种模式

### 仿真/伪代码

```python
def linearize_cache_indices_from_row_idx(cache_hash_size_cumsum, update_table_indices, update_row_indices):
    sentinel = cache_hash_size_cumsum[-1]
    linear_cache_indices = []
    for i in range(len(update_table_indices)):
        table_id = update_table_indices[i]
        row_id = update_row_indices[i]
        offset = cache_hash_size_cumsum[table_id]
        if offset >= 0 and row_id >= 0:
            linear_cache_indices.append(offset + row_id)
        else:
            linear_cache_indices.append(sentinel)
    return linear_cache_indices
```

## 简述主流程

1. 读取 `cache_hash_size_cumsum` 的最后一个元素作为哨兵值（全局缓存总行数）
2. 对每条更新记录，以 `update_table_indices[i]` 为下标查表得到该表在全局缓存中的起始偏移
3. 若偏移和行索引均 `>= 0`，输出 `offset + rowId`；否则输出哨兵值

## 参数说明

|  名称  |  输入/输出  |  数据类型  |  数据格式  |  说明  |
|  ---- |  ---- |  ----  |  ----  |  ----  |
|  cache_hash_size_cumsum | 输入 | int64 | ND [num_tables + 1] | 各嵌入表哈希大小的前缀和数组，最后一个元素为全局缓存总行数（哨兵值） |
|  update_table_indices | 输入 | int32，int64 | ND [total_updates] | 每条更新记录对应的嵌入表索引，负值表示无效 |
|  update_row_indices | 输入 | int32, int64 | ND [total_updates] | 每条更新记录在对应嵌入表内的行索引，负值表示无效 |
|  linear_cache_indices | 输出 | int32, int64 | ND [total_updates] | 每条更新记录在全局扁平化缓存中的线性绝对索引；非法记录输出哨兵值，dtype 与 update_row_indices 一致 |


### 参数约束
- `update_table_indices` 和 `update_row_indices` 必须为一维张量，且长度相等
- `cache_hash_size_cumsum` 为一维张量，长度为 `num_tables + 1`，表示前缀和数组
- dtype 仅支持 int32/int64
- 当 `update_row_indices` 长度为 0 时，直接返回空张量

### 算子调用示例
```python
import torch
import torch_npu
import fbgemm_ascend

def test_linearize_cache_indices_from_row_idx():
    # 假设有 3 张嵌入表，大小分别为 [100, 200, 200]
    # cache_hash_size_cumsum 为各表哈希大小的前缀和，最后一个元素 500 为哨兵值（全局缓存总行数）
    cache_hash_size_cumsum = torch.tensor([0, 100, 300, 500], dtype=torch.long, device='npu')

    # update_table_indices：每条更新记录对应的嵌入表索引
    # update_row_indices：每条更新记录在对应嵌入表内的行索引
    # 最后一条记录 table_id=-1 表示非法，输出将被填充为哨兵值
    update_table_indices = torch.tensor([0, 1, 2, -1], dtype=torch.int32, device='npu')
    update_row_indices   = torch.tensor([5, 10, 50, 0], dtype=torch.int32, device='npu')

    # 线性化缓存索引
    linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices_from_row_idx(
        cache_hash_size_cumsum,
        update_table_indices,
        update_row_indices,
    )

    # 验证结果：预期输出 [5, 110, 350, 500]
    #   i=0: cumsum[0] + 5  = 0   + 5   = 5
    #   i=1: cumsum[1] + 10 = 100 + 10  = 110
    #   i=2: cumsum[2] + 50 = 300 + 50  = 350
    #   i=3: table_id=-1 非法 → sentinel = 500
    print(linear_cache_indices)
```

## 编译与测试
- Ascend C 算子编译与适配层编译参考仓库根目录 README.md
   - 测试示例参考：bench/.../test_xxx.py（或 test/...xxx.py