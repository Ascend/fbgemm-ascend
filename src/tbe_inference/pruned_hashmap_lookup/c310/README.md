# pruned_hashmap_lookup算子说明

本算子仅支持NPU调用。

# 产品支持情况

| 硬件型号              | 是否支持                  |
| -------------------- | ------------------------ |
| Atlas A5训练系列产品  | 是  |

# pruned_hashmap_lookup算子目录层级

```shell
-- pruned_hashmap_lookup
   |-- c310
      |-- op_host                 # 算子host侧实现
      |-- op_kernel               # 算子kernel侧实现
      |-- pruned_hashmap_lookup.json    # 算子原型配置
      |-- README.md               # 算子说明文档
      |-- run.sh                  # 算子编译部署脚本
```

# 功能

`pruned_hashmap_lookup`算子用于在嵌入表剪枝后的哈希表中查找原始稀疏索引对应的致密索引。

该算子通过哈希查找机制，将原始的稀疏索引映射到经过剪枝优化后的致密索引空间，从而减少内存占用和计算开销，提升推理性能。

# 算子实现原理

将indices中数据按batch区分，并定位到batch所在table对应的哈希表数据，遍历每个索引并使用SIMT多线程线性探测哈希表，查找对应的致密索引。SIMT多线程可以并行处理多个索引，以及并行探测哈希表多个槽位，提高查找效率。

python伪代码如下：

```python
def pruned_hashmap_lookup_torch_vectorized(indices, offsets, hash_table, hash_table_offsets):
    # 计算表数量T和批次数量B
    T = hash_table_offsets.size(0) - 1  # 表的数量
    B = (offsets.size(0) - 1) // T      # 每个表的批次数量
    
    assert B > 0, "B must be greater than 0"
    
    # 初始化输出张量，形状与indices相同，类型与indices相同
    dense_indices = torch.empty_like(indices)
    dense_indices.fill_(-1)  # 初始化为-1，表示未找到
    
    # 遍历每一个batch
    for i in range(B):
        indices_start = offsets[i]
        indices_end = offsets[i + 1]
        table_idx = i / B
        table_indices_start = hash_table_offsets[table_idx]
        table_indices_end = hash_table_offsets[table_idx + 1]

        # 如果table对应的哈希表数据为空，则表示不进行剪枝，直接输出原稀疏索引
        if table_indices_start == table_indices_end:
            for j in range(indices_start, indices_end):
                dense_indices[j] = indices[j]
            continue
        
        # 遍历batch中每个index
        for j in range(indices_start, indices_end):
            sparse_idx = indices[j]
            for k in range(table_indices_start, table_indices_end):
                table_sparse_idx = int(hash_table[k, 0].item())
                table_dense_idx = int(hash_table[k, 1].item())
                if sparse_idx == table_sparse_idx:
                    dense_indices[j] = table_dense_idx
                break
        
    return dense_indices
```

# 算子输入与输出

| 名称 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | 说明 |
|---|---|---|---|---|---|
| indices | 输入 | Tensor | int32/int64 | [T \* B \* L,] | 一维tensor，表示多个表的稀疏索引。其中每个元素为一个稀疏索引，用于在hash_table中查找对应的致密索引。<br>T为表的数量，B为每个表包含多少个batch的index，L为每个batch中index数量。<br>多个表之间，每个表的batch数量必须相同。<br>每个表内的indices索引必须是unique的（单个表内部不能存在重复索引）。<br>不同表可以拥有不同数量的indices，即不同batch的L值可以有差异，但可能会负载不均衡而导致影响性能。 |
| offsets | 输入 | Tensor | int32/int64 | [T \* B + 1,]  | 一维tensor，表示每个batch对应稀疏索引的偏移。<br>其中第一个元素为0，后续元素为每个batch对应稀疏索引数量的累加和。<br>数据类型和indices一致。 |
| hash_table | 输入 | Tensor | int32/int64 | [x, 2]  | 二维tensor，表示多个表的稀疏索引和致密索引的映射关系。<br>第二维中，第一个元素为表的稀疏索引，第二个元素为稀疏索引对应的致密索引。<br>x需小于int32类型最大值。<br>支持多个致密索引表长度不相等。<br>允许有稀疏表对应的致密索引数量为0，代表不对该表的稀疏索引做剪枝操作，输出的致密索引为原稀疏索引值。<br>数据类型和indices一致。<br>hash_table中每个表需要至少一个空槽位（即每个表中至少有一行数据，第二维的第一个元素为-1）。 |
| hash_table_offsets | 输入 | Tensor | int64 | [T + 1,]  | 一维tensor，表示每个表对应hash_table中致密索引的偏移，长度为表的个数+1。<br>第一个元素为0，后续每个元素为hash_table中每个表的致密索引数量的累加和。<br>其中第i个数据必须<=第i+1个数据，相等时代表该稀疏表对应的致密所有数量为0，表示不对该表的稀疏索引做剪枝操作，输出的致密索引和原稀疏索引相同。 |
| dense_indices | 输出 | Tensor | int32/int64 | [T \* B \* L,]  | 一维tensor，稀疏索引转换后的致密索引。<br>数据类型和indices一致。 |

# 算子编译部署

算子编译请参考[README.md](../../../../README.md)中"源码编译与安装"章节。

注：详细算子调用示例参考Pytorch框架下[README.md](../README.md)。
