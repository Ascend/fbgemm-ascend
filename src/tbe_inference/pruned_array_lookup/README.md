# pruned_array_lookup

本算子仅支持NPU调用。通过连续数组进行索引剪枝。

# 目录结构

```shell
-- pruned_array_lookup
   |-- c310
      |-- op_host                 # 算子host侧实现
      |-- op_kernel               # 算子kernel侧实现
      |-- pruned_array_lookup.json    # 算子原型配置
      |-- README.md               # 算子说明文档
      |-- run.sh                  # 算子编译部署脚本
```

# 产品支持情况

| 实现目录              | 典型硬件                  |
| -------------------- | ------------------------ |
| c310/  | Atlas A5训练系列产品  |

# 接口定义

```python
torch.ops.fbgemm.pruned_array_lookup(indices: torch.Tensor, offsets: torch.Tensor, index_remappings: torch.Tensor, index_remappings_offsets: torch.Tensor) -> torch.Tensor
```

# 功能说明

`pruned_array_lookup`算子用于在嵌入表剪枝后的列表中查找原始稀疏索引对应的致密索引。

## 算子实现原理

将indices中数据按batch区分，并定位到batch所在table对应的列表数据，遍历每个索引并使用SIMT多线程模式查找对应的致密索引。

功能逻辑的python伪代码如下：

```python
def pruned_array_lookup_torch_vectorized(indices, offsets, index_remappings, index_remappings_offsets):
    # 计算表数量T和批次数量B
    T = index_remappings_offsets.size(0) - 1  # 表的数量
    B = (offsets.size(0) - 1) // T      # 每个表的批次数量

    # 初始化输出张量，形状与indices相同，类型与indices相同
    dense_indices = torch.empty_like(indices)
    dense_indices.fill_(-1)

    # 遍历每一个batch
    for i in range(B):
        indices_start = offsets[i]
        indices_end = offsets[i + 1]
        table_idx = i / B
        index_remappings_start = index_remappings_offsets[table_idx]
        index_remappings_end = index_remappings_offsets[table_idx + 1]

        # 如果table对应的哈希表数据为空，则表示不进行剪枝，直接输出原稀疏索引
        if index_remappings_start == index_remappings_end:
            for j in range(indices_start, indices_end):
                dense_indices[j] = indices[j]
            continue

        # 遍历batch中每个index
        for j in range(indices_start, indices_end):
            sparse_idx = indices[j]
            dense_indices[j] = index_remappings[index_remappings_start + sparse_idx]

    return dense_indices
```

# 算子输入与输出

| 名称 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | 说明 |
|---|---|---|---|---|---|
| indices | 输入 | Tensor | int32/int64 | [T \* B \* L,] | 一维tensor，表示多个表的稀疏索引。其中每个元素为一个稀疏索引，用于在index_remappings中查找对应的致密索引。<br>索引值需要大于等于0。<br>T为表的数量，B为每个表包含多少个batch的index，L为每个batch中index数量。<br>多个表之间，每个表的batch数量必须相同。<br>每个表内的indices索引必须是unique的（单个表内部不能存在重复索引）。<br>不同表可以拥有不同数量的indices，即不同batch的L值可以有差异，但可能会负载不均衡而导致影响性能。<br>tensor中元素值需要小于index_remappings中表对应的元素个数。 |
| offsets | 输入 | Tensor | int32/int64 | [T \* B + 1,]  | 一维tensor，表示每个batch对应稀疏索引的偏移。<br>其中第一个元素为0，后续元素为每个batch对应稀疏索引数量的累加和。<br>数据类型和indices一致。 |
| index_remappings | 输入 | Tensor | int32/int64 | [original_E * T]  | 一维tensor，表示多个表的稀疏索引对应的待重映射的值。<br>长度需小于int64类型最大值。<br>支持多个致密索引表长度不相等。<br>允许有稀疏表对应的致密索引数量为0，代表不对该表的稀疏索引做剪枝操作，输出的致密索引为原稀疏索引值。<br>original_E表示剪枝前的稀疏表大小。 |
| index_remappings_offsets | 输入 | Tensor | int64 | [T + 1,]  | 一维tensor，表示每个表对应index_remappings中致密索引的偏移，长度为表的个数+1。<br>第一个元素为0，后续每个元素为index_remappings中每个表的致密索引数量的累加和。<br>其中第i个数据必须<=第i+1个数据，相等时代表该稀疏表对应的致密索引数量为0，表示不对该表的稀疏索引做剪枝操作，输出的致密索引和原稀疏索引相同。<br>`(offsets.size(0) - 1) % (index_remappings_offsets.size(0) - 1)`的值必须为0，后者表示表数量，前者需要为表数量的整数倍。 |
| dense_indices | 输出 | Tensor | int32/int64 | [T \* B \* L,]  | 一维tensor，稀疏索引转换后的致密索引。<br>数据类型和indices一致。 |

# 调用示例

```python
indices: torch.Tensor
offsets: torch.Tensor
index_remappings: torch.Tensor
index_remappings_offsets: torch.Tensor
dense_indices = torch.ops.fbgemm.pruned_array_lookup(indices, offsets, index_remappings, index_remappings_offsets)
```

# 编译与测试

算子编译请参考[README.md](../../../README.md)中"源码编译与安装"章节。

算子测试用例请参考[test](../../../test/tbe/utils/split_embeddings_utils_test.py)。
