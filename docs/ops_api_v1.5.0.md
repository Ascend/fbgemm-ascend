# Fbgemm-Ascend 算子列表 (v1.5.0)

Fbgemm-Ascend 是 FBGEMM 算子在昇腾 NPU 平台上的算子实现，通过 `torch.ops.fbgemm.*` 提供高性能稀疏/稠密算子，帮助推荐、搜索等场景在 Ascend 设备上获得与 GPU 同步的训练体验。

## 简介

| 模块                                                                 | 文档                                                                                        | 功能介绍                                                                                                                                      | 硬件支持       |
|--------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|------------|
| src/embedding_inplace_ops/pruned_array_lookup_from_row_idx         | [README](../src/embedding_inplace_ops/pruned_array_lookup_from_row_idx/README.md)         | 从row_idx中查找对应的pruned_array元素，实现嵌入表剪枝后的行索引查找。                                                                                              | A5         |
| src/embedding_inplace_ops/emb_inplace_update | [README](../src/embedding_inplace_ops/emb_inplace_update/README.md)                       | 将稀疏更新中的若干 (table_id, row_id, payload) 三元组按字节级 scatter 写回 TBE 格式的扁平化权重张量，用于 INT-NBit TBE 推理场景下的权重原地更新。 | A5 |
| src/intraining_embedding_pruning_ops/init_address_lookup           | [README](../src/intraining_embedding_pruning_ops/init_address_lookup/README.md)           | 初始化地址查找表，用于训练中嵌入剪枝（In-training Embedding Pruning）。                                                                                        | A5         |
| src/jagged_tensor_ops/dense_to_jagged                              | [README](../src/jagged_tensor_ops/dense_to_jagged/v220/README.md)                         | 将密集三维张量(dense Tensor)转换为锯齿状二维张量(jagged Tensor)，用于处理变长序列数据。                                                                                | A2, A3, A5 |
| src/jagged_tensor_ops/jagged_dense_elementwise_binary_jagged_output                     | [README](../src/jagged_tensor_ops/jagged_dense_elementwise_binary_jagged_output/README.md)                     | 将jagged tensor与dense tensor进行逐元素加法或者乘法，最终输出jagged tensor                                                                  | A5 |
| src/jagged_tensor_ops/jagged_dense_elementwise                     | [README](../src/jagged_tensor_ops/jagged_dense_elementwise/README.md)                     | 将jagged tensor转换为与dense tensor同形状的padded dense tensor后，与dense tensor进行逐元素加法。                                                                    | A5 |
| src/jagged_tensor_ops/jagged_to_padded_dense                       | [README](../src/jagged_tensor_ops/jagged_to_padded_dense/v220/README.md)                  | 实现将jagged tensor转为padded dense tensor功能。                                                                                                  | A2, A3, A5 |
| src/jagged_tensor_ops/jagged_to_padded_dense_v2                    | [README](../src/jagged_tensor_ops/jagged_to_padded_dense_v2/README.md)                    | 将jagged tensor转为padded dense tensor的v2版本，增加了多offsets场景的功能。                                                                                | A2, A3, A5 |
| src/jagged_tensor_ops/jagged_1d_to_dense                           | [README](../src/jagged_tensor_ops/jagged_1d_to_dense/README.md)                           | 将1D jagged tensor转为padded dense tensor。                                                                                                   | A2, A3, A5 |
| src/jagged_tensor_ops/jagged_2d_to_dense                           | [README](../src/jagged_tensor_ops/jagged_2d_to_dense/README.md)                           | 将2D jagged tensor转为padded dense tensor。                                                                                                   | A2, A3, A5 |
| src/jagged_tensor_ops/select_dim1_to_permute                       | [README](../src/jagged_tensor_ops/select_dim1_to_permute/c310/README.md)                  | 将元素个数为batch_size的select_dim1 通过attr属性扩展到多个batch_num（lengthsSize / batchSize），生成permute，同时根据select_dim1中的元素值，将permute中的元素值对应到lengths中的元素值。 | A5         |
| src/merge_pooled_embedding_ops                                     | [README](../src/merge_pooled_embedding_ops/README.md)                                     | 实现将多个NPU设备上的张量合并到目标设备上（all_to_one_device）。                                                                                                | A5         |
| src/pooled_embedding_ops/permute_pooled_embs                       | [README](../src/pooled_embedding_ops/permute_pooled_embs/v220/README.md)                  | permute_pooled_embs 是用于对池化后的嵌入（pooled embeddings）输出进行特征维度重排列的核心算子。                                                                        | A2, A3, A5 |
| src/quantize_ops/bfloat16_quantized_to_float                       | [README](../src/quantize_ops/bfloat16_quantized_to_float/README.md)                       | 将bfloat16量化格式的嵌入表权重反量化为float32格式，用于推理时的精度恢复。                                                                                              | A5         |
| src/quantize_ops/float_or_half_to_fused_nbit_rowwise               | [README](../src/quantize_ops/float_or_half_to_fused_nbit_rowwise/README.md)               | 将float16/float32张量转换为融合的n-bit行级量化格式，用于嵌入表的量化压缩。                                                                                           | A5         |
| src/quantize_ops/float_to_bfloat16_quantized                       | [README](../src/quantize_ops/float_to_bfloat16_quantized/README.md)                       | 将float32张量量化为bfloat16量化格式，减少嵌入表存储空间。                                                                                                      | A5         |
| src/sparse_ops/asynchronous_complete_cumsum                        | [README](../src/sparse_ops/asynchronous_complete_cumsum/README.md)                        | 对输入的一维Tensor累积求和。                                                                                                                         | A2, A3, A5 |
| src/sparse_ops/block_bucketize_sparse_features                     | 无                                                                                         | 将稀疏特征的索引按block大小进行分桶，并计算分桶后每个样本的新长度。                                                                                                      | A5         |
| src/sparse_ops/expand_into_jagged_permute                          | [README](../src/sparse_ops/expand_into_jagged_permute/README.md)                          | 用于将稀疏数据置换索引从表维度扩展到批次维度。                                                                                                                   | A5         |
| src/sparse_ops/group_index_select_dim0                             | [README](../src/sparse_ops/group_index_select_dim0/c310/README.md)                        | 实现批量从多个数据表中按行号挑选数据的功能，等价于对每个表独立执行torch.index_select(input, 0, indices)操作。                                                                 | A5         |
| src/sparse_ops/group_index_select_dim0_backward                    | [README](../src/sparse_ops/group_index_select_dim0_backward/c310/README.md)               | 实现group_index_select_dim0前向算子的反向传播，根据输出梯度计算输入梯度。                                                                                          | A5         |
| src/sparse_ops/invert_permute                                      | [README](../src/sparse_ops/invert_permute/c310/README.md)                                 | 对输入张量进行逆置换操作。                                                                                                                             | A5         |
| src/sparse_ops/offsets_range                                       | [README](../src/sparse_ops/offsets_range/README.md)                                       | 根据offsets生成分段内局部下标。                                                                                                                       | A2, A3, A5 |
| src/sparse_ops/permute2d_sparse_data                               | [README](../src/sparse_ops/permute2d_sparse_data/v220/README.md)                          | 对二维稀疏数据进行重排。                                                                                                                              | A2, A3, A5 |
| src/sparse_ops/segment_sum_csr                                     | [README](../src/sparse_ops/segment_sum_csr/README.md)                                     | 根据batch_size和csr_seg对values各个分段求和。                                                                                                        | A2, A3, A5 |
| src/split_embeddings_cache/get_unique_indices                      | [README](../src/split_embeddings_cache/get_unique_indices/README.md)                      | 对输入的cache索引进行去重，返回唯一的索引列表，用于LRU缓存查找前的预处理。                                                                                                 | A5         |
| src/split_embeddings_cache/linearize_cache_indices                 | [README](../src/split_embeddings_cache/linearize_cache_indices/README.md)                 | 将多维cache索引线性化为一维索引，便于后续LRU缓存操作。                                                                                                           | A5         |
| src/split_embeddings_cache/linearize_cache_indices_from_row_idx    | [README](../src/split_embeddings_cache/linearize_cache_indices_from_row_idx/README.md)    | 从row_idx出发对cache索引进行线性化，将表索引转换为全局cache地址。                                                                                                 | A5         |
| src/split_embeddings_cache/lru_cache_find_uncached                 | [README](../src/split_embeddings_cache/lru_cache_find_uncached/README.md)                 | 在LRU缓存中查找未缓存的索引项，返回需要从UVM加载的索引列表。                                                                                                         | A5         |
| src/split_embeddings_cache/lru_cache_insert_byte                   | [README](../src/split_embeddings_cache/lru_cache_insert_byte/README.md)                   | 将UVM中的量化字节权重写入LRU缓存权重区，并更新缓存状态和LRU状态。                                                                                                     | A5         |
| src/split_embeddings_cache/lru_cache_populate_byte                 | [README](../src/split_embeddings_cache/lru_cache_populate_byte/README.md)                 | LRU缓存填充的主入口算子，组合调用lru_cache_find_uncached和lru_cache_insert_byte完成缓存填充全流程。                                                                 | A5         |
| src/split_embeddings_cache/direct_mapped_lru_cache_find_uncached   | [README](../src/split_embeddings_cache/direct_mapped_lru_cache_find_uncached/README.md)   | Direct-mapped方式在LRU缓存中查找未缓存项，通过atomicMax竞争机制解决direct-mapped冲突，返回每个索引对应的cache set。                                                              | A5         |
| src/split_embeddings_cache/direct_mapped_lru_cache_insert_byte     | [README](../src/split_embeddings_cache/direct_mapped_lru_cache_insert_byte/README.md)     | Direct-mapped方式将UVM中的量化字节权重写入LRU缓存权重区，并更新缓存状态和LRU时间戳。                                                                              | A5         |
| src/split_embeddings_cache/direct_mapped_lru_cache_populate_byte   | [README](../src/split_embeddings_cache/direct_mapped_lru_cache_populate_byte/README.md)   | Direct-mapped LRU缓存填充的主入口算子，组合调用direct_mapped_lru_cache_find_uncached和direct_mapped_lru_cache_insert_byte完成缓存填充全流程。                                         | A5         |
| codegen/inference | [README](../codegen/inference/README.md) | 对多精度嵌入表进行索引查找和池化，支持多种量化精度的推理场景。                                                                                                           | A5         |
| src/tbe_inference/pruned_hashmap_lookup                            | [README](../src/tbe_inference/pruned_hashmap_lookup/README.md)                            | 在嵌入表剪枝后的哈希表中查找原始稀疏索引对应的致密索引，通过哈希查找机制减少内存占用和计算开销。                                                                                          | A5         |
| src/tbe_inference/pruned_array_lookup | [README](../src/tbe_inference/pruned_array_lookup/README.md)                              | 在嵌入表剪枝后的列表中查找原始稀疏索引对应的致密索引，通过索引剪枝减少内存占用和计算开销。 | A5 |
| src/tbe_training/backward_codegen_adagrad_unweighted_exact         | [README](../src/tbe_training/backward_codegen_adagrad_unweighted_exact/README.md)         | 将上游梯度 grad_output 按索引散射回各 embedding 行，并用 Adagrad 优化器更新权重和动量。                                                                              | A2, A3, A5 |
| src/tbe_training/bounds_check_indices                              | [README](../src/tbe_training/bounds_check_indices/README.md)                              | 对嵌入表查表索引进行边界检查，防止越界访问导致的运行时错误。                                                                                                            | A5         |
| src/tbe_training/dense_embedding_codegen_lookup_function           | [README](../src/tbe_training/dense_embedding_codegen_lookup_function/v220/README.md)      | 根据 indices 和 offsets 从平铺的稠密权重表 dev_weights 中按行取出对应的 embedding 向量并拼接输出。                                                                    | A2, A3, A5 |
| src/tbe_training/dense_embedding_codegen_lookup_function_grad      | [README](../src/tbe_training/dense_embedding_codegen_lookup_function_grad/v220/README.md) | 将上游梯度 weights_grad 按查表索引散射累加回权重表 dev_weights 的对应行，完成梯度回传。                                                                                 | A2, A3, A5 |
| src/tbe_training/split_embedding_codegen_forward_unweighted        | [README](../src/tbe_training/split_embedding_codegen_forward_unweighted/README.md)        | 根据 indices/offsets 从多张嵌入表中取出对应行向量，并按 batch 和 feature 维度进行 sum 或 mean 池化，得到每个样本的稠密嵌入表示。                                                    | A2, A3, A5 |

更多算子可在对应目录的 README.md 中查看具体接口、输入输出张量格式及样例代码。

## 算子目录结构

```text
src/
├── embedding_inplace_ops/                                 # 嵌入表原地操作
│   ├── pruned_array_lookup_from_row_idx/                  # 剪枝后数组行索引查找
│   └── emb_inplace_update/                                # 嵌入表权重原地更新
├── intraining_embedding_pruning_ops/                      # 训练中嵌入剪枝
│   └── init_address_lookup/                               # 地址查找表初始化
├── jagged_tensor_ops/                                    # Jagged 张量操作
│   ├── dense_to_jagged/                                  # 稠密→Jagged 转换
│   ├── jagged_dense_elementwise/                         # Jagged 与 Dense 逐元素加法
│   ├── jagged_to_padded_dense/                           # Jagged→填充稠密 转换
│   ├── jagged_to_padded_dense_v2/                        # Jagged→填充稠密 转换 v2
│   └── select_dim1_to_permute/                           # dim1 维度选择置换
├── merge_pooled_embedding_ops/                           # 池化嵌入合并操作
│   └── all_to_one_device                                 # 多设备张量合并
├── pooled_embedding_ops/                                 # 池化嵌入操作
│   └── permute_pooled_embs/                              # 池化嵌入置换
├── quantize_ops/                                         # 量化操作
│   ├── bfloat16_quantized_to_float/                      # bfloat16量化→浮点 反量化
│   ├── float_or_half_to_fused_nbit_rowwise/              # 浮点→n-bit行级量化
│   └── float_to_bfloat16_quantized/                      # 浮点→bfloat16量化
├── sparse_ops/                                           # 稀疏操作
│   ├── asynchronous_complete_cumsum/                     # 异步完整累积和
│   ├── block_bucketize_sparse_features/                  # 稀疏特征分桶
│   ├── expand_into_jagged_permute/                       # 展开为 Jagged 置换
│   ├── group_index_select_dim0/                          # 分组索引选择(dim0)
│   ├── group_index_select_dim0_backward/                 # 分组索引选择(dim0)反向
│   ├── invert_permute/                                   # 逆置换
│   ├── offsets_range/                                    # 偏移量 Range 生成
│   ├── permute2d_sparse_data/                            # 2D/1D 稀疏数据置换
│   └── segment_sum_csr/                                  # CSR 分段求和
├── split_embeddings_cache/                               # Split 嵌入缓存
│   ├── get_unique_indices/                               # 缓存索引去重
│   ├── linearize_cache_indices/                          # 缓存索引线性化
│   ├── linearize_cache_indices_from_row_idx/             # 从row_idx线性化缓存索引
│   ├── lru_cache_find_uncached/                          # LRU缓存查找未缓存项
│   ├── lru_cache_insert_byte/                            # LRU缓存插入(字节)
│   ├── lru_cache_populate_byte/                          # LRU缓存填充(主入口)
│   ├── direct_mapped_lru_cache_find_uncached/            # Direct-mapped LRU缓存查找未命中项
│   ├── direct_mapped_lru_cache_insert_byte/              # Direct-mapped LRU缓存插入(字节)
│   └── direct_mapped_lru_cache_populate_byte/            # Direct-mapped LRU缓存填充(主入口)
├── tbe_inference/                                        # TBE 推理算子
│   ├── int_nbit_split_embedding_codegen_lookup_function/ # 整数量化嵌入推理查询
│   └── pruned_hashmap_lookup/                            # 剪枝哈希表查找
│   └── pruned_array_lookup/                              # 剪枝列表查找
└── tbe_training/                                         # TBE 训练算子
    ├── backward_codegen_adagrad_unweighted_exact/        # Adagrad 反向
    ├── bounds_check_indices/                             # 索引边界检查
    ├── dense_embedding_codegen_lookup_function/          # 稠密嵌入前向查询
    ├── dense_embedding_codegen_lookup_function_grad/     # 稠密嵌入反向梯度
    └── split_embedding_codegen_forward_unweighted/       # Split 嵌入前向查询
```
