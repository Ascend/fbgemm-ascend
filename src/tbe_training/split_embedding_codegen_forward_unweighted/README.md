# split_embedding_codegen_forward_unweighted

仅支持 NPU 调用。该接口是 Table Batched Embedding training 场景的底层多表查表算子，为内部查表接口，不建议直接调用，推荐通过 `torchrec` / `fbgemm_gpu` 上层封装使用。

## 目录结构

```text
split_embedding_codegen_forward_unweighted
|-- split_embedding_codegen_forward_unweighted.cpp
|-- split_embedding_codegen_forward_unweighted.h
|-- split_embedding_codegen_common_utils.h
|-- README.md
|-- c310/
|   |-- split_embedding_codegen_forward_unweighted.json
|   |-- op_host/
|   |-- op_kernel/
|   `-- run.sh
`-- v220/
    |-- split_embedding_codegen_forward_unweighted.json
    |-- op_host/
    |-- op_kernel/
    `-- run.sh
```

## 硬件支持

| 实现目录 | 典型硬件 |
| --- | --- |
| `c310/` | Atlas A5 训练系列 |
| `v220/` | Atlas A2 / A3 训练系列 |

## PyTorch 底层接口原型

```python
torch.ops.fbgemm.split_embedding_codegen_forward_unweighted_cuda(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor D_offsets,
    SymInt total_D,
    SymInt max_D,
    Tensor indices,
    Tensor offsets,
    int pooling_mode,
    Tensor lxu_cache_locations,
    Tensor uvm_cache_stats,
    int output_dtype,
    bool is_experimental,
    Tensor? hash_indices=None,
    Tensor? offset_per_key=None,
    Tensor? rows_per_table=None,
) -> Tensor
```

## 功能说明

该算子根据多张 embedding 表的权重布局、索引和 offsets 完成前向查表：

- `pooling_mode = 0` 时执行 sum pooling。
- `pooling_mode = 1` 时执行 mean pooling。
- `pooling_mode = 2` 时执行 no-bag 查表。

输出形状：

- sum / mean: `[batch_size, total_D]`
- no-bag: `[len(indices), max_D]`

其中 `batch_size = (offsets.numel() - 1) / weights_offsets.numel()`。

## 主要参数

|  名称  |  输入/输出  |  数据类型  |  数据格式  |  范围  |  说明  |
|  ---- |  ---- |  ----  |  ----  |  ----  |  ----  |
|  dev_weights | 输入 | float32 | [total_table_size] | NA | 一维数组,所有表的权重，表的embedding_dim必须为8的整数倍 |
|  uvm_weight | 输入 | float32 | NA | NA  | 保留参数|
|  lxu_cache_weight | 输入 | float32 | NA | NA | 保留参数 |
|  weights_pacements | 输入 | int32 | NA | NA | 保留参数 |
|  weights_offsets | 输入 | int64 | [feat_cnt] | feat_cnt >= table_num, | 一维数组 |
|  D_offsets | 输入 | int64 | [feat_cnt + 1] | 数值必须从0开始依次递增 | 每个特征的embedding_dim的累加和 |
|  indices | 输入 | int64 | NA | len(indices) = offset[-1], 每张表的索引的大小[0, num_embedding] | 查表索引，需用户自行保证合法性，否则可能导致算子执行失败 |
|  hash_indices | 输入 | int64 | NA | len(indices) = offset[-1], 每张表的索引的大小[0, num_embedding] | 查表索引，需用户自行保证合法性，否则可能导致算子执行失败 |
|  offsets | 输入 | float32/int64 | [ [feat_cnt * batch_size + 1]] | 数值必须从0开始依次递增 | 查表索引对应的偏移 |
|  lxu_cache_locations | 输入 | int32 | NA | NA |保留参数 |
|  total_D | 属性 | int64 |  NA | NA | 所有特征的embedding_dim之和 |
|  max_D | 属性 | int64 | NA | NA | 最大的embedding_dim |
|  pooling_mode | 属性 | int64 | NA  | NA | poolingSum:0, poolingMean:1, poolingNone:2 |
|  output_dtype | 属性 | int64 | NA | NA |保留参数 |
|  is_experimental | 属性 | bool |  NA | NA | 保留参数 |
|  out | 输出 | float32 |poolingSum/poolingMean: [batch_size, total_D] poolingNone:[len(indices), maxD] | NA | NA |

其余参数如 `uvm_weights`、`lxu_cache_weights`、`weights_placements`、`lxu_cache_locations`、`uvm_cache_stats`、`output_dtype`、`is_experimental` 由上层 embedding 框架维护，直接调用时通常使用占位 tensor 或默认配置。

## 推荐上层调用方式

```python
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)
from torchrec import ComputeDevice

embedding_specs = [
    (98, 16, EmbeddingLocation.DEVICE, ComputeDevice.NPU),
    (14, 16, EmbeddingLocation.DEVICE, ComputeDevice.NPU),
    (20, 16, EmbeddingLocation.DEVICE, ComputeDevice.NPU),
]

tbe = SplitTableBatchedEmbeddingBagsCodegen(
    embedding_specs,
    optimizer=EmbOptimType.EXACT_SGD,
    device=torch.device("npu:0"),
    pooling_mode=PoolingMode.SUM,
)

output = tbe(indices, offsets)
loss = torch.sum(output ** 2 / 2)
loss.backward()
```

## 约束

- `weights_offsets.size(0) > 0`，`indices.numel() > 0`，`offsets.size(0) > 1`。
- `offsets` 需要满足 `(offsets.size(0) - 1) % weights_offsets.size(0) == 0`。
- embedding dim 需与算子实现约束一致，要求是 8 的倍数。

## 编译与测试

- Ascend C 算子编译与适配层编译参考仓库根目录 [README.md](../../../README.md)。
- 上层调用与精度测试可参考 [bench/tbe_training/split_embedding_codegen_lookup_adagrad_function_test/test_split_embedding_codegen_lookup_function.py](../../../bench/tbe_training/split_embedding_codegen_lookup_adagrad_function_test/test_split_embedding_codegen_lookup_function.py)。
