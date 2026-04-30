# int_nbit_split_embedding_codegen_lookup_function

本算子仅支持 NPU 调用，用于 TBE inference 场景下量化 embedding 表的前向查表。

## 目录结构

```text
int_nbit_split_embedding_codegen_lookup_function
|-- int_nbit_split_embedding_codegen_lookup_function.cpp
|-- README.md
`-- c310/
    |-- int_nbit_split_embedding_codegen_lookup_function.json
    |-- op_host/
    |-- op_kernel/
    `-- run.sh
```

## 产品支持情况

| 实现目录 | 典型硬件 |
| --- | --- |
| `c310/` | Atlas A5 训练系列 |

## PyTorch 接口原型

```python
torch.ops.fbgemm.int_nbit_split_embedding_codegen_lookup_function(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    int total_D,
    int max_int2_D,
    int max_int4_D,
    int max_int8_D,
    int max_float16_D,
    int max_float32_D,
    Tensor indices,
    Tensor offsets,
    int pooling_mode,
    Tensor? indice_weights,
    int output_dtype,
    Tensor? lxu_cache_weights,
    Tensor? lxu_cache_locations,
    int? row_alignment=None,
    int? max_float8_D=None,
    int? fp8_exponent_bits=None,
    int? fp8_exponent_bias=None,
) -> Tensor
```

## 功能说明

实现 FBGEMM `int_nbit_split_embedding_codegen_lookup_function` 的前向查表，支持：

- bag 模式：`SUM` / `MEAN`
- nobag 模式：`NONE`
- 多种权重量化类型：`INT2` / `INT4` / `INT8` / `FP16` / `FP32` / `BF16` / `FP8`
- 多种输出类型：`FP32` / `FP16` / `BF16` / `INT8`

伪代码如下：

```python
def int_nbit_split_embedding(
    dev_weights, weights_offsets, weights_tys, D_offsets,
    indices, offsets, pooling_mode, indice_weights=None,
    output_dtype=SparseType.FP32,
):
    feat_cnt = len(weights_offsets)
    if pooling_mode == PoolingMode.NONE:
        outputs = []
        out_dim1 = max_row_dim(weights_tys)
        for t in range(feat_cnt):
            for idx in indices[offsets[t]: offsets[t + 1]]:
                row = load_quant_row(dev_weights, weights_offsets[t], idx)
                vec = dequantize_row(row)
                outputs.append(cast(vec, output_dtype))
        return stack(outputs)
    else:
        batch = (len(offsets) - 1) // feat_cnt
        result = zeros((batch, D_offsets[-1]), dtype=output_dtype)
        for t in range(feat_cnt):
            dim_start, dim_end = D_offsets[t], D_offsets[t + 1]
            for b in range(batch):
                start = offsets[t * batch + b]
                end = offsets[t * batch + b + 1]
                acc = zeros(dim_end - dim_start, dtype=float)
                for k, idx in enumerate(indices[start:end]):
                    row = load_quant_row(dev_weights, weights_offsets[t], idx)
                    vec = dequantize_row(row)
                    weight = indice_weights[start + k] if indice_weights is not None else 1.0
                    acc += vec * weight
                if pooling_mode == PoolingMode.MEAN and end > start:
                    acc /= (end - start)
                result[b, dim_start:dim_end] = cast(acc, output_dtype)
        return result
```

## 输入输出概要

| 名称 | 输入/输出 | 数据类型 | 数据格式                                           | 范围 | 说明                                              |
| --- |-------| --- |------------------------------------------------| --- |-------------------------------------------------|
| dev_weights | 输入    | uint8 | [total_weight_bytes]                           | NA | 多表合并后的量化权重缓冲区                                   |
| uvm_weights | 输入    | uint8 | NA                                             | NA | 保留参数                                            |
| lxu_cache_weights | 输入    | uint8 | NA                                             | NA | 保留参数                                            |
| weights_placements | 输入    | int32 | [T]                                            | PlacementType | 每张表的放置策略（DEVICE/HOST/MANAGED/…）                 |
| weights_offsets | 输入    | int64 | [T]                                            | 单调递增 | 每张表在 `dev_weights` 中的起始偏移                       |
| weights_tys | 输入    | uint8 | [T]                                            | SparseType | 每张表的量化类型，暂时只支持FP8（FP8/INT8/FP16/FP32/INT4/INT2） |
| D_offsets | 输入    | int32 | [T+1]                                          | 单调递增 | embedding 维度前缀和，用于定位各表输出区间                      |
| indices | 输入    | int32/64 | [N]                                            | offsets[-1] | 查表索引（nobag 在适配层转成 int32）                        |
| offsets | 输入    | 与 indices 同型 | [T*B+1]                                        | 单调递增 | bag 的起止偏移                                       |
| lxu_cache_locations | 输入    | int32 | NA                                             | NA | 保留参数                                            |
| offset_per_key | 输入    | int32 | [T+1]                                          | 单调递增 | 每张表在 offsets 中的起点（仅 nobag 模式使用）                 |
| indice_weights | 输入    | float | [N]                                            | NA | weighted bag 的 per-sample 权重，非加权时传空             |
| total_D | 属性    | int64 | NA                                             | NA | 所有表 embedding 维度之和（bag 输出宽度）                    |
| max_D | 属性    | int64 | NA                                             | NA | 各表 embedding 维度的最大值（nobag 输出宽度）                 |
| max_int2_D | 属性    | int64 | NA                                             | NA | INT2 权重量化的最大维度                                  |
| max_int4_D | 属性    | int64 | NA                                             | NA | INT4 权重量化的最大维度                                  |
| max_int8_D | 属性    | int64 | NA                                             | NA | INT8 权重量化的最大维度                                  |
| max_float16_D | 属性    | int64 | NA                                             | NA | FP16 权重量化的最大维度                                  |
| max_float32_D | 属性    | int64 | NA                                             | NA | FP32 权重（或输出）最大维度                                |
| max_float8_D | 属性    | int64 | NA                                             | NA | FP8 权重量化的最大维度                                   |
| pooling_mode | 属性    | int64 | NA                                             | {0,1,2} | 0=SUM，1=MEAN，2=NONE（nobag）                      |
| output_dtype | 属性    | int64 | NA                                             | SparseType | 输出数据类型（FP32/FP16/BF16/INT8）                     |
| row_alignment | 属性    | int64 | NA                                             | NA | 行级对齐要求（默认为 16）                                  |
| fp8_exponent_bits | 属性    | int64 | NA                                             | NA | FP8 反量化时使用的指数位数，若未启用 FP8 传 -1                   |
| fp8_exponent_bias | 属性    | int64 | NA                                             | NA | FP8 反量化时的 bias，若未启用 FP8 传 -1                    |
| out | 输出    | FP32/FP16/BF16/uint8 | bag: [B, total_D]；nobag: [len(indices), max_D] | NA | 查表结果；仅当所有表的权重类型为 INT8 时才允许 uint8 输出             |

## 约束说明

- `weights_offsets.size(0) > 0`，`indices.numel() > 0`，`offsets.size(0) > 1`。
- `weights_tys.size(0)` 必须等于 `weights_offsets.size(0)`。
- bag 模式要求 `(offsets.size(0) - 1) % feat_cnt == 0`。
- 若 `output_dtype == INT8`，则所有表的 `weights_tys` 也必须为 `INT8`。

### Embedding 维度约束

- FP8 权重量化时，embedding 维度 `D` 必须是 `4` 的倍数，且 `D <= 4096`。
- nobag 模式下，所有表的 embedding 维度必须一致，即每张表的 `D_offsets[i + 1] - D_offsets[i]` 相同。

## 上层调用示例

```python
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)

embedding_specs = [
    ("table0", 1024, 64, SparseType.INT8, EmbeddingLocation.DEVICE),
    ("table1", 2048, 64, SparseType.INT8, EmbeddingLocation.DEVICE),
]

op = IntNBitTableBatchedEmbeddingBagsCodegen(
    embedding_specs=embedding_specs,
    device="npu:0",
    pooling_mode=PoolingMode.SUM,
    output_dtype=SparseType.FP16,
    indices_dtype=torch.int64,
)

output = torch.ops.fbgemm.int_nbit_split_embedding_codegen_lookup_function(
    dev_weights=op.weights_dev,
    uvm_weights=op.weights_uvm,
    weights_placements=op.weights_placements,
    weights_offsets=op.weights_offsets,
    weights_tys=op.weights_tys,
    D_offsets=op.D_offsets,
    total_D=op.total_D,
    max_int2_D=op.max_int2_D,
    max_int4_D=op.max_int4_D,
    max_int8_D=op.max_int8_D,
    max_float16_D=op.max_float16_D,
    max_float32_D=op.max_float32_D,
    indices=indices,
    offsets=offsets,
    pooling_mode=int(op.pooling_mode),
    indice_weights=None,
    output_dtype=op.output_dtype,
    lxu_cache_weights=None,
    lxu_cache_locations=None,
    row_alignment=op.row_alignment,
    max_float8_D=op.max_float8_D,
    fp8_exponent_bits=op.fp8_exponent_bits,
    fp8_exponent_bias=op.fp8_exponent_bias,
)
```

## 编译与测试

- Ascend C 算子编译与适配层编译参考仓库根目录 [README.md](../../../README.md)。
- 调用示例与精度测试参考 [bench/tbe_inference/int_nbit_split_embedding_codegen_lookup_function_test/test_int_nbit_split_embedding_codegen_lookup_function.py](../../../bench/tbe_inference/int_nbit_split_embedding_codegen_lookup_function_test/test_int_nbit_split_embedding_codegen_lookup_function.py)。
