**说明**

本算子仅支持NPU调用。

# 产品支持情况
| 硬件型号              | 是否支持 |
| -------------------- | ------ |
| Atlas A5训练系列产品  | 是      |
| Atlas A2训练系列产品  | 否      |
| Atlas A3训练系列产品  | 否      |
| Atlas 推理系列产品    | 否      |

# get_unique_indices算子目录层级

```shell
get_unique_indices
|-- get_unique_indices.cpp        # PyTorch适配层实现（aclnnSort + aclnnRunLengthEncode）
|-- c310
    |-- op_host                   # 算子host侧实现
    |-- op_kernel                 # 算子kernel侧实现
    |-- run_length_encode.json    # 算子原型配置
    |-- README.md                 # 算子说明文档
    |-- run.sh                    # 算子编译部署脚本
```

# 功能

`get_unique_indices(_with_inverse)` 用于对大量重复索引排序去重，可选返回每个唯一值出现次数以及稳定排序后的原始位置映射，减少后续 embedding 查询与聚合的重复计算。

`get_unique_indices(_with_inverse)` 对非负输入 `linear_indices` 进行全局去重：
- 输出去重后的 `unique_indices`
- 输出 unique 元素个数 `unique_indices_length`
- 可选输出每个 unique 值出现次数 `unique_indices_count`
- 可选输出稳定排序后的原始位置索引 `inverse_indices`，用于后续 gather 恢复原始顺序

`get_unique_indices` 主要流程为：
1. 调用 `aclnnSort` 对 `linear_indices` 做基数排序
2. 调用 `aclnnRunLengthEncode` 对 `sorted_indices` 做游程编码
# 算子实现原理

输入：

```python
sorted_indices = [1, 1, 3, 3, 3, 6, 8, 8]
return_count = True
```

输出：

```python
unique_indices = [1, 3, 6, 8]
unique_indices_count = [2, 3, 1, 2]
unique_indices_length = [4]
```

空输入时：
- `unique_indices` shape 为 `[0]`
- `unique_indices_length = [0]`
- `unique_indices_count` shape 为 `[0]`
- `inverse_indices` shape 为 `[0]`

# 算子输入与输出
`run_length_encode`（Ascend C算子）：

| 名称 | 输入/输出 | 数据类型 | 数据格式 | 范围 | 说明 |
|------|---|---|---|---|---|
| sorted_indices | 输入 | int32/int64 | [dim0] | 一维，dim0∈[0, 2^31-1] | 已排序索引序列 |
| return_count | 输入（属性） | bool | NA | {true,false} | 是否返回 count 输出 |
| unique_indices | 输出 | int32/int64 | [dim0] | 一维，与输入等长 | 去重后的索引，dtype与输入一致 |
| unique_indices_count | 输出 | int32 | [dim0] | 一维，与输入等长 | 每个 unique 值的出现次数；当 `return_count=false` 时该输出可忽略 |
| unique_indices_length | 输出 | int32 | [1] | 固定长度1 | unique 元素个数 |

`get_unique_indices`（PyTorch适配层接口）：

| 名称 | 输入/输出 | 数据类型 | 数据格式 | 范围 | 说明 |
|------|---|---|---|---|---|
| linear_indices | 输入 | int32/int64 | [dim0] | 一维，与输入等长，dim0∈[0, 2^31-1] | 原始索引序列，`0 <= linear_indices[i] < max_indices` |
| max_indices | 输入（属性） | int64 | NA | `>= 0` | `linear_indices[i] < max_indices` |
| compute_count | 输入（属性） | bool | NA | {true,false} | 是否返回 count 输出 |
| unique_indices | 输出 | int32/int64 | [dim0] | 一维，与输入等长 | 去重后的索引，dtype与输入一致 |
| unique_indices_length | 输出 | int32 | [1] | 固定长度1 | unique 元素个数 |
| unique_indices_count | 输出（可选） | int32 | [dim0] | 一维，与输入等长 | unique 元素出现次数，`compute_count=true` 时返回，否则为 `None` |

`get_unique_indices_with_inverse`（PyTorch适配层接口）：

| 名称 | 输入/输出 | 数据类型 | 数据格式 | 范围 | 说明 |
|------|---|---|---|---|---|
| linear_indices | 输入 | int32/int64 | [dim0] | 一维，dim0∈[0, 2^31-1] | 原始索引序列，`0 <= linear_indices[i] < max_indices` |
| max_indices | 输入（属性） | int64 | NA | `>= 0` | `linear_indices[i] < max_indices` |
| compute_count | 输入（属性） | bool | NA | {true,false} | 是否返回 count 输出 |
| compute_inverse_indices | 输入（属性） | bool | NA | {true,false} | 是否返回 inverse 输出 |
| unique_indices | 输出 | int32/int64 | [dim0] | 一维，与输入等长 | 去重后的索引，dtype与输入一致 |
| unique_indices_length | 输出 | int32 | [1] | 固定长度1 | unique 元素个数 |
| unique_indices_count | 输出（可选） | int32 | [dim0] | 一维，与输入等长 | unique 元素出现次数，`compute_count=true` 时返回，否则为 `None` |
| inverse_indices | 输出（可选） | int32 | [dim0] | 一维，与输入等长 | 稳定排序后的原始位置序列，`compute_inverse_indices=true` 时返回，否则为 `None` |

约束说明：
- 输入 `linear_indices` 都必须为1维，dtype 仅支持 `int32/int64`。
- 输入 `max_indices` 必须为 `int64` 类型，须满足 `max_indices >= 0`。
- 输入 `linear_indices` 中的每个元素必须满足 `0 <= linear_indices[i] < max_indices`。
- Host 侧硬约束 `len(linear_indices) <= INT32_MAX`，超限在 tiling 阶段报错。

# 算子编译部署

注：算子调用与精度校验可参考[bench/split_embeddings_cache/get_unique_indices_test/test_get_unique_indices.py](../../../../bench/split_embeddings_cache/get_unique_indices_test/test_get_unique_indices.py)。
