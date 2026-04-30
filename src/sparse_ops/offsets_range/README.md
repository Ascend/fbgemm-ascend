# offsets_range

本算子仅支持 NPU 调用，用于根据 offsets 生成分段内局部下标。

## 目录结构

```text
offsets_range
|-- offsets_range.cpp
|-- README.md
|-- c310/
|   `-- run.sh
`-- v220/
    |-- offsets_range.json
    |-- op_host/
    |-- op_kernel/
    `-- run.sh
```

## 硬件支持

| 目录 | 说明 |
| --- | --- |
| `c310/` | 提供 C310 构建脚本 |
| `v220/` | 提供 V220 Ascend C 实现 |

支持硬件为 Atlas A2 / A3 / A5 训练系列。

## PyTorch 接口原型

```python
torch.ops.mxrec.offsets_range(Tensor offsets, int range_size) -> Tensor
torch.ops.fbgemm.offsets_range(Tensor offsets, int range_size) -> Tensor
```

## 功能说明

根据输入分段起始位置 `offsets` 和输出长度 `rangeSize`，生成每个分段内的局部下标。

输出 `result` 的定义如下：
- 第 `i` 段区间：`[offsets[i], offsets[i+1])`（最后一段为 `[offsets[last], rangeSize)`）
- 在每段内填充 `0, 1, 2, ...`

## 算子实现原理

输入：

```python
offsets = [0, 2, 5, 5]  # 一维，表示4个分段的起始位置
rangeSize = 7            # 输出长度
```

分段区间为：
- 第0段: `[0, 2)` -> `[0, 1]`
- 第1段: `[2, 5)` -> `[0, 1, 2]`
- 第2段: `[5, 5)` -> `[]`（空段）
- 第3段: `[5, 7)` -> `[0, 1]`

输出：

```python
result = [0, 1, 0, 1, 2, 0, 1]
```

## 参数与约束

| 名称 | 输入/输出 | 数据类型 | 数据格式 | 范围 | 说明 |
|------|---|---|---|---|---|
| offsets | 输入 | int32/int64 | [dim0] | 一维，dim0∈[1,2^17] | 各分段起始位置 |
| rangeSize | 输入（属性） | int | NA | rangeSize∈[1,2^32] | 输出长度 |
| result | 输出 | int32/int64 | [rangeSize] | 一维，长度为 `rangeSize` | 每个分段内的局部下标 |

约束说明：
- `offsets` 必须非递减。
- `offsets[0]` 必须为 `0`。
- `offsets[-1] <= rangeSize`。
- 允许空分段（即 `offsets[i] == offsets[i+1]`）。
- 输出 `result` 的数据类型与 `offsets` 一致。
- `v220/run.sh` 默认构建目标仅支持int32，`c310/run.sh`默认构建目标支持int32/int64。

## 调用示例

```python
import sysconfig
import torch
import torch_npu
import fbgemm_ascend

offsets = torch.tensor([0, 2, 5, 5], dtype=torch.int32, device="npu")
result = torch.ops.fbgemm.offsets_range(offsets, 7)
```

## 编译与测试

- Ascend C 算子编译与适配层编译参考仓库根目录 [README.md](../../../README.md)。
- 测试示例参考：
  - [bench/sparse/offsets_range/test_offsets_range.py](../../../bench/sparse/offsets_range/test_offsets_range.py)
  - [bench/sparse/offsets_range/special_test_offsets_range.py](../../../bench/sparse/offsets_range/special_test_offsets_range.py)
