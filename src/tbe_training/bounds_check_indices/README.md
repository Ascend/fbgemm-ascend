# bounds_check_indices 算子文档

## 1. 概述

`bounds_check_indices` 是嵌入查表（Embedding Lookup）操作中的边界检查算子，用于验证 indices 和 offsets 的合法性，防止越界访问。

在分布式嵌入训练中，indices 可能包含无效值（负数、超出表大小等），该算子可在 NPU 上对这些值进行检查和修正。

### 应用场景

- 嵌入查表（Embedding Lookup）前的输入验证
- 稀疏特征处理中的边界检查

---

## 2. 支持的 BoundsCheckMode

| Mode | 值 | 行为 |
|------|-----|------|
| `FATAL` | 0 | DEBUG模式检测到越界时触发设备断言，导致进程终止 |
| `WARNING` | 1 | 检测到越界时：修正越界值为0/有效范围，累加警告计数，输出日志 |
| `IGNORE` | 2 | 检测到越界时：静默修正越界值为0/有效范围，不输出警告 |

---

## 3. V1 与 V2 版本差异

| 特性 | V1 | V2 |
|------|-----|-----|
| 线程配置 | 256 threads/block (32×8) | 1024 threads/block (32×32) |
| block维度 | `dim3(32, 8, 1)` | `dim3(32, 32, 1)` |
| VBE支持 | 支持（通过B_offsets） | 支持（通过B_offsets + b_t_map） |
| b_t_map / info_B_num_bits / info_B_mask | ❌ 不支持 | ✅ 支持 |
| prefetch_pipeline | ❌ 不支持 | ✅ 支持 |

---

## 4. PyTorch 接口

```python
torch.ops.fbgemm.bounds_check_indices(
    rows_per_table,           # Tensor[int64]       - 每张表的行数 (T,)
    indices,                  # Tensor[int32/int64] - 待检查的索引 (N,)
    offsets,                  # Tensor[int32/int64] - 偏移数组 (total_B+1,)  包含累积长度
    bounds_check_mode,        # int                 - 0=FATAL, 1=WARNING, 2=IGNORE
    warning,                  # Tensor[int64]       - 警告计数输出
    weights=None,             # Tensor[float]       - 可选权重
    B_offsets=None,           # Tensor[int32]       - VBE模式：每张表的batch偏移
    max_B=-1,                 # int                 - VBE模式：最大batch size
    b_t_map=None,             # Tensor[int32]       - V2专用：bTIdx映射表
    info_B_num_bits=-1,       # int                 - V2专用：信息位数
    info_B_mask=-1,           # int                 - V2专用：信息掩码
    bounds_check_version=1,   # int                 - 1或2
) -> ()
```

### 输出说明

算子**原地（in-place）修改**以下tensor：
- `indices` - 越界索引被修正为0
- `offsets` - 越界偏移被修正为有效范围
- `warning` - WARNING模式下累加越界计数

---

## 5. 数据布局

### 5.1 普通模式（non-VBE）

```text
# 数据结构
rows_per_table: [rows[0], rows[1], ..., rows[T-1]]                # 每张表的行数，T个元素
offsets:        [0, L[0], L[0]+L[1], ..., L[0]+...+L[total_B-1]]  # total_B+1个元素
indices:        [i[0], i[1], ..., i[N-1]]                         # 共N=sum(L_i)个索引

# 索引映射关系
对于 bTIdx ∈ [0, total_B)：
  - table t = bTIdx // B
  - batch b = bTIdx % B
  - 该table-batch的indice范围：[offsets[bTIdx], offsets[bTIdx+1])
```

### 5.2 VBE模式（Variable Batch per Embedding）

```text
# 数据结构
B_offsets: [0, B[0], B[0]+B[1], ..., B[0]+...+B[T-1]]  # T+1个元素，total_B = B_offsets[-1]
offsets:   同普通模式

# V2 VBE特有映射
b_t_map[bTIdx] = (t << info_B_num_bits) | b
  - t: table index，范围 [0, T)
  - b: batch index within that table
```

---

## 6. 算子原型定义（JSON）

### 6.1 BoundsCheckIndicesV1

**输入：**
| 名称 | 类型 | 必填 | 说明 |
|------|------|------|------|
| rows_per_table | int64 | ✓ | 每张表的行数 |
| indices | int32/int64 | ✓ | 待检查索引 |
| offsets | int32/int64 | ✓ | 偏移数组 |
| warning | int64 | ✓ | 警告计数输出 |
| B_offsets | int32 | ○ | VBE模式batch偏移 |

**属性：**
| 名称 | 类型 | 说明 |
|------|------|------|
| bounds_check_mode | int32 | 检查模式 |
| max_B | int32 | VBE最大batch size |
| T | int32 | 表数量 |
| B | int32 | 平均每张表batch size |
| total_B | int32 | 总batch数 |
| vbe | bool | 是否VBE模式 |

### 6.2 BoundsCheckIndicesV2

**额外输入：**
| 名称 | 类型 | 必填 | 说明 |
|------|------|------|------|
| b_t_map | int32 | ○ | bTIdx映射表 |

**额外属性：**
| 名称 | 类型 | 说明 |
|------|------|------|
| info_B_num_bits | int64 | batch信息位数 |
| info_B_mask | int64 | batch信息掩码 |
| prefetch_pipeline | bool | 预取流水线开关 |

---

## 7. Tiling 策略

### 7.1 V1 Tiling

```cpp
blockDim = dim3(32, 8)
gridSize = ceil(MaxB * T / 8)
```

V1单核起256个线程，每32个线程1个warp。当batch不固定时，MaxB为所有batchSize的最大值，饱和式申请资源。

### 7.2 V2 Tiling

```cpp
blockDim = dim3(32, 32)
gridSize = min(ceil(total_B / 32), maxAvailBlocks)
```

V1单核起1024个线程，每32个线程1个warp。V2使用更大的并行度，batch不固定场景对NPU资源的利用率更高。

---

## 8. TilingData 结构体

```cpp
struct BoundsCheckIndicesTilingData {
    int64_t  numIndices;          // indices总数
    int32_t  numTables;           // 表数量
    int32_t  batchSize;           // batch size
    int32_t  totalB;              // 总样本数
    int32_t  boundsCheckMode;     // 检查模式
    int32_t  vbe;                 // 是否VBE
    int32_t  infoBNumBits;        // 信息位数（V2）
    uint32_t  infoBMask;          // 信息掩码（V2）
    uint32_t batchSizeDivMagic;   // 除数的近似倒数放大
    uint32_t batchSizeDivShift;   // 除数的位移量，用于处理缩放
};
```

---

## 9. 索引检查逻辑

#### 第一步：检验 offsets 数组最后一个元素的值和indices数组长度是否相等

```text
# 修正offsets[totalB]要等于numIndices
if bounds_check_mode == FATAL:
    assert offsets[totalB] == numIndices
elif bounds_check_mode == WARNING:
    if offsets[totalB] != numIndices:
        print("WARNING: last element in offsets is incorrect...")
        offsets[totalB] = numIndices
elif bounds_check_mode == IGNORE:
    if offsets[totalB] != numIndices:
        offsets[totalB] = numIndices
```

#### 第二步：检查offsets数组是否正常: 0 < offsets[i] < offsets[i+1] < len(indices)

```text
# 修正偏移范围
if bounds_check_mode == FATAL:
    assert indicesStart >= 0
    assert indicesStart <= indicesEnd
    assert indicesEnd <= numIndices
elif bounds_check_mode == WARNING:
    if indicesStart < 0 or indicesStart > indicesEnd or indicesEnd > numIndices:
        print("WARNING: out of bounds access for batch, table, indicesStart, indicesEnd...")
        indicesStart = max(0, min(indicesStart, numIndices))
        indicesEnd = max(indicesStart, min(indicesEnd, numIndices))
        offsets[bTIdx] = indicesStart
        offsets[bTIdx + 1] = indicesEnd
elif bounds_check_mode == IGNORE:
    if indicesStart < 0 or indicesStart > indicesEnd or indicesEnd > numIndices:
        indicesStart = max(0, min(indicesStart, numIndices))
        indicesEnd = max(indicesStart, min(indicesEnd, numIndices))
        offsets[bTIdx] = indicesStart
        offsets[bTIdx + 1] = indicesEnd
```

#### 第三步：检查indice是否均小于其所在embedding表的行数

```text
bagSize = indicesEnd - indicesStart
# 检查每个索引
for i in range(threadIdx.x, bagSize, warpSize):
    idx = indices[indicesStart + i]
    if idx == -1:
        continue  # -1 表示 pruned rows，跳过检查
    if bounds_check_mode == FATAL:
        assert idx >= 0 and idx < numRows
    elif bounds_check_mode == WARNING:
        if idx < 0 or idx >= numRows:
            indices[indicesStart + i] = 0
            warningInc += 1
    elif bounds_check_mode == IGNORE:
        if idx < 0 or idx >= numRows:
            indices[indicesStart + i] = 0

if bounds_check_mode == WARNING and warningInc > 0:
    if atomic_add(&warning[0], warningInc) == 0:
        print("WARNING: (at least one) out of bounds access for batch, table, bag_element, idx, numRows...")
```

---

## 10. 文件结构

```text
bounds_check_indices/
├── bounds_check_indices.cpp              # PyTorch注册层（mxrec/fbgemm PrivateUse1）
└── c310/
    ├── bounds_check_indices.json         # 算子原型定义
    ├── op_host/
    │   ├── bounds_check_indices.cpp      # Host侧Tiling逻辑
    │   └── bounds_check_indices_tiling.h
    └── op_kernel/
        ├── bounds_check_indices_v1.cpp   # V1 kernel实现
        ├── bounds_check_indices_v2.cpp   # V2 kernel实现
        └── bounds_check_indices_common.h # 公共定义
```

---

## 11. 使用示例

```python
import torch
import fbgemm_ascend

# 基本用法
rows_per_table = torch.tensor([1000, 2000, 3000], dtype=torch.int64, device='npu')  # 3张表
indices = torch.tensor([0, 500, 1500, 999, 2500, -1, 5000], dtype=torch.int64, device='npu')
offsets = torch.tensor([0, 2, 4, 7], dtype=torch.int64, device='npu')  # B=1, T=3 → total_B=3
warning = torch.tensor([0], dtype=torch.int64, device='npu')

torch.ops.fbgemm.bounds_check_indices(
    rows_per_table,
    indices,
    offsets,
    bounds_check_mode=1,  # WARNING
    warning=warning,
    bounds_check_version=2,
)
print(f"Warning count: {warning.item()}")
```

---

## 12. 注意事项

1. **FATAL模式**：若设置NDEBUG断言不会生效，若未设置NDEBUG会导致断言失败NPU中断执行
2. **索引-1**：检查时会自动跳过
3. **VBE模式**：需要正确生成 `B_offsets` 和 `b_t_map`，否则映射关系错误
4. **dtype一致性**：`indices` 和 `offsets` 必须使用相同的数据类型（int32或int64）
5. **prefetch_pipeline**：当prefetch_pipeline为true时，NPU不会限制使用的核数（8核）。这样设计主要是为了与fbgemm接口行为保持一致。
6. **debug**：fbgemm_gpu源码编译支持设置debug的等级，当debug>=1时会开启TORCH_USE_CUDA_DSA。CANN暂未支持该能力，因此本算子暂时不支持设置debug等级开启DSA。