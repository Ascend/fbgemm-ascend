# direct_mapped_lru_cache_populate_byte

本算子实现 direct‑mapped（直接映射）LRU 嵌入缓存填充，包含 find_uncached 与 insert_byte 两个子算子。行为对齐 **FBGEMM CUDA** 入口 `direct_mapped_lru_cache_populate_byte_cuda`（`fbgemm_gpu/src/split_embeddings_cache/lru_cache_populate_byte.cu`）。

> **与 32‑way LRU 的主要差异**：direct‑mapped 每个 cache set 仅 1 个槽位（无 `unique_indices` 去重/排序步骤），冲突时通过 `atomicMax` 竞争决定哪个 miss 获得插入权。因此 `linear_cache_indices` 直接传入 find_uncached，返回的 `cache_sets` 与 `linear_cache_indices` 一一对应，无需 SortPairs。

## PyTorch 框架对外接口原型

通过 **fbgemm** 已注册的 schema 挂载 NPU 实现：

```python
torch.ops.fbgemm.direct_mapped_lru_cache_populate_byte(
    Tensor weights,
    Tensor hash_size_cumsum,
    int total_cache_hash_size,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor linear_cache_indices,
    Tensor(a!) lxu_cache_state,
    Tensor(b!) lxu_cache_weights,
    int time_stamp,
    Tensor(c!) lru_state,
    Tensor lxu_cache_miss_timestamp,
    int row_alignment = 16,
    bool gather_cache_stats = False,
    Tensor(d!)? uvm_cache_stats = None,
) -> ()
```

不依赖 `fbgemm_gpu` Python 包时，可使用本目录注册的 **mxrec** 接口（参数语义与上式一致）：

```python
torch.ops.mxrec.direct_mapped_lru_cache_populate_byte(
    Tensor weights,
    Tensor hash_size_cumsum,
    int total_cache_hash_size,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor linear_cache_indices,
    Tensor(a!) lxu_cache_state,
    Tensor(b!) lxu_cache_weights,
    int time_stamp,
    Tensor(c!) lru_state,
    Tensor lxu_cache_miss_timestamp,
    int row_alignment = 16,
    bool gather_cache_stats = False,
    Tensor(d!)? uvm_cache_stats = None,
) -> ()
```

## 参数说明

| 名称 | 输入/输出 | 数据类型 | 形状 | 说明 |
|------|---------|---------|------|------|
| weights | 输入 | uint8 | 一维 | UVM 侧字节权重（已量化） |
| hash_size_cumsum | 输入 | int64 | `[num_tables]` | 每张表的累积 hash size |
| total_cache_hash_size | 输入(标量) | int | — | max_indices 哨兵值，用于标记无效/被删索引 |
| cache_index_table_map | 输入 | int32 | `[total_cache_hash_size]` | 索引 → 表编号的映射 |
| weights_offsets | 输入 | int64 | `[num_tables]` | 每张表在 weights 中的起始偏移 |
| weights_tys | 输入 | uint8 | `[num_tables]` | 每张表的稀疏类型编码（见 SparseType） |
| D_offsets | 输入 | int32 | `[num_tables + 1]` | 每张表的嵌入维度偏移（`d_offsets[t+1] - d_offsets[t]` = 维度 D） |
| linear_cache_indices | 输入 | int32 / int64 | `[N]` | 已线性化的缓存索引（未经去重） |
| lxu_cache_state | 输入/输出 | int64 | `[C, 1]` | Cache 状态（direct‑mapped 仅有 1 路） |
| lxu_cache_weights | 输入/输出 | uint8 | `[C, row_bytes]` | Cache 权重存储，须 contiguous |
| time_stamp | 输入(标量) | int | — | 写入 LRU 的时间戳 |
| lru_state | 输入/输出 | int64 | `[C, 1]` | LRU 时间戳，须 contiguous |
| lxu_cache_miss_timestamp | 输入 | int64 | `[C, 1]` | 每个 cache set 最近一次 miss 的时间戳（direct‑mapped 特有，用于 atomicMax 竞争） |
| row_alignment | 输入(标量) | int | 默认 16 | 行对齐（字节），传入 insert 算子 |
| gather_cache_stats | 输入(标量) | bool | 默认 false | 为 true 时须提供 `uvm_cache_stats` |
| uvm_cache_stats | 输入/输出 | int32 | 一维 | 统计收集张量（可选） |


## 实现流程

```
direct_mapped_lru_cache_populate_byte
  │
  ├─ Step 1: direct_mapped_lru_cache_find_uncached
  │     ├─ 遍历 linear_cache_indices（非去重）
  │     ├─ MurmurHash3(idx, C) 计算 cache_set
  │     ├─ 命中: 更新 lru_state，标记 cache_sets[n] = -1
  │     └─ 未命中: Lane 0 执行 atomicMax 竞争写入权
  │         ├─ 胜出: cache_sets[n] = cache_set
  │         └─ 失败: cache_sets[n] = -1
  │
  └─ Step 2: direct_mapped_lru_cache_insert_byte
        ├─ 跳过 cache_sets[pos] == -1 的项
        ├─ 跳过 lru_state[cache_set] == time_stamp 的项（本批次已命中）
        ├─ 从 weights 解析行数据并计算 padded 大小
        ├─ Warp 内 Lane 并行拷贝到 lxu_cache_weights
        └─ Lane 0 更新 lxu_cache_state 和 lru_state
```

## 算子编译与部署

子算子（`direct_mapped_lru_cache_find_uncached`、`direct_mapped_lru_cache_insert_byte`）编译部署：

```bash
# 编译 find_uncached 子算子
cd ../direct_mapped_lru_cache_find_uncached/c310
chmod +x run.sh
./run.sh

# 编译 insert_byte 子算子
cd ../direct_mapped_lru_cache_insert_byte/c310
chmod +x run.sh
./run.sh
```

PyTorch 适配层编译请参考 [fbgemm-ascend/README.md](../../../README.md) 中「单算子使用说明」章节。

## 算子调用示例

```python
import fbgemm_ascend

# 以下为接口形态示例，具体张量构造见测试用例
torch.ops.fbgemm.direct_mapped_lru_cache_populate_byte(
    weights,
    hash_size_cumsum,
    total_cache_hash_size,
    cache_index_table_map,
    weights_offsets,
    weights_tys,
    D_offsets,
    linear_cache_indices,
    lxu_cache_state,
    lxu_cache_weights,
    time_stamp,
    lru_state,
    lxu_cache_miss_timestamp,
    row_alignment=16,
    gather_cache_stats=False,
    uvm_cache_stats=None,
)
```

完整测试用例请参考 [bench/direct_mapped_lru_cache_populate_byte/](../../../bench/direct_mapped_lru_cache_populate_byte/)。

## 目录结构

```shell
direct_mapped_lru_cache_populate_byte
├── direct_mapped_lru_cache_populate_byte.cpp    # 合并算子 PyTorch 适配层
└── README.md
```

子算子位于同级目录：

```shell
../direct_mapped_lru_cache_find_uncached/
├── README.md
└── c310/
    ├── run.sh
    ├── direct_mapped_lru_cache_find_uncached.json
    ├── op_host/                                 # Host 侧 tiling + OpDef 注册
    └── op_kernel/                               # Kernel 侧 SIMT 实现

../direct_mapped_lru_cache_insert_byte/
├── README.md
└── c310/
    ├── run.sh
    ├── direct_mapped_lru_cache_insert_byte.json
    ├── op_host/                                 # Host 侧 tiling + OpDef 注册
    └── op_kernel/                               # Kernel 侧 SIMT 实现（含 cache_constants / padded_row）
```

## 子算子说明

- [direct_mapped_lru_cache_find_uncached](../direct_mapped_lru_cache_find_uncached/README.md) — 查找未命中项，通过 atomicMax 竞争写入权
- [direct_mapped_lru_cache_insert_byte](../direct_mapped_lru_cache_insert_byte/README.md) — 将 miss 的权重数据插入 cache

## aclnn 底层说明

适配层通过 `EXEC_NPU_CMD` 调用 `libopapi.so` 中的 `aclnnDirectMappedLruCacheFindUncached` / `aclnnDirectMappedLruCacheInsertByte`。张量与标量顺序须与 CANN 定义一致；若升级 CANN 后签名变化，请同步修改 `.cpp` 中的 `EXEC_NPU_CMD` 实参。

find_uncached aclnn 参数顺序:
1. `linear_cache_indices`
2. `lxu_cache_state`
3. `lru_state`
4. `lxu_cache_miss_timestamp`
5. `uvm_cache_stats`
6. `cache_sets`（输出）
7. `max_indices`（attr → tiling）
8. `time_stamp`（attr → tiling）
9. `gather_cache_stats`（attr → tiling）

insert_byte aclnn 参数顺序:
1. `weights`
2. `cache_hash_size_cumsum`
3. `cache_index_table_map`
4. `weights_offsets`
5. `weights_tys`
6. `d_offsets`
7. `lxu_cache_state`（in/out）
8. `lxu_cache_weights`（in/out）
9. `lru_state`（in/out）
10. `linear_cache_indices`
11. `lxu_cache_miss_timestamp`
12. `cache_sets`
13. `uvm_cache_stats`（in/out）
14. `reserved_out`（输出）
15. `gather_cache_stats`（attr → tiling）
16. `time_stamp`（attr → tiling）
17. `row_alignment`（attr → tiling）

## 限制说明

- 如果打开gather_cache_stats，那么必须提供uvm_cache_stats
- gather_cache_stats当前仅支持false，目前暂不支持uvm相关功能。故gather_cache_stats和uvm_cache_stats当前为保留参数，暂不支持相关功能
