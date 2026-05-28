# direct_mapped_lru_cache_insert_byte

本算子是 direct_mapped_lru_cache_populate_byte 的子接口，不支持直接调用。完整入口参考 [direct_mapped_lru_cache_populate_byte/README.md](../direct_mapped_lru_cache_populate_byte/README.md)。

## 产品支持情况

| 硬件型号   | 是否支持 |
|-----------|---------|
| Atlas A5  | 是      |

## 算子目录层级

```shell
direct_mapped_lru_cache_insert_byte
├── README.md
└── c310/
    ├── run.sh
    ├── direct_mapped_lru_cache_insert_byte.json
    ├── op_host/                                 # Host 侧 tiling + OpDef 注册
    └── op_kernel/                               # Kernel 侧 SIMT 实现（含 cache_constants / padded_row）
```

## 功能

将 UVM 中的量化字节权重按 `cache_sets` 与 `linear_cache_indices` 写入 `lxu_cache_weights`，并更新 `lxu_cache_state`、`lru_state` 及可选的 `uvm_cache_stats`。

语义与 **FBGEMM** `direct_mapped_lru_cache_insert_byte_kernel`（`fbgemm_gpu/src/split_embeddings_cache/lru_cache_populate_byte.cu`）对齐。

与 32‑way LRU `lru_cache_insert_byte` 的关键差异：

- **无需 sorted_cache_sets / cache_set_sorted_unique_indices**：direct‑mapped 无排序/去重步骤，直接使用 `cache_sets` 和 `linear_cache_indices`
- **新增 linear_cache_indices**：原始索引直接传入，无需通过 SortPairs 后的中间张量
- **新增 lxu_cache_miss_timestamp**：作为输入张量保留（与 find_uncached 一致），insert 阶段不读取但需占位
- **W = 1**：`lxu_cache_state` 形状为 `[C, 1]` 而非 `[C, 32]`

## 算子实现原理

依据 `cache_hash_size_cumsum`、`cache_index_table_map`、`weights_offsets`、`weights_tys`、`d_offsets` 等元数据，从一维 `weights`（uint8）中解析各行并写入 cache。

处理流程（单 warp 处理一个 cache_set）：

1. 跳过 `cache_sets[pos] == -1`（find 阶段标记的不插入项）
2. 跳过 `lru_state[cache_set] == time_stamp`（本批次其他 miss 已写入同一 cache_set）
3. 通过 `cache_index_table_map[insert_idx]` 获取表编号和权重信息
4. 计算 `idx_insert = insert_idx - cache_hash_size_cumsum[t_insert]`，从 weights 中定位行数据
5. 通过 `PaddedRowSizeBytes` 计算对齐后的行大小
6. Warp 内各 lane 并行拷贝到 `lxu_cache_weights`
7. Lane 0 更新 `lxu_cache_state[cache_set]` 和 `lru_state[cache_set]`

## 算子输入与输出

| 名称 | 输入/输出 | 数据类型 | 形状 | 说明 |
|------|---------|---------|------|------|
| weights | 输入 | uint8 | 一维 | UVM 侧字节权重 |
| cache_hash_size_cumsum | 输入 | int64 | `[num_tables]` | 每张表的累积 hash size |
| cache_index_table_map | 输入 | int32 | `[total_cache_hash_size]` | 索引 → 表编号映射 |
| weights_offsets | 输入 | int64 | `[num_tables]` | 每张表在 weights 中的起始偏移 |
| weights_tys | 输入 | uint8 | `[num_tables]` | 每张表的稀疏类型编码 |
| d_offsets | 输入 | int32 | `[num_tables + 1]` | 每张表的嵌入维度偏移 |
| lxu_cache_state | 输入/输出 | int64 | `[C, 1]` | Cache 状态 |
| lxu_cache_weights | 输入/输出 | uint8 | `[C, row_bytes]` | Cache 权重存储 |
| lru_state | 输入/输出 | int64 | `[C, 1]` | LRU 时间戳 |
| linear_cache_indices | 输入 | int32 / int64 | `[N]` | 线性缓存索引 |
| lxu_cache_miss_timestamp | 输入 | int64 | `[C, 1]` | 最近 miss 时间戳（direct‑mapped 特有） |
| cache_sets | 输入 | int32 | `[N]` | 由 find_uncached 输出的 cache set 分配 |
| uvm_cache_stats | 输入/输出 | int32 | 一维 | 统计收集（gather_cache_stats=false 时为空张量） |
| reserved_out | 输出 | int32 | 占位 | 与 aclnn 插件约定的占位输出 |

### 属性参数（经 tiling 下发）

| 名称 | 必选/可选 | 类型 | 说明 |
|------|----------|------|------|
| gather_cache_stats | 可选 | bool | 默认 false，为 true 时写入 num_conflict_unique_misses |
| lru_timestamp | 必选 | int | 当前 LRU 时间戳 |
| row_alignment | 必选 | int | 行对齐（字节），默认 16 |

## 配套头文件

算子 kernel 侧依赖两个辅助头文件：

- [cache_constants.h](c310/op_kernel/cache_constants.h) — FBGEMM 兼容常量（kCacheStateInvalid、kINT8QparamsBytes、UvmCacheStatsIndex）
- [padded_row.h](c310/op_kernel/padded_row.h) — 行大小计算（UnpaddedRowSizeBytes / PaddedRowSizeBytes），支持 FP32 / FP16 / BF16 / FP8 / INT8 / INT4 / INT2

## 算子编译部署

```bash
chmod +x run.sh
./run.sh
```
