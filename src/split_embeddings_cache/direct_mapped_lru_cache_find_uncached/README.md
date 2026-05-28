# direct_mapped_lru_cache_find_uncached

本算子是 direct_mapped_lru_cache_populate_byte 的子接口，不支持直接调用。完整入口参考 [direct_mapped_lru_cache_populate_byte/README.md](../direct_mapped_lru_cache_populate_byte/README.md)。

## 产品支持情况

| 硬件型号   | 是否支持 |
|-----------|---------|
| Atlas A5  | 是      |

## 算子目录层级

```shell
direct_mapped_lru_cache_find_uncached
├── README.md
└── c310/
    ├── run.sh
    ├── direct_mapped_lru_cache_find_uncached.json
    ├── op_host/                                 # Host 侧 tiling + OpDef 注册
    └── op_kernel/                               # Kernel 侧 SIMT 实现
```

## 功能

根据原始线性缓存索引（`linear_cache_indices`，未经去重）在 `lxu_cache_state [C, 1]` 中查找命中和未命中项，通过 `atomicMax` 竞争机制解决 direct‑mapped 冲突，写出每个索引对应的 cache set；命中时更新 `lru_state`。

语义与 **FBGEMM** `direct_mapped_lru_cache_find_uncached_kernel`（`fbgemm_gpu/src/split_embeddings_cache/lru_cache_populate_byte.cu`）对齐。

与 32‑way LRU `lru_cache_find_uncached` 的关键差异：

- **无需 unique_indices**：`linear_cache_indices` 直接输入，不先去重
- **无需 unique_indices_length / lock_cache_line**：无专用的 unique 长度和锁计数器
- **新增 lxu_cache_miss_timestamp**：用于 `atomicMax` 竞争，决定哪个 miss 获得插入权
- **cache_sets 与 linear_cache_indices 一一对应**：无排序/索引重排步骤，长度相同
- **W = 1**：direct‑mapped 每个 cache set 仅 1 个槽位

## 算子实现原理

对 `linear_cache_indices` 中每个元素（单 warp 处理一个元素，32 lane 协作）：

1. 用 **MurmurHash3** 64‑bit 哈希函数计算 `cache_set = CacheSlot(idx, C)`
2. 检查 `lxu_cache_state[cache_set] == idx`（命中判断）
3. 命中：所有 lane 写 `lru_state[cache_set] = time_stamp`，lane 0 写 `cache_sets[n] = -1`
4. 未命中：仅 lane 0 执行 `atomicMax(&lxu_cache_miss_timestamp[cache_set], time_stamp + 1)`
   - 胜出（old < time_stamp + 1）：`cache_sets[n] = cache_set`，可插入
   - 失败：`cache_sets[n] = -1`，由其他索引插入

## 算子输入与输出

| 名称 | 输入/输出 | 数据类型 | 形状 | 说明 |
|------|---------|---------|------|------|
| linear_cache_indices | 输入 | int32 / int64 | `[N]` | 原始线性缓存索引（未经去重） |
| lxu_cache_state | 输入 | int64 | `[C, 1]` | Cache 状态（direct‑mapped 仅有 1 路） |
| lru_state | 输入/输出 | int64 | `[C, 1]` | LRU 时间戳，命中时原地更新 |
| lxu_cache_miss_timestamp | 输入/输出 | int64 | `[C, 1]` | 最近 miss 时间戳，通过 atomicMax 竞争更新 |
| cache_sets | 输出 | int32 | `[N]` | 每个索引对应的 cache set（‑1 = 不插入，‑1 标记与 CUDA 一致） |

### 属性参数（经 tiling 下发）

| 名称 | 必选/可选 | 类型 | 说明 |
|------|----------|------|------|
| max_indices | 必选 | int | `total_cache_hash_size` 哨兵值，匹配的索引视为无效/已删除 |
| lru_timestamp | 必选 | int | 当前 LRU 时间戳 |
| gather_cache_stats | 可选 | bool | 默认 false，为 true 时更新 uvm_cache_stats |

## 算子编译部署

```bash
chmod +x run.sh
./run.sh
```

## 限制说明

- 如果打开gather_cache_stats，那么必须提供uvm_cache_stats
- gather_cache_stats当前仅支持false，目前暂不支持uvm相关功能。故gather_cache_stats和uvm_cache_stats当前为保留参数，暂不支持相关功能
