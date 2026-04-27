# lru_cache_find_uncached说明

本算子仅支持NPU调用。

## 产品支持情况

| 硬件型号           | 是否支持 |
|----------------|------|
| Atlas A5 | 是    |

## lru_cache_find_uncached 算子目录层级

```shell
lru_cache_find_uncached
|-- c310
   |-- op_host                        # 算子 host 侧实现
   |-- op_kernel                      # 算子 kernel 侧实现
   |-- lru_cache_find_uncached.json   # 算子原型配置
   |-- README.md                      # 算子说明文档
   |-- run.sh                         # 算子编译部署脚本
```

## 功能

在推荐嵌入缓存路径中，根据去重后的线性缓存索引（`unique_indices`）在 `lxu_cache_state` 中查找未命中项，写出每个索引对应的 cache set，并**原地**更新 `lru_state` 中命中槽的 LRU 时间戳；语义与 **FBGEMM** `lru_cache_find_uncached_kernel` / `lru_cache_find_uncached_cuda`（`fbgemm_gpu/src/split_embeddings_cache/lru_cache_find.cu`）对齐，张量形参与 Attr 划分一致。

## 算子实现原理

对 `unique_indices` 中每个元素，在 `[C, W]` 的 cache 状态上完成查找与 LRU 更新逻辑；`cache_sets` 初值为哨兵 `C`（与 CUDA 一致）。Host 侧对 `cache_sets` 与 `unique_indices` 的 **SortPairs**（得到 `sorted_cache_sets` / `cache_set_sorted_unique_indices`）由 PyTorch 适配层或其它算子完成，本 AscendC 算子仅覆盖 **find** 内核语义。

与 CUDA 的差异简述：

- 当前 **`SetBlockDim(1)`**，单核串行；无多 warp / `__any_sync`。
- `unique_indices` 为 `int64` 时，可在构建中对 `op_kernel/lru_cache_find_uncached.cpp` 增加 `-DDTYPE_UNIQUE=int64_t`（与原型中类型选择一致）。

## 算子输入与输出

| 名称 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | 范围 | 说明 |
|------|---------|---------|---------|--------|------|------|
| unique_indices | 输入 | Tensor | int32 / int64 | [N] | | 与 CUDA buffer 一致的一维张量 |
| unique_indices_length | 输入 | Tensor | int32 | [1] | | `*` 即为 `K` |
| lxu_cache_state | 输入 | Tensor | int64 | [C, W] | | Cache 状态 |
| lru_state | 输入/输出 | Tensor | int64 | [C, W] | | 命中槽 LRU 时间戳原地更新 |
| uvm_cache_stats | 输入/输出 | Tensor | int32 | 一维 | `gather_cache_stats=true` 时长度 ≥ 4 | 与 FBGEMM `uvm_cache_stats_index` 写入语义一致 |
| lxu_cache_locking_counter | 输入 | Tensor | int32 | [C, W] 或 [0, 0] | | `lock_cache_line=true` 时须与 cache 同形；否则可为空形状 |
| cache_sets | 输出 | Tensor | int32 | [N] | 初值哨兵为 `C` | 每个 `unique_indices` 元素对应的 cache set |

### 属性参数（经 tiling 下发）

| 名称 | 输入(属性) | 类型 | 说明 |
|------|-----------|------|------|
| gather_cache_stats | 可选 | bool | 是否汇总 cache 统计 |
| max_indices | 必选 | int | 与 CUDA kernel 标量参数一致 |
| lru_timestamp | 必选 | int | 与 CUDA 中时间戳语义一致（OpDef 中命名为 `lru_timestamp`） |
| lock_cache_line | 可选 | bool | 是否启用 cache line 锁定计数 |

## 算子编译部署

进入本目录后执行：

```bash
chmod +x run.sh
./run.sh
```
