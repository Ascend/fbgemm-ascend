# lru_cache_populate_byte

本算子是实现lru_cache_populate_byte的子接口，不支持直接调用。需通过lru_cache_populate_byte调用参考[lru_cache_populate_byte](../lru_cache_populate_byte/README.md)

## 产品支持情况

| 硬件型号           | 是否支持 |
|----------------|------|
| Atlas A5 | 是    |

## lru_cache_insert_byte 算子目录层级

```shell
lru_cache_insert_byte
|-- c310
   |-- op_host                        # 算子 host 侧实现
   |-- op_kernel                      # 算子 kernel 侧实现（含 SIMT 辅助头文件）
   |-- lru_cache_insert_byte.json     # 算子原型配置
   |-- run.sh                         # 算子编译部署脚本
```

# 功能

将 UVM 中的量化字节权重按 `sorted_cache_sets` / `cache_set_sorted_unique_indices` 写入 `lxu_cache_weights`，并更新 `lxu_cache_state`、`lru_state` 及可选的 `uvm_cache_stats`；语义与 **FBGEMM** `lru_cache_insert_byte_kernel`（`fbgemm_gpu/src/split_embeddings_cache/lru_cache_populate_byte.cu`）对齐。

更完整的设计说明见 [`docs/lru_cache_insert_byte_DETAILED_DESIGN.md`](docs/lru_cache_insert_byte_DETAILED_DESIGN.md)（按 ascendc-ops-architect 详细设计模板）。

# 算子实现原理

依据 `cache_hash_size_cumsum`、`cache_index_table_map`、`weights_offsets`、`weights_tys`、`d_offsets` 等元数据，从一维 `weights`（uint8）中解析各行并写入 cache；输入的 `sorted_cache_sets` 与 `cache_set_sorted_unique_indices` 须已由上游完成排序配对（与 CUDA 路径中 CUB SortPairs 之后的数据一致）。

`cache_set_sorted_unique_indices` 为 **int32** 时，可在构建中对 `op_kernel/lru_cache_insert_byte.cpp` 使用 `-DDTYPE_INDEX=int32_t`（默认 int64）。

# 算子输入与输出

| 名称 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | 范围 | 说明 |
|------|---------|---------|---------|--------|------|------|
| weights | 输入 | Tensor | uint8 | 一维 | | UVM 侧字节权重 |
| cache_hash_size_cumsum | 输入 | Tensor | int64 | ND | | 与 FBGEMM 一致 |
| cache_index_table_map | 输入 | Tensor | int32 | ND | | |
| weights_offsets | 输入 | Tensor | int64 | ND | | |
| weights_tys | 输入 | Tensor | uint8 | ND | | 表类型编码 |
| d_offsets | 输入 | Tensor | int32 | ND | | 维度偏移 |
| sorted_cache_sets | 输入 | Tensor | int32 | 一维 | | 已排序的 cache set |
| cache_set_sorted_unique_indices | 输入 | Tensor | int32 / int64 | 一维 | | 与 sorted_cache_sets 对齐的去重索引 |
| unique_indices_length | 输入 | Tensor | int32 | ND | | 长度张量，与 find 阶段一致 |
| lxu_cache_state | 输入/输出 | Tensor | int64 | [C, W] | | |
| lxu_cache_weights | 输入/输出 | Tensor | uint8 | [C, row_bytes] | | Cache 权重存储 |
| lru_state | 输入/输出 | Tensor | int64 | [C, W] | | |
| uvm_cache_stats | 输入/输出 | Tensor | int32 | 一维 | `gather_cache_stats=false` 时由调用方保证合法形状 | |
| reserved_out | 输出 | Tensor | int32 | 占位 | | 与 aclnn / 插件约定的一致占位输出 |

## 属性参数（经 tiling 下发）

| 名称 | 输入(属性) | 类型 | 说明 |
|------|-----------|------|------|
| gather_cache_stats | 可选 | bool | 默认 false |
| lru_timestamp | 必选 | int | LRU 时间戳 |
| row_alignment | 必选 | int | 行对齐，默认 16 |

# 算子编译部署

算子编译请参考 [RecSDK\cust_op\README.md](../../../../README.md) 中「单算子使用说明」-「算子编译」章节。

进入本目录后执行：

```bash
chmod +x run.sh
./run.sh
```
