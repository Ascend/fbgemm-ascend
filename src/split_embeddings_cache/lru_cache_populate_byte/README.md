**使用 PyTorch 框架调用方式调用 lru_cache_populate_byte 算子**

# PyTorch 框架对外接口原型

通过 **fbgemm** 已注册的 schema 挂载 NPU 实现（不在本库重复 `m.def`）：

```python
torch.ops.fbgemm.lru_cache_populate_byte(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    int total_cache_hash_size,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor linear_cache_indices,
    Tensor(a!) lxu_cache_state,
    Tensor(b!) lxu_cache_weights,
    int time_stamp,
    Tensor(c!) lxu_state,
    int row_alignment = 16,
    bool gather_cache_stats = False,
    Tensor(d!)? uvm_cache_stats = None,
) -> ()
```

不依赖 `fbgemm_gpu` Python 包时，可使用本目录注册的 **mxrec** 接口（参数语义与上式一致，部分形参命名与 FBGEMM C++ 侧一致）：

```python
torch.ops.mxrec.lru_cache_populate_byte(
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
    int row_alignment = 16,
    bool gather_cache_stats = False,
    Tensor(d!)? uvm_cache_stats = None,
) -> ()
```

实现上将 `linear_cache_indices` 去重后依次调用 `aclnnLruCacheFindUncached`（AscendC）与 `aclnnLruCacheInsertByte`（AscendC），行为对齐 FBGEMM CUDA 入口 `lru_cache_populate_byte_cuda`（`fbgemm_gpu/src/split_embeddings_cache/lru_cache_populate_byte.cu`）。

# 参数说明

| 名称 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | 范围 | 说明 |
|------|---------|---------|---------|--------|------|------|
| weights / weights_uvm | 输入 | Tensor | uint8 | 一维 UVM 权重 | | 与 FBGEMM 一致 |
| hash_size_cumsum / cache_hash_size_cumsum | 输入 | Tensor | int64 | ND | | 累积 hash 大小 |
| total_cache_hash_size | 输入(属性) | int | int64 | 标量 | | |
| cache_index_table_map | 输入 | Tensor | int32 | ND | | |
| weights_offsets | 输入 | Tensor | int64 | ND | | |
| weights_tys | 输入 | Tensor | uint8 | ND | | 稀疏类型编码 |
| D_offsets | 输入 | Tensor | int32 | ND | | 嵌入维度偏移 |
| linear_cache_indices | 输入 | Tensor | int64 等 | 一维 | | 已线性化的缓存索引 |
| lxu_cache_state | 输入/输出 | Tensor | int64 | [C, W] | | 须 contiguous |
| lxu_cache_weights | 输入/输出 | Tensor | uint8 | 与 cache 布局一致 | | 须 contiguous |
| time_stamp | 输入(属性) | int | int64 | 标量 | | 写入 LRU 的时间戳 |
| lxu_state / lru_state | 输入/输出 | Tensor | int64 | [C, W] | | fbgemm 参数名为 `lxu_state`，与 `lru_state` 为同一逻辑张量 |
| row_alignment | 输入(属性) | int | int64 | 默认 16 | | 传入 insert 算子 |
| gather_cache_stats | 输入(属性) | bool | bool | 默认 false | | 为 true 时必须提供 `uvm_cache_stats` |
| uvm_cache_stats | 输入/输出 | Tensor，可选 | int32 | 一维 | | `gather_cache_stats=false` 时可不传，内部以空张量占位 |

# 运行算子样例

## 算子编译与部署

AscendC 子算子（`LruCacheFindUncached`、`LruCacheInsertByte`）编译部署请参考 [RecSDK\cust_op\README.md](../../../../README.md) 中「单算子使用说明」-「算子编译」章节。

## PyTorch 编译

PyTorch 适配层编译请参考 [RecSDK\cust_op\README.md](../../../../README.md) 中「单算子使用说明」-「算子适配层编译」章节。进入本目录执行：

```bash
cd lru_cache_populate_byte
bash build_ops.sh
```

集成在 `libfbgemm_npu_api.so` 中时，通常由包内路径 `load_library` 加载，无需单独加载本目录产物。

## 算子调用示例

以下示例展示加载动态库并调用 **fbgemm** 算子（需已安装 `fbgemm_gpu` 并完成 schema 注册；设备为 NPU）：

```python
from pathlib import Path

import torch
import torch_npu

torch.npu.set_device(0)
torch.ops.load_library(str(Path("path/to/build/liblru_cache_populate_byte.so").resolve()))

# 张量形状与数值需与真实 TBE 推理场景一致，此处仅作 API 形态示例。
# 详见仓库内测试用例构造方式。
torch.ops.fbgemm.lru_cache_populate_byte(
    weights_uvm,
    cache_hash_size_cumsum,
    total_cache_hash_size,
    cache_index_table_map,
    weights_offsets,
    weights_tys,
    D_offsets,
    linear_cache_indices,
    lxu_cache_state,
    lxu_cache_weights,
    timestep,
    lxu_state,
    row_alignment=16,
    gather_cache_stats=False,
    uvm_cache_stats=None,
)
```

## aclnn 底层说明

适配层通过 `EXEC_NPU_CMD(aclnnLruCachePopulateByte, ...)` 调用 `libopapi.so` 中的 `aclnnLruCachePopulateByte` / `aclnnLruCachePopulateByteGetWorkspaceSize` 时，**张量与标量顺序须与 CANN 定义一致**；若升级 CANN 后签名变化，请同步修改 `lru_cache_populate_byte.cpp` 中 `EXEC_NPU_CMD` 实参及 `GetWorkspaceSize` 调用。

当前文档化顺序（便于与源码对照）为：

1. `weights`（uint8）
2. `hash_size_cumsum`（int64）
3. `cache_index_table_map`
4. `weights_offsets`
5. `weights_tys`
6. `D_offsets`
7. `linear_cache_indices`
8. `lxu_cache_state`（in/out）
9. `lxu_cache_weights`（in/out）
10. `lru_state`（in/out）
11. `uvm_cache_stats`（in/out；不统计时为空 tensor）
12. `total_cache_hash_size`（int64）
13. `time_stamp`（int64）
14. `row_alignment`（int64）
15. `gather_cache_stats`（bool）

注：上述用例为接口形态说明；完整精度与多场景测试请参考 [test_lru_cache_populate_byte.py](../../../../test/lru_cache_populate_byte/torch/test_lru_cache_populate_byte.py)。
