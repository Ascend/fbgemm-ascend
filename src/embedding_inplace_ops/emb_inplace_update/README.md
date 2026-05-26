# **emb_inplace_update**

本算子仅支持NPU调用。

## emb_inplace_update算子目录层级
```shell
-- emb_inplace_update
   |-- c310
      |-- op_host                            # 算子host侧实现
      |-- op_kernel                          # 算子kernel侧实现
      |-- emb_inplace_update.json            # 算子原型配置
      |-- run.sh                             # 算子编译部署脚本
   |-- README.md                          # 算子说明文档
```
## 硬件支持情况
| 实现目录              | 典型硬件                  |
| -------------------- | ------------------------ |
| c310/     | Atlas A5 训练系列     |

## 接口定义
```
torch.ops.fbgemm.emb_inplace_update(
           Tensor(a!) dev_weights,
           Tensor(b!) uvm_weights,
           Tensor weights_placements,
           Tensor weights_offsets,
           Tensor weights_tys,
           Tensor D_offsets,
           Tensor update_weights,
           Tensor update_table_indices,
           Tensor update_row_indices,
           Tensor update_offsets,
           int row_alignment,
           Tensor? lxu_cache_weights=None,
           Tensor? lxu_cache_locations=None
           ) -> ()
```
## 功能说明

- 用于将稀疏更新中的若干 `(table_id, row_id, payload)` 三元组按字节级 scatter 写回 TBE（Table Batched Embedding）格式的扁平化权重张量
- 是 `torch.ops.fbgemm.emb_inplace_update` 的 NPU 适配实现，主要用于 INT-NBit TBE 推理场景下的权重原地更新
- 行字节数 `D_bytes` 由表级 SparseType 编码动态推导（FP32/FP16/BF16/INT8/INT4/INT2/FP8 等），与 FBGEMM 一致

### 仿真/伪代码

```python
def emb_inplace_update(dev_weights, uvm_weights, weights_placements, weights_offsets,
                      weights_tys, D_offsets, update_weights, update_table_indices,
                      update_row_indices, update_offsets, row_alignment):
    N = len(update_row_indices)
    for n in range(N):
        t = update_table_indices[n]
        r = update_row_indices[n]
        D = D_offsets[t + 1] - D_offsets[t]
        D_bytes = padded_row_size_in_bytes(D, weights_tys[t], row_alignment)
        # 根据 placement 选择目标 buffer
        if weights_placements[t] == HOST:  # HOST = 3
            dst = dev_weights
        else:
            dst = uvm_weights
        # 字节级 scatter 拷贝
        dst_offset = weights_offsets[t] + D_bytes * r
        src_offset = update_offsets[n]
        memcpy(dst[dst_offset:dst_offset + D_bytes],
               update_weights[src_offset:src_offset + D_bytes])
```

## 简述主流程

1. 对每条更新记录 `n`，读取目标表 id `t = update_table_indices[n]` 与表内行 id `r = update_row_indices[n]`
2. 由 `D_offsets[t+1] - D_offsets[t]` 计算该表的 embedding 维度 `D`，再结合 `weights_tys[t]` 与 `row_alignment` 推导该行字节数 `D_bytes`
3. 根据 `weights_placements[t]` 选择目标 buffer：`HOST(3)` 写入 `dev_weights`，其它写入 `uvm_weights`
4. 以 `weights_offsets[t] + D_bytes * r` 为目标偏移、`update_offsets[n]` 为源偏移，进行 `D_bytes` 字节的 scatter 拷贝

## 参数说明

|  名称  |  输入/输出  |  数据类型  |  数据格式  |  说明  |
|  ---- |  ---- |  ----  |  ----  |  ----  |
|  dev_weights | 输入/输出（in-place） | uint8 | ND [N_dev] | 设备端 TBE 权重字节流 |
|  uvm_weights | 输入/输出（in-place） | uint8 | ND [N_uvm] | UVM 权重字节流，保留参数暂不支持 |
|  weights_placements | 输入 | int32 | ND [T] | 各表的 placement 编码（0/1/2=非 HOST，3=HOST） |
|  weights_offsets | 输入 | int64 | ND [T] | 各表在权重字节流中的起始偏移 |
|  weights_tys | 输入 | uint8 | ND [T] | 各表的 SparseType 编码 |
|  D_offsets | 输入 | int32 | ND [T+1] | 各表 embedding 维度的前缀和 |
|  update_weights | 输入 | uint8 | ND [N_upd] | 待更新权重字节流 |
|  update_table_indices | 输入 | int32 | ND [N] | 每条记录的目标表 id |
|  update_row_indices | 输入 | int32, int64 | ND [N] | 每条记录在表内的行 id |
|  update_offsets | 输入 | int64 | ND [N+1] | 每条记录在 update_weights 中的字节偏移 |
|  row_alignment | 输入（属性） | int | - | 行字节对齐数（GPU 默认 16，NPU 推荐 32），默认 1 |
|  lxu_cache_weights | 输入（可选） | - | - | 当前 NPU 推理不使用，保留对齐 fbgemm schema |
|  lxu_cache_locations | 输入（可选） | - | - | 当前 NPU 推理不使用，保留对齐 fbgemm schema |


### 参数约束
- `update_row_indices` 必须为一维张量，dtype ∈ {int32, int64}
- `update_table_indices` 必须为一维 int32 张量，长度与 `update_row_indices` 一致
- `update_offsets` 长度为 N 或 N+1（N 为更新记录数）
- `dev_weights`、`uvm_weights`、`update_weights`、`weights_tys` 必须为 uint8 张量
- 当 `update_row_indices` 长度为 0 时，直接返回，不进行任何写入
- 表的`embedding dim` 大小没有内置上限，受npu显存限制
- 当前uvm昇腾暂不支持,只保留参数
- 当前阶段忽略 `lxu_cache_weights / lxu_cache_locations`（NPU 推理不使用 LXU 缓存机制）

### 算子调用示例
```python
import torch
import torch_npu
import fbgemm_ascend

def test_emb_inplace_update():
    # 假设有 2 张表，embedding 维度均为 4，dtype 为 FP32（4 字节）
    # row_alignment = 1，则每行字节数 D_bytes = 4 * 4 = 16
    T = 2
    D = 4
    row_alignment = 1
    rows_per_table = [3, 2]

    # 表 0 占 3*16=48 字节，表 1 占 2*16=32 字节，共 80 字节
    dev_weights = torch.zeros(80, dtype=torch.uint8, device='npu')
    uvm_weights = torch.zeros(0, dtype=torch.uint8, device='npu')

    # placement=3(HOST) 表示写入 dev_weights
    weights_placements = torch.tensor([3, 3], dtype=torch.int32, device='npu')
    weights_offsets = torch.tensor([0, 48], dtype=torch.int64, device='npu')
    # SparseType 编码：FP32 在 fbgemm 中为 0
    weights_tys = torch.tensor([0, 0], dtype=torch.uint8, device='npu')
    D_offsets = torch.tensor([0, 4, 8], dtype=torch.int32, device='npu')

    # 两条更新记录：表 0 的第 1 行、表 1 的第 0 行
    update_table_indices = torch.tensor([0, 1], dtype=torch.int32, device='npu')
    update_row_indices = torch.tensor([1, 0], dtype=torch.int32, device='npu')
    # 每条记录 16 字节，update_offsets 长度 N+1
    update_offsets = torch.tensor([0, 16, 32], dtype=torch.int64, device='npu')
    update_weights = torch.arange(32, dtype=torch.uint8, device='npu')

    torch.ops.fbgemm.emb_inplace_update(
        dev_weights, uvm_weights,
        weights_placements, weights_offsets, weights_tys, D_offsets,
        update_weights, update_table_indices, update_row_indices, update_offsets,
        row_alignment,
    )

    # 预期：dev_weights[16:32]   = update_weights[0:16]   （表 0 第 1 行）
    #       dev_weights[48:64]   = update_weights[16:32]  （表 1 第 0 行）
    print(dev_weights)
```

## 编译与测试
- Ascend C 算子编译与适配层编译参考仓库根目录 README.md
   - 测试示例参考：bench/.../test_xxx.py（或 test/...xxx.py）
