# PrunedArrayLookupFromRowIdx

本算子仅支持 NPU 调用。


# 目录结构

```shell
-- pruned_array_lookup_from_row_idx
   |-- c310
      |-- op_host                                   # 算子 host 侧实现
      |-- op_kernel                                 # 算子 kernel 侧实现
      |-- pruned_array_lookup_from_row_idx.json
      |-- README.md
      |-- run.sh
```

## 功能说明

- 算子功能：与 FBGEMM `torch.ops.fbgemm.pruned_array_lookup_from_row_idx` 对齐。对每条待更新索引 `i`，根据表号 `update_table_indices[i]` 在 `index_remappings_offsets` 中取该表的稠密映射段 `[start, end)`；若段长度大于 0，则在 `index_remappings` 中按逻辑行号 `update_row_indices[i]` 查得稠密行号写入输出；若该表映射段长度为 0，则输出透传逻辑行号 `update_row_indices[i]`。

- Python 伪代码实现：

```python
def pruned_array_lookup_from_row_idx(
    update_row_indices,
    update_table_indices,
    index_remappings,
    index_remappings_offsets,
):
    """
    按表将逻辑行号映射为稠密行号；无映射段时透传逻辑行号。

    Args:
        update_row_indices: 逻辑行号，一维，长度 N
        update_table_indices: 表索引 int32，一维，长度 N
        index_remappings: 展平后的稠密行号映射，一维
        index_remappings_offsets: 各表在 index_remappings 中的段起点，int64，长度 T+1

    Returns:
        dense_indices: 与 update_row_indices 同 dtype、同长度 N 的一维张量
    """
    n = update_row_indices.numel()
    dense_indices = empty_like(update_row_indices)
    for i in range(n):
        r = int(update_row_indices[i])
        t = int(update_table_indices[i])
        start = int(index_remappings_offsets[t])
        end = int(index_remappings_offsets[t + 1])
        if end > start:
            dense_indices[i] = index_remappings[start + r]
        else:
            dense_indices[i] = r
    return dense_indices
```

## 参数说明

<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 500px">
  <col style="width: 250px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>update_row_indices</td>
      <td>输入</td>
      <td>每条更新对应的逻辑行号，与 FBGEMM 语义一致。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>update_table_indices</td>
      <td>输入</td>
      <td>每条更新对应的表索引。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index_remappings</td>
      <td>输入</td>
      <td>各表稠密行号拼接成的一维映射缓冲。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index_remappings_offsets</td>
      <td>输入</td>
      <td>各表在 index_remappings 中的段边界，长度为表数加 1。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dense_indices</td>
      <td>输出</td>
      <td>与公式中的 dense_indices 一致；dtype 与 update_row_indices 一致。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `update_row_indices` 与 `update_table_indices` 须为一维且长度相同。
- `index_remappings`、`index_remappings_offsets` 须为一维；`index_remappings_offsets` 为 int64。
- 用户需自行保证索引在有效范围内，且各张量长度与数值不超过对应数据类型的表示范围。

## 编译与部署

参考 RecSDK/cust_op/README.md “单算子使用说明”章节的编译、适配层部署流程。可在本目录 `c310` 下执行 `bash run.sh`（默认 `ai_core-Ascend950`）完成 AscendC 算子编译安装；自定义算子需安装到 `ASCEND_CUSTOM_OPP_PATH` 所指向的 vendor 包中。

更多 PyTorch 调用与精度校验见 `bench/embedding_inplace/pruned_array_lookup_from_row_idx_test/`；适配层入口见上级目录 `pruned_array_lookup_from_row_idx.cpp`（`EXEC_NPU_CMD(aclnnPrunedArrayLookupFromRowIdx, ...)`）。
