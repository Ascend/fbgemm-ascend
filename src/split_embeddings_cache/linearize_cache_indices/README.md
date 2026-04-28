# linearize_cache_indices 算子文档

## 使用PyTorch框架调用linearize_cache_indices算子

本算子仅支持NPU调用。

### PyTorch框架对外接口原型

```python
torch.ops.fbgemm.linearize_cache_indices(
    Tensor cache_hash_size_cumsum,
    Tensor indices,
    Tensor offsets,
    Tensor? B_offsets=None,
    int max_B=-1,
    int indices_base_offset=0
) -> Tensor
```

#### 参数说明
| 名称 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | 范围 | 说明 |
|------|----|----|----|----|----|----|
| cache_hash_size_cumsum | 输入 | Tensor | int64 | torch.tensor([value1, value2, value3 ...]) | 长度:[1, 2^63-1) | 累积哈希大小数组，单调递增 |
| indices | 输入 | Tensor | int32/int64 | torch.tensor([value1, value2, value3 ...]) | 长度:[1, 2^63-1) | 需要线性化的索引 |
| offsets | 输入 | Tensor | int32/int64 | torch.tensor([value1, value2, value3 ...]) | 长度:[1, 2^63-1) | 表边界偏移量数组 |
| B_offsets | 输入(可选) | Tensor? | int32/int64 | torch.tensor([value1, value2, value3 ...]) | 长度:[1, 2^63-1) | PTA模式下指示当前批次各表在offsets中的位置 |
| max_B | 输入 | int | int | - | max_B >= 0 | 最大批量大小，用于定义offsets维度，PTA模式下必须提供 |
| indices_base_offset | 输入 | int | int | - | 任意 | 索引基础偏移量 |
| linearized_indices | 输出 | Tensor | int64 | torch.tensor([value1, value2, value3 ...]) | - | 线性化后的索引 |

### 算子调用示例
```python
import torch
import fbgemm_gpu

def test_linearize_cache_indices():
    # 假设有两个嵌入表，大小分别为 [100, 200]
    # PTA 模式下，offsets 通常预分配为最大可能大小
    # B_offsets 用于指示当前批次中每个表的实际偏移位置
    cache_hash_size_cumsum = torch.tensor([0, 100, 300], dtype=torch.long, device='npu')
    indices = torch.tensor([10, 50, 150], dtype=torch.long, device='npu')

    # 假设 max_B=2, num_tables=2, offsets 长度为 2*2+1=5
    # 这里仅为示例，实际 offsets 内容取决于具体批次数据布局
    offsets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long, device='npu') 
    # B_offsets 指示每个表在当前批次 offsets 中的起始索引
    # 例如表0从 offsets[0] 开始，表1从 offsets[2] 开始
    B_offsets = torch.tensor([0, 2, 4], dtype=torch.long, device='npu')

    # 线性化索引 (PTA 模式)
    linearized_indices = torch.ops.fbgemm.linearize_cache_indices(
        cache_hash_size_cumsum,
        indices,
        offsets,
        B_offsets=B_offsets,
        max_B=2,
        indices_base_offset=0
    )
    
    # 验证结果
    print(linearized_indices)
```

注：更详细精度、多场景测试用例请参考用例[benchmarks](../../../bench/split_embeddings_cache/linearize_cache_indices_test/test_linearize_cache_indices.py)。