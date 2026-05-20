# merge_pooled_embedding

本算子仅支持NPU调用。

## 目录结构

```text
merge_pooled_embedding_ops
|-- merge_pooled_embedding_ops_npu.cpp
|-- README.md
```

## 硬件支持情况

| 实现目录 | 典型硬件 |
| --- | --- |
| - | Atlas A5 训练系列 |

## 接口定义

### merge_pooled_embeddings

```python
torch.ops.fbgemm.merge_pooled_embeddings(
    Tensor[] pooled_embeddings,
    int uncat_dim_size,
    Device target_device,
    int cat_dim = 1
) -> Tensor
```

### all_to_one_device

```python
torch.ops.fbgemm.all_to_one_device(
    Tensor[] input_tensors,
    Device target_device
) -> Tensor[]
```

## 功能说明

### merge_pooled_embeddings

将多个来自不同NPU设备的pooled embedding张量按指定维度拼接成一个张量。该算子会自动将不同设备上的张量汇聚到目标设备，然后进行拼接操作。

### all_to_one_device

实现将多个NPU设备上的张量合并到目标设备上。

## 参数说明

### merge_pooled_embeddings

| 名称 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | 范围 | 说明 |
| ------ | ---- | ---- | ---- | ---- | ---- | ---- |
| pooled_embeddings | 输入 | Tensor[] | float16/bfloat16/float32等 | - | - | 输入张量列表，每个张量可以来自不同的NPU设备 |
| uncat_dim_size | 输入 | int64_t | int64 | - | - | 未拼接维度的大小，所有张量在该维度上的大小必须相同 |
| target_device | 输入 | Device | - | - | - | 目标设备类型"npu:0"、"npu:1"等或"cpu" |
| cat_dim | 输入 | int64_t | int64 | - | 0或1 | 拼接维度，默认为1 |
| output | 输出 | Tensor | - | - | - | 拼接后的输出张量，位于target_device上 |

### all_to_one_device

| 名称 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | 范围 | 说明 |
| ------ | ---- | ---- | ---- | ---- | ---- | ---- |
| input_tensors | 输入 | Tensor[] | float16/bfloat16/float32等 | - | - | 输入张量列表，每个张量可以来自不同的NPU设备 |
| target_device | 输入 | Device | - | - | - | 目标设备类型"npu:0"、"npu:1"等或"cpu" |
| output_tensors | 输出 | Tensor[] | - | - | - | 输出张量列表, 每个张量都为target_device下的张量 |

### 参数约束

- pooled_embeddings/input_tensors中的每个张量必须是NPU设备上的张量
- 对于merge_pooled_embeddings，所有张量在非拼接维度上的大小必须相同

## 算子调用示例

### merge_pooled_embeddings

```python
import torch
import fbgemm_gpu
import fbgemm_ascend

def test_merge_pooled_embeddings():
    num_embds = 4
    embedding_dim = 128
    batch_size = 32

    # 假设有4个NPU设备，每个设备上有部分embedding
    npu_tensors = []
    for i in range(num_embds):
        with torch.npu.device(f"npu:{i}"):
            tensor = torch.randn(batch_size, embedding_dim, dtype=torch.float16)
            npu_tensors.append(tensor)

    # 合并到npu:0设备
    dstDevice = torch.device("npu:0")
    uncat_dim_size = batch_size
    merged = torch.ops.fbgemm.merge_pooled_embeddings(
        npu_tensors,
        uncat_dim_size,
        dstDevice,
        cat_dim=1
    )
    print(f"Merged shape: {merged.shape}")  # (32, 512)
```

### all_to_one_device

```python
import torch
import fbgemm_gpu
import fbgemm_ascend

def test_all_to_one_device():
    numNpus = 8
    dstDevice = torch.device("npu:0")
    with torch.npu.device(dstDevice):
        inputs = [torch.randn(10, 20, dtype=dtype, device="cpu") for _ in range(numNpus)]
        npu_inputs = [
            input.to(f"npu:{i % numNpus}") for i, input in enumerate(inputs)
        ]
        npu_outpus = torch.ops.fbgemm.all_to_one_device(npu_inputs, dstDevice)
        for i, o in zip(inputs, npu_outpus):
            print(i)
            print(o)
```

## 编译与测试

- Ascend C 算子编译与适配层编译参考仓库根目录下的[README.md](../../README.md)。
- 更详细精度、多场景测试用例请参考用例[benchmarks](../../bench/merge_pooled_embeddings_test.py)
