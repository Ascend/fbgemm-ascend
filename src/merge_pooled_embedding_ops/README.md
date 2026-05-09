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

```python
torch.ops.fbgemm.all_to_one_device(
    Tensor[] input_tensors,
    Device target_device
) -> Tensor[]
```

## 功能说明

实现将多个NPU设备上的张量合并到目标设备上。

## 参数说明

| 名称 | 输入/输出 | 参数类型 | 数据类型 | 数据格式 | 范围 | 说明 |
| ------ | ---- | ---- | ---- | ---- | ---- | ---- |
| input_tensors | 输入 | Tensor[] | float16/bfloat16/float32等 | - | - | 输入张量列表，每个张量可以来自不同的NPU设备 |
| target_device | 输入 | Device | - | - | - | 目标设备类型"npu:0"、"npu:1"等或"cpu" |
| output_tensors | 输出 | Tensor[] | - | - | - | 输出张量列表, 每个张量都为target_device下的张量 |

### 参数约束

- input_tensors中的每个张量必须是NPU设备上的张量

## 算子调用示例

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
