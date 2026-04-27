# group_index_select_dim0_backward 算子说明

本算子仅支持NPU调用。

## 产品支持情况
| 硬件型号              | 是否支持                  |
| -------------------- | ------------------------ |
| Atlas A2训练系列产品  | 是  |
| Atlas A5训练系列产品  | 是  |

## group_index_select_dim0_backward算子目录层级

```shell
-- group_index_select_dim0_backward
   |-- v220
      |-- op_host                 # 算子host侧实现
      |-- op_kernel               # 算子kernel侧实现
      |-- group_index_select_dim0_backward.json   # 算子原型配置
      |-- README.md               # 算子说明文档
      |-- run.sh                  # 算子编译部署脚本
```

## 功能

实现group_index_select_dim0前向算子的反向传播。根据输出梯度grad_outputs和索引indices_group，计算输入input_group的梯度grad_inputs。

## 算子实现原理

算子工作原理说明：
1. 输入多个输出梯度张量（grad_outputs），每个对应前向输出output_group的梯度，形状为(num_indices, *shape)
2. 输入多个索引张量（indices_group），每个是一维张量，包含num_indices个索引值，与正向算子使用的索引相同
3. 算子根据索引将grad_outputs中的梯度值累加到对应的行位置，生成input_return_group，与正向算子使用的shape相同

## 举例说明

前向示例：
```python
# 前向输入
input_group[0] = [
    [a0, a1, a2, a3],  # 第0行
    [b0, b1, b2, b3],  # 第1行
    [c0, c1, c2, c3]   # 第2行
]
indices_group[0] = [0, 2, 1]  # 取第0行、第2行、第1行

# 前向输出
output_group[0] = [
    [a0, a1, a2, a3],  # 第0行
    [c0, c1, c2, c3],  # 第2行
    [b0, b1, b2, b3]   # 第1行
]
```

反向示例：
```python
# 假设输出梯度
grad_outputs[0] = [
    [d0, d1, d2, d3],  # 对应第0行输出
    [e0, e1, e2, e3],  # 对应第2行输出
    [f0, f1, f2, f3]   # 对应第1行输出
]
indices_group[0] = [0, 2, 1]

# 反向计算：将梯度累加到输入对应的行
# 第0行输入接收索引0对应的梯度
# 第1行输入接收索引1对应的梯度
# 第2行输入接收索引2对应的梯度
input_return_group[0] = [
    [d0, d1, d2, d3],  # 第0行
    [f0, f1, f2, f3],  # 第1行
    [e0, e1, e2, e3]   # 第2行
]
```

## 算子输入与输出
|  名称  |  输入/输出  |  数据类型  |  数据格式  |  范围  |  说明  |
|  ---- |  ---- |  ----  |  ----  |  ----  |  ----  |
|  grad_outputs  |  输入  |  float16/float32  |  [(num_indices, *shape)]  |  num_groups: 1~32  |  多个输出梯度张量的列表，每个形状为(num_indices, *shape)，对应前向输出output_group中每个元素的梯度  |
|  indices_group  |  输入  |  int64	|  (num_indices,)  |  num_indices: 1~num_input_rows  |  多个索引张量的列表，每个索引张量为一维，与前向算子使用的索引相同，元素值需小于对应表的num_input_rows  |
|  input_return_group  |  输出  |  float16/float32  |  [(num_input_rows, *shape)]  |  NA  |  多个输入梯度张量的列表，每个形状为(num_input_rows, *shape)，每个元素是累积后的梯度值  |

## 注意事项

1. 本算子为group_index_select_dim0的反向实现，需与前向算子配合使用
2. 如果indices_group中存在重复索引，对应的grad_outputs会累加到同一行
3. 输出input_return_group的形状与前向的输入input_group完全一致

## 算子编译部署

算子编译请参考[README.md](../../../../README.md)中"源码编译与安装"章节。