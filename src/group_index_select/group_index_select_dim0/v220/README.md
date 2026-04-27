# group_index_select_dim0 算子说明

本算子仅支持NPU调用。

## 产品支持情况
| 硬件型号              | 是否支持                  |
| -------------------- | ------------------------ |
| Atlas A5训练系列产品  | 是  |

## group_index_select_dim0算子目录层级

```shell
-- group_index_select_dim0
   |-- v220
      |-- op_host                 # 算子host侧实现
      |-- op_kernel               # 算子kernel侧实现
      |-- group_index_select_dim0.json   # 算子原型配置
      |-- README.md               # 算子说明文档
      |-- run.sh                  # 算子编译部署脚本
```

## 功能

实现批量从多个数据表中按行号挑选数据的功能，等价于对每个表独立执行torch.index_select(input, 0, indices)操作。

## 算子实现原理

算子工作原理说明：
1. 输入多个数据表（input_group），每个表是一个二维张量，形状为(num_input_rows, *shape)，其中num_input_rows范围为num_input_rows>=1，shape每个维度范围为1~32
2. 输入多个索引张量（indices_group），每个是一维张量，包含num_indices个索引值，范围为1~32
3. 算子根据每个表对应的索引，从表中挑选对应的行
4. 输出张量output_group的每个元素output_group[i][j] = input_group[i][indices_group[i][j]]

## 举例说明

输入示例：
```python
# 表0：3行，每行4个元素
input_group[0] = [
    [a0, a1, a2, a3],  # 第0行
    [b0, b1, b2, b3],  # 第1行
    [c0, c1, c2, c3]   # 第2行
]

# 表0的索引
indices_group[0] = [0, 2, 1]  # 取第0行、第2行、第1行

# 输出
output_group[0] = [
    [a0, a1, a2, a3],  # 第0行
    [c0, c1, c2, c3],  # 第2行
    [b0, b1, b2, b3]   # 第1行
]
```

## 输出
```python
output_group = torch.ops.fbgemm.group_index_select_dim0(input_group, indices_group)
```

## 算子输入与输出
|  名称  |  输入/输出  |  数据类型  |  数据格式  |  范围  |  说明  |
|  ---- |  ---- |  ----  |  ----  |  ----  |  ----  |
|  input_group  |  输入  |  float16/float32  |  [(num_input_rows, *shape)]  |  num_groups: 1~32  |  多个数据表的列表，每个表形状为(num_input_rows, *shape)，num_input_rows>=1，shape每个维度: 1~32  |
|  indices_group  |  输入  |  int64	|  (num_indices,)  |  num_indices: 1~num_input_rows  |  多个索引张量的列表，每个索引张量为一维，元素值需小于对应表的num_input_rows  |
|  output_group  |  输出  |  float16/float32  |  [(num_indices, *shape)]  |  NA  |  多个输出张量的列表，每个形状为(num_indices, *shape)  |


## 算子编译部署

算子编译请参考[README.md](../../../../README.md)中"源码编译与安装"章节。
