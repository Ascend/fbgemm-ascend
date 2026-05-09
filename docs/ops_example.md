# FBGEMM-Ascend 算子执行样例

## 简介

fbgemm-ascend 是 FBGEMM 算子在昇腾 NPU 平台上的算子实现，通过 `torch.ops.fbgemm.*` 提供高性能稀疏/稠密算子，帮助推荐、搜索等场景在 Ascend 设备上获得与 GPU 同步的训练体验。项目目标是承接社区 [FBGEMM](https://github.com/pytorch/FBGEMM) 的新能力，并针对 Ascend AI Core 进行深度调优。

详细介绍请见[FBGEMM-Ascend](https://gitcode.com/Ascend/fbgemm-ascend)

## 概述

以`permute_pooled_embs`算子为例演示算子执行过程

`permute_pooled_embs` 是用于对池化后的嵌入（pooled embeddings）输出进行特征维度重排列的核心算子。该算子主要用于推荐系统和深度学习场景中，当需要对多个特征的嵌入输出进行重新排序时使用。

### 主要功能
- 对嵌入输出张量沿特征维度进行重排列
- 支持 float32、float16、bfloat16 数据类型
- 支持 CPU 和 NPU 两种计算后端
- 针对 NPU 进行了 SIMD 并行优化

### 应用场景
- **分布式训练**：在不同设备间重新组织嵌入特征
- **特征重组**：根据业务需求调整特征顺序
- **模型并行**：在模型并行训练中重新分配特征

## 接口定义

### 算子输入与输出

| 名称                 | 输入/输出 | 参数类型 | 数据类型                      | 数据格式                    | 范围/说明 |
|---------------------|----------|----------|------------------------------|----------------------------|----------|
| pooled_embs          | 输入     | Tensor   | float32/float16/bfloat16     | [B_local, total_global_D]  | 池化后的嵌入输出张量。B_local为batch size，total_global_D为所有特征的embedding维度之和 |
| offset_dim_list      | 输入     | Tensor   | int64                         | [T+1]                      | 每个特征embedding维度的累积和。offset_dim_list[0]=0, offset_dim_list[T]=total_global_D |
| permute_list         | 输入     | Tensor   | int64                         | [T]                        | 输出特征顺序，值范围[0, T-1]，不可重复 |
| inv_offset_dim_list  | 输入     | Tensor   | int64                         | [T+1]                      | 重排后特征embedding维度的累积和 |
| inv_permute_list     | 输入     | Tensor   | int64                         | [T]                        | permute_list的逆索引/逆排列 |
| permuted_pooled_embs | 输出     | Tensor   | float32/float16/bfloat16     | [B_local, total_global_D]  | 重排列后的嵌入张量。与输入pooled_embs形状一致，特征列顺序发生改变 |

## 算法原理

### 核心思想
对每个特征块独立执行重排列，通过预计算的偏移列表确定每个特征在输入和输出张量中的位置范围。

## 文件结构
```
fbgemm-ascend/src/permute_pooled_embs_ops/
├── c310/
│   └── run.sh                         # 编译脚本
├── v220/
│   ├── op_host/                       # Host侧实现
│   │   ├── permute_pooled_embs_tiling.h
│   │   └── permute_pooled_embs.cpp
│   ├── op_kernel/                     # Kernel侧实现
│   │   ├── permute_pooled_embs_kernel.h
│   │   └── permute_pooled_embs.cpp
│   ├── permute_pooled_embs.json       # 算子配置
│   ├── README.md                      # 说明文档
│   └── run.sh                         # 编译脚本
└── permute_pooled_embs.cpp            # 适配层
```

## 环境部署

项目可在 Atlas A2/A3/A5 训练系列产品上运行，推荐的基础环境如下：

- 操作系统：Ubuntu 22.04，或其它 CANN 官方支持的 Linux 发行版。
- Python：>= 3.8。
- PyTorch：与 `torch_npu` 版本匹配的官方/Ascend 分发包。
- CANN toolkit：例如 `cann-9.0.T501`，需包含 `Ascend-cann-toolkit` 与编译依赖。
- 构建依赖：`scikit-build`、`cmake`、`ninja`、`gcc/g++`（建议 9.x 或 10.x）。

部署建议：

1. 安装对应硬件驱动、固件与基础软件，确保 `npu-smi info` 可识别芯片与健康状态。
2. 按照 CANN 文档安装 toolkit 与 `torch_npu`，并在 shell 中 `source ${CANN_PATH}/set_env.sh` 以注入编译/运行所需环境变量。
3. 准备 Python 虚拟环境，执行 `pip install -r requirements.txt` 以补齐构建依赖。
4. 安装系统 Ninja（Ubuntu: `apt-get install -y ninja-build`，CentOS/RHEL: `yum install -y ninja-build`）。

项目附带 `build_whl.sh`，会清理 `_skbuild/`、`dist/` 等缓存并执行：

```bash
bash build_whl.sh
```

生成的 whl 包位于 `dist/`，内容与源码安装完全一致，可直接 `pip install dist/fbgemm_ascend-*.whl` 部署。

## 算子编译部署

### 源码下载

```bash
git clone https://gitcode.com/Ascend/fbgemm-ascend.git
cd fbgemm-ascend
```

### 源码安装

如需安装前确保依赖完备，可先执行：

```bash
pip install -r requirements.txt
```
项目默认使用 `scikit-build` + `cmake` 构建，可以直接从源码安装：

```bash
pip install . --no-build-isolation
```

### 环境设置

安装后，默认无需额外环境配置：

- `import fbgemm_ascend` 会在当前进程自动刷新 `ASCEND_CUSTOM_OPP_PATH`，使包内 AscendC 自定义算子对 CANN runtime 可见。

如需在 shell 级别预先设置环境（供其他进程复用），可以手动 source 环境脚本：

```bash
source $(python3 -c "import fbgemm_ascend; print(fbgemm_ascend.env_setup_path())")
```

## 单算子验证

### 算子调用方式

```python
import fbgemm_ascend

# torch.ops.fbgemm.permute_pooled_embs 算子即可在 NPU 上调用
```
### 算子调用接口

```python
torch.ops.fbgemm.permute_pooled_embs(
    pooled_embs,         # 输入 embedding
    offset_dim_list,     # 维度偏移
    permute_list,        # 排列顺序
    inv_offset_dim_list, # 逆排列维度偏移
    inv_permute_list,    # 逆排列顺序
) -> Tensor
```

### 算子调用示例

进入`permute_pooled_embs`算子测试脚本所在目录：

```bash
cd fbgemm-ascend/bench/pooled_embedding/permute_pooled_embs
```

以pytest方式调用为例：

```bash
pytest test_permute_pooled_embs.py
```