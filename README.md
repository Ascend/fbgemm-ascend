# fbgemm-ascend

<div align="center">

[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Ascend/fbgemm-ascend)&nbsp;&nbsp;&nbsp;&nbsp;
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/Ascend/fbgemm-ascend)

</div>

## 简介

fbgemm-ascend 是 FBGEMM 算子在昇腾 NPU 平台上的算子实现，通过 `torch.ops.fbgemm.*` 提供高性能稀疏/稠密算子，帮助推荐、搜索等场景在 Ascend 设备上获得与 GPU 同步的训练体验。项目目标是承接社区 [FBGEMM](https://github.com/pytorch/FBGEMM) 的新能力，并针对 Ascend AI Core 进行深度调优。

- **Ascend 定制算子**：提供 AscendC 实现的核心推荐算子，并向上提供 Python 绑定。
- **PyTorch 生态无缝集成**：与 Torch、TorchRec 等组件协同，直接复用 `torch.ops.fbgemm.*` 接口。
- **多芯片自适应**：自动探测 Atlas A2/A3/A5 训练系列产品芯片，区分编译目标。

## 目录结构

```
fbgemm-ascend/                                 # 项目根目录
|-- bench/                                     # 推荐算子性能基准测试脚本
|-- cmake/                                     # CMake 模块、工具链与宏定义
|-- codegen/                                   # AscendC 模板生成算子（待补充
|-- docs/                                      # 设计/使用文档（待补充
|-- fbgemm_ascend/                             # Python 包入口，封装环境探测逻辑
|-- include/                                   # C++/AscendC 头文件
|-- src/                                       # 自定义算子实现、注册源码
|-- test/                                      # Python 端单元测试（待补充
|-- FbgemmAscend.cmake                         # 顶层 CMake 入口
|-- requirements.txt                           # Python 依赖
|-- setup.py                                   # Python 构建脚本
```

## API 文档说明

| 模块 | 文档                                                                                                                                                                      |
| ---- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| src/intraining_embedding_pruning_ops/init_address_lookup | [c310](src/intraining_embedding_pruning_ops/init_address_lookup/c310/README.md)                                                                                         |
| src/jagged_tensor_ops/dense_to_jagged | [v220](src/jagged_tensor_ops/dense_to_jagged/v220/README.md)                                                                                                            |
| src/jagged_tensor_ops/jagged_to_padded_dense | [v220](src/jagged_tensor_ops/jagged_to_padded_dense/v220/README.md)                                                                                                     |
| src/jagged_tensor_ops/select_dim1_to_permute | [c310](src/jagged_tensor_ops/select_dim1_to_permute/c310/README.md)                                                                                                     |
| src/pooled_embedding_ops/permute_pooled_embs | [v220](src/pooled_embedding_ops/permute_pooled_embs/v220/README.md)                                                                                                     |
| src/sparse_ops/asynchronous_complete_cumsum | [c310](src/sparse_ops/asynchronous_complete_cumsum/c310/README.md) / [v220](src/sparse_ops/asynchronous_complete_cumsum/v220/README.md)                                 |
| src/sparse_ops/block_bucketize_sparse_features | 无                                                                                                                                                                       |
| src/sparse_ops/expand_into_jagged_permute | [c310](src/sparse_ops/expand_into_jagged_permute/c310/README.md)                                                                                                        |
| src/sparse_ops/invert_permute | [c310](src/sparse_ops/invert_permute/c310/README.md)                                                                                                                    |
| src/sparse_ops/offsets_range | [v220](src/sparse_ops/offsets_range/v220/README.md) |
| src/sparse_ops/permute2d_sparse_data | [v220](src/sparse_ops/permute2d_sparse_data/v220/README.md)                                                                                                             |
| src/sparse_ops/segment_sum_csr | [v220](src/sparse_ops/segment_sum_csr/v220/README.md)                                                                                                                   |
| src/tbe_inference/int_nbit_split_embedding_codegen_lookup_function | [c310](src/tbe_inference/int_nbit_split_embedding_codegen_lookup_function/c310/README.md)                                                                               |
| src/tbe_training/backward_codegen_adagrad_unweighted_exact | [c310](src/tbe_training/backward_codegen_adagrad_unweighted_exact/c310/README.md) / [v220](src/tbe_training/backward_codegen_adagrad_unweighted_exact/v220/README.md)   |
| src/tbe_training/dense_embedding_codegen_lookup_function | [v220](src/tbe_training/dense_embedding_codegen_lookup_function/v220/README.md)                                                                                         |
| src/tbe_training/dense_embedding_codegen_lookup_function_grad | [v220](src/tbe_training/dense_embedding_codegen_lookup_function_grad/v220/README.md)                                                                                    |
| src/tbe_training/split_embedding_codegen_forward_unweighted | [c310](src/tbe_training/split_embedding_codegen_forward_unweighted/c310/README.md) / [v220](src/tbe_training/split_embedding_codegen_forward_unweighted/v220/README.md) |

更多算子可在对应目录的 README.md 中查看具体接口、输入输出张量格式及样例代码。

## 构建变体（A5 / A2 / A3）

- **A5（c310，Ascend95x）**：包含所有算子，适配 A5 系列芯片。
- **A2（v220，Ascend910Bx）**：基于 v220 算子，自动跳过仅支持 c310 的算子。
- **A3（v220，Ascend910_93x）**：目前只打包以下算子：`asynchronous_complete_cumsum`、`dense_to_jagged`、`jagged_to_padded_dense`、`permute_pooled_embs`、`permute2d_sparse_data`、`split_embedding_codegen_forward_unweighted`、`backward_codegen_adagrad_unweighted_exact`、`dense_embedding_codegen_lookup_function`、`dense_embedding_codegen_lookup_function_grad`。

安装包包含 `fbgemm_ascend_py_a5.so` 与 `fbgemm_ascend_py_a2a3.so` 两份适配层（分别对应 `NPU_CHIP_A5=1` 与 `0`），运行期会根据芯片加载匹配的 `.so` 并切换到对应的 `opp/A5`、`opp/A2`、`opp/A3` 目录。

默认构建会同时生成 A5/A2/A3 三套算子，若只需要其中一套，可在执行 `pip install` 或 `bash build_whl.sh` 前设置 `FBGEMM_ASCEND_BUILD_VERS=A5`（或 A2/A3）。运行期也会根据芯片自动选择目录，必要时可以通过 `FBGEMM_ASCEND_FORCE_BUILD_VER=A3` 等变量强制指定。

## 与 fbgemm_gpu 的对齐

参考仓库：[FBGEMM/fbgemm_gpu](https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu)。本项目会周期性同步该仓库更新，并保持以下一致性：

- 算子名称、Python API 语义与 CPU/GPU 的 FBGEMM 行为保持一致，便于跨设备迁移。
- 关键算子会复用上游单测思路，补充 Ascend 特有的 tiling 策略与性能调优文档。

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

## 源码编译与安装

### 源码安装

项目默认使用 `scikit-build` + `cmake` 构建，可以直接从源码安装：

```bash
pip install . --no-build-isolation
```

如需安装前确保依赖完备，可先执行：

```bash
pip install -r requirements.txt
```

### 重新安装/清理缓存

重新安装前建议清理本地缓存并卸载旧包：

```bash
rm -rf _skbuild
pip uninstall fbgemm_ascend -y
pip install . --no-build-isolation
```

### 芯片版本与自动探测

- **自动探测（推荐）**：

  - 构建默认会同时生成 A5/A2/A3 三套 AscendC 自定义算子（分别对应 Ascend95x-c310、Ascend910B系列-v220、Ascend910_93-v220），安装后会在 `fbgemm_ascend/opp/A5`、`opp/A2`、`opp/A3` 下提供 vendors 目录。
  - `import fbgemm_ascend` 会通过 `npu-smi info -m` 检测芯片型号，并在运行期把匹配的目录加入 `ASCEND_CUSTOM_OPP_PATH`，无需额外配置。

- **显式指定**：

  - 如需指定编译某一套版本，可通过设置`FBGEMM_ASCEND_BUILD_VERS=A5`（或 `A2`/`A3`）控制需要编译的版本。例如：

    ```bash
    FBGEMM_ASCEND_BUILD_VERS=A5 pip install . --no-build-isolation
    ```

### 环境设置

安装后，默认无需额外环境配置：

- `import fbgemm_ascend` 会在当前进程自动刷新 `ASCEND_CUSTOM_OPP_PATH`，使包内 AscendC 自定义算子对 CANN runtime 可见。

如需在 shell 级别预先设置环境（供其他进程复用），可以手动 source 环境脚本：

```bash
source $(python3 -c "import fbgemm_ascend; print(fbgemm_ascend.env_setup_path())")
```

## 使用示例

```python
import fbgemm_ascend

# torch.ops.fbgemm.asynchronous_complete_cumsum 等算子即可在 NPU 上调用
```

## 免责声明

本仓库包含多个开发分支，用于快速验证 Ascend 新特性或对齐上游 FBGEMM 变更。这些分支可能尚未经过完整的回归和性能验证，不建议直接用于生产或关键业务环境。请优先使用已发布的稳定版本或经过严格测试的分支。因使用开发版本造成的任何损失、数据问题或性能回退，项目及贡献者不承担责任。
