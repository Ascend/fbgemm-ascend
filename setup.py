# fbgemm_ascend setup（NPU 算子包，参考 fbgemm_recsdk 重构）
# 使用方式：在 fbgemm_ascend 目录下执行
# pip install . --no-build-isolation （需先安装 scikit-build）
# 依赖：torch, torch_npu（NPU 环境）, scikit-build

import os
import setuptools
from skbuild import setup as skbuild_setup


def _get_torch_prefix():
    import torch
    return os.path.dirname(torch.__file__)


def _get_cxx11_abi():
    try:
        import torch
        return int(torch._C._GLIBCXX_USE_CXX11_ABI)
    except Exception:
        return 0


_DEFAULT_VARIANTS = ["A5", "A2", "A3"]


def _ascendc_build_variants():
    value = os.environ.get("FBGEMM_ASCEND_BUILD_VERS")
    if value:
        return value
    return ",".join(_DEFAULT_VARIANTS)


def cmake_args():
    torch_root = _get_torch_prefix()

    os.environ.setdefault(
        "CMAKE_BUILD_PARALLEL_LEVEL",
        str((os.cpu_count() or 4) // 2),
    )

    build_variants = _ascendc_build_variants()

    return [
        f"-DCMAKE_PREFIX_PATH={torch_root}",
        f"-D_GLIBCXX_USE_CXX11_ABI={_get_cxx11_abi()}",
        f"-DFBGEMM_ASCEND_BUILD_VERS={build_variants}",
    ]


skbuild_setup(
    name="fbgemm_ascend",
    version="1.2.0",
    description="FBGEMM Ascend NPU operators (torch.ops.fbgemm on NPU)",
    packages=setuptools.find_packages(),
    install_requires=["numpy"],
    python_requires=">=3.8",
    cmake_args=cmake_args(),
    cmake_install_dir="fbgemm_ascend",
)
