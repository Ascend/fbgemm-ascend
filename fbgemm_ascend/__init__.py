# fbgemm_ascend: NPU 算子包
# import fbgemm_ascend 后即可使用 torch.ops.fbgemm.* 的 NPU 实现，无需手动 load_library
import logging
import os
import shutil
import subprocess

import torch
import fbgemm_gpu  # noqa: F401  — 注册 fbgemm::* schema 及 CPU 实现

open_source: bool = True
__version__: str = "1.2.0"


def _package_has_opp(payload_dir: str) -> bool:
    vendors_in_root = os.path.join(payload_dir, "opp", "vendors")
    if os.path.isdir(vendors_in_root):
        return True
    opp_root = os.path.join(payload_dir, "opp")
    if not os.path.isdir(opp_root):
        return False
    for name in os.listdir(opp_root):
        candidate = os.path.join(opp_root, name, "vendors")
        if os.path.isdir(candidate):
            return True
    return False


def env_setup_path() -> str:
    """返回 env_setup.sh 的绝对路径，供手动调试或自定义环境使用。

    默认情况下，import fbgemm_ascend 时会在当前进程内自动刷新
    ASCEND_CUSTOM_OPP_PATH，日常使用通常无需再 source 该脚本。

    如需在 shell 级别预先配置环境（例如给其他进程复用），可执行：
        source $(python3 -c "import fbgemm_ascend; print(fbgemm_ascend.env_setup_path())")
    """
    import site

    candidates = []
    # 优先检查 site-packages 中的安装版（避免源码目录覆盖）
    for sp in site.getsitepackages():
        candidates.append(os.path.join(sp, "fbgemm_ascend", "env_setup.sh"))
    # 再检查当前 __file__ 所在目录
    pkg_dir = os.path.realpath(os.path.dirname(__file__))
    candidates.append(os.path.join(pkg_dir, "env_setup.sh"))
    for c in candidates:
        if os.path.isfile(c) and _package_has_opp(os.path.dirname(c)):
            return c
    raise FileNotFoundError(
        f"fbgemm_ascend: env_setup.sh with packaged opp/*/vendors not found (tried: {candidates})"
    )


def _detect_soc_from_npu_smi():
    npu_smi = shutil.which("npu-smi")
    if not npu_smi:
        logging.warning("fbgemm_ascend: npu-smi not found.")
        return None
    try:
        out = subprocess.check_output(
            [npu_smi, "info", "-m"],
            stderr=subprocess.STDOUT,
        ).decode(errors="ignore")
    except subprocess.CalledProcessError:
        logging.warning("fbgemm_ascend: npu-smi failed.")
        return None

    for line in out.splitlines():
        if "Ascend" in line and "Mcu" not in line:
            suffix = line.split("Ascend", 1)[1].strip().split()[0]
            return f"Ascend{suffix}"
    return None


def _detect_soc_version():
    soc = os.environ.get("SOC_VERSION")
    if soc:
        return soc
    return _detect_soc_from_npu_smi()


def _normalize_variant(value: str | None) -> str | None:
    if not value:
        return None
    upper = value.strip().upper()
    if upper in {"A5", "A2", "A3"}:
        return upper
    return None


def _map_soc_to_variant(soc: str | None) -> str:
    if not soc:
        logging.warning("fbgemm_ascend: SOC not detected, defaulting to A5 variants")
        return "A5"
    if soc.startswith("Ascend95"):
        return "A5"
    if soc.startswith("Ascend910B"):
        return "A2"
    if soc.startswith("Ascend910_93"):
        return "A3"
    logging.warning("fbgemm_ascend: unknown SOC '%s', defaulting to A5", soc)
    return "A5"


def _select_opp_variant() -> str:
    override = _normalize_variant(os.environ.get("FBGEMM_ASCEND_FORCE_BUILD_VER"))
    if override:
        return override
    return _map_soc_to_variant(_detect_soc_version())


def _setup_custom_opp_path() -> None:
    """在当前进程内刷新 ASCEND_CUSTOM_OPP_PATH，仅依赖 opp/ 目录结构。

    设计原则：
    - 只改 ASCEND_CUSTOM_OPP_PATH，不动 LD_LIBRARY_PATH（实测不需要）
    - 幂等：多次 import / 多次调用不会无限追加同一前缀
    - 不覆盖用户自定义路径：保留原有非本包前缀
    """
    pkg_dir = os.path.realpath(os.path.dirname(__file__))
    opp_root = os.path.join(pkg_dir, "opp")
    variant = _select_opp_variant()
    variant_root = os.path.join(opp_root, variant)
    vendors_dir = os.path.join(variant_root, "vendors")
    if not os.path.isdir(vendors_dir):
        logging.warning(
            "fbgemm_ascend: variant '%s' not packaged (expected %s)",
            variant,
            vendors_dir,
        )
        return

    new_paths = []
    try:
        for name in os.listdir(vendors_dir):
            vendor_path = os.path.join(vendors_dir, name)
            if os.path.isdir(vendor_path):
                new_paths.append(vendor_path)
    except Exception as e:  # 仅记录日志，不阻塞导入
        logging.error("fbgemm_ascend: failed to scan vendors dir '%s': %s", vendors_dir, e)
        return

    # 追加 OPP 根目录（供 CANN runtime 按 vendors/config.ini 查找）
    new_paths.append(variant_root)

    existing = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    old_parts = [p for p in existing.split(os.pathsep) if p and not p.startswith(opp_root)]
    merged = new_paths + old_parts
    # 去重同时保持顺序
    seen = set()
    deduped: list[str] = []
    for p in merged:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(deduped)
    logging.info(
        "fbgemm_ascend: ASCEND_CUSTOM_OPP_PATH=%s (variant=%s)",
        os.environ["ASCEND_CUSTOM_OPP_PATH"],
        variant,
    )


# 扩展库名（CMake 产出 fbgemm_ascend_py_*.so，Windows 上可能为 .pyd）
_EXT_SUFFIX = ".pyd" if os.name == "nt" else ".so"
_HOST_LIB_CANDIDATES = {
    "A5": [
        "fbgemm_ascend_py_a5" + _EXT_SUFFIX,
        "libfbgemm_ascend_py_a5.so",
    ],
    "A2A3": [
        "fbgemm_ascend_py_a2a3" + _EXT_SUFFIX,
        "libfbgemm_ascend_py_a2a3.so",
    ],
}


def _candidate_lib_names(variant: str) -> list[str]:
    key = "A5" if variant == "A5" else "A2A3"
    names = list(_HOST_LIB_CANDIDATES.get(key, []))
    for other_key, values in _HOST_LIB_CANDIDATES.items():
        if other_key != key:
            names.extend(values)
    return names


def _load_library(no_throw: bool = False) -> None:
    pkg_dir = os.path.realpath(os.path.dirname(__file__))
    # 可编辑安装时 .so 可能在：包目录、_skbuild/.../cmake-install/fbgemm_ascend/ 或 .../cmake-install/fbgemm_ascend/fbgemm_ascend/
    search_dirs = [pkg_dir]
    pkg_root = os.path.dirname(pkg_dir)
    skbuild_dir = os.path.join(pkg_root, "_skbuild")
    if os.path.isdir(skbuild_dir):
        for name in os.listdir(skbuild_dir):
            install_base = os.path.join(skbuild_dir, name, "cmake-install", "fbgemm_ascend")
            if os.path.isdir(install_base):
                search_dirs.append(install_base)
            install_nested = os.path.join(install_base, "fbgemm_ascend")
            if os.path.isdir(install_nested):
                search_dirs.append(install_nested)
    # 在尝试加载 .so 之前，先在当前进程内设置 ASCEND_CUSTOM_OPP_PATH
    variant = _select_opp_variant()
    _setup_custom_opp_path()

    lib_names = _candidate_lib_names(variant)

    for search_dir in search_dirs:
        for filename in lib_names:
            lib_path = os.path.join(search_dir, filename)
            if os.path.isfile(lib_path):
                try:
                    torch.ops.load_library(lib_path)
                    logging.info("fbgemm_ascend: loaded '%s'", lib_path)
                    return
                except Exception as e:
                    logging.error("fbgemm_ascend: could not load '%s': %s", lib_path, e)
                    if not no_throw:
                        raise
    msg = (
        "fbgemm_ascend: library not found (variant=%s, dirs=%s, candidates=%s)"
        % (variant, search_dirs, lib_names)
    )
    if no_throw:
        logging.warning(msg)
        return
    raise FileNotFoundError(msg)


_load_library(no_throw=False)
