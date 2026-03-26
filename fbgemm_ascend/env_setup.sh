#!/bin/bash
# fbgemm_ascend 环境设置脚本
# 使用前 source 此脚本以启用包内 AscendC 自定义算子：
#   source $(python3 -c "import fbgemm_ascend; print(fbgemm_ascend.env_setup_path())")
#
# 脚本自动检测所在目录，遍历 opp/<variant>/vendors/ 下所有算子 vendor，
# 设置 ASCEND_CUSTOM_OPP_PATH 和 LD_LIBRARY_PATH，可扩展到多算子。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPP_ROOT="${SCRIPT_DIR}/opp"

detect_soc() {
    if [ -n "${SOC_VERSION}" ]; then
        echo "${SOC_VERSION}"
        return
    fi
    if ! command -v npu-smi >/dev/null 2>&1; then
        return
    fi
    local line
    while IFS= read -r line; do
        case "${line}" in
            *Ascend*)
                case "${line}" in
                    *Mcu*) ;;
                    *)
                        line=${line#*Ascend}
                        line=${line%% *}
                        echo "Ascend${line}"
                        return
                        ;;
                esac
                ;;
        esac
    done < <(npu-smi info -m 2>/dev/null)
}

select_variant() {
    case "${FBGEMM_ASCEND_FORCE_BUILD_VER:-}" in
        A5|a5) echo "A5" ; return ;;
        A2|a2) echo "A2" ; return ;;
        A3|a3) echo "A3" ; return ;;
    esac
    local soc
    soc=$(detect_soc)
    case "${soc}" in
        Ascend95*) echo "A5" ; return ;;
        Ascend910B*) echo "A2" ; return ;;
        Ascend910_93*) echo "A3" ; return ;;
        *)
            echo "[fbgemm_ascend] Warning: unknown SOC '${soc:-unknown}', defaulting to A5" >&2
            echo "A5"
            return
            ;;
    esac
}

VARIANT=$(select_variant)
VENDORS_DIR="${OPP_ROOT}/${VARIANT}/vendors"
if [ ! -d "${VENDORS_DIR}" ]; then
    echo "[fbgemm_ascend] Warning: variant ${VARIANT} not packaged under ${VENDORS_DIR}" >&2
    return 1 2>/dev/null || exit 1
fi
VENDORS_ROOT="$(dirname "${VENDORS_DIR}")"

# 先清除上次 source 留下的本包路径（幂等：多次 source 不重复）
_old_custom=""
IFS=':' read -ra _parts <<< "${ASCEND_CUSTOM_OPP_PATH:-}"
for _p in "${_parts[@]}"; do
    case "$_p" in "${VENDORS_ROOT}"*) ;; *) _old_custom="${_old_custom:+${_old_custom}:}${_p}" ;; esac
done

_old_ld=""
IFS=':' read -ra _parts <<< "${LD_LIBRARY_PATH:-}"
for _p in "${_parts[@]}"; do
    case "$_p" in "${VENDORS_ROOT}"*) ;; *) _old_ld="${_old_ld:+${_old_ld}:}${_p}" ;; esac
done

_FBGEMM_CUSTOM_OPP=""
_FBGEMM_LD_LIB=""

for vendor_dir in "${VENDORS_DIR}"/*/; do
    [ -d "$vendor_dir" ] || continue
    _FBGEMM_CUSTOM_OPP="${_FBGEMM_CUSTOM_OPP}${vendor_dir%/}:"
    if [ -d "${vendor_dir}op_api/lib" ]; then
        _FBGEMM_LD_LIB="${_FBGEMM_LD_LIB}${vendor_dir}op_api/lib:"
    fi
done

# 追加 OPP 根目录（供 GetCustLibPath() 通过 vendors/config.ini 查找算子）
_FBGEMM_CUSTOM_OPP="${_FBGEMM_CUSTOM_OPP}${VENDORS_ROOT}"

export ASCEND_CUSTOM_OPP_PATH="${_FBGEMM_CUSTOM_OPP}${_old_custom:+:${_old_custom}}"
export LD_LIBRARY_PATH="${_FBGEMM_LD_LIB}${_old_ld}"

unset _FBGEMM_CUSTOM_OPP _FBGEMM_LD_LIB _old_custom _old_ld _parts _p

echo "[fbgemm_ascend] ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}"
