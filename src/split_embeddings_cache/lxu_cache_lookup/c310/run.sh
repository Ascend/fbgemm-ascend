#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

set -e

readonly THIS_SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
readonly WORK_DIR="$(dirname "${THIS_SCRIPT}")"
readonly UTILS_SCRIPT="${WORK_DIR}/../../../scripts/op_builder_utils.sh"

if [ ! -f "$UTILS_SCRIPT" ]; then
    echo "ERROR: Cannot find op_builder_utils.sh at ${UTILS_SCRIPT}" >&2
    exit 1
fi

source "$UTILS_SCRIPT"

vendor_name="lxu_cache_lookup"
export AI_CORE_PROFILE="c310"
export COPY_KERNEL_COMMON_UTILS="1"

parse_arguments "$@" || exit 1
build_and_install_operator "$WORK_DIR" "$vendor_name" || exit 1
