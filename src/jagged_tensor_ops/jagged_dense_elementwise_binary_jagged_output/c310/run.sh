#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e

readonly THIS_SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
readonly WORK_DIR="$(dirname "${THIS_SCRIPT}")"
readonly UTILS_SCRIPT="${WORK_DIR}/../../../scripts/op_builder_utils.sh"

if [ ! -f "$UTILS_SCRIPT" ]; then
    echo "ERROR: Cannot find op_builder_utils.sh at ${UTILS_SCRIPT}" >&2
    exit 1
fi

source "$UTILS_SCRIPT"

vendor_name="jagged_dense_elementwise_binary_jagged_output"
export AI_CORE_PROFILE="c310"
export OPERATOR_JSON_FILE="$(readlink -f "${WORK_DIR}/${vendor_name}.json")"
export OPERATOR_SOURCE_ROOT="$(readlink -f "${WORK_DIR}")"
export CMAKE_PRESET_VENDOR_NAME="jagged_dense_elementwise_binary_jagged_output"
export MSOPGEN_OP_NAME="JaggedDenseElementwiseBinaryJaggedOutput"
export COPY_KERNEL_COMMON_UTILS="1"

parse_arguments "$@" || exit 1

sed -i 's/SetAtomicAdd<uint16_t>();/SetAtomicAdd<int16_t>();/g' "${OPERATOR_SOURCE_ROOT}/op_kernel/${vendor_name}.cpp"
build_and_install_operator "$WORK_DIR" "$vendor_name" || exit 1
