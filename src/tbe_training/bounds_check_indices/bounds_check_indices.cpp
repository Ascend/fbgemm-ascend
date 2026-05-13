/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../../common/pytorch_npu_helper.hpp"
#include "../../common/common_utils.h"

#include <cstdlib>


bool check_env_enable_v2() {
  const auto value = std::getenv("FBGEMM_BOUNDS_CHECK_INDICES_V2");
  if (!value) {
    return false;
  }

  try {
    return std::stoi(value) == 1;
  } catch (const std::invalid_argument&) {
    return false;
  }
}

void bounds_check_indices_impl_npu(
    at::Tensor& rows_per_table,
    at::Tensor& indices,
    at::Tensor& offsets,
    int64_t bounds_check_mode,
    at::Tensor& warning,
    const c10::optional<at::Tensor>& weights,
    const c10::optional<at::Tensor>& B_offsets,
    int64_t max_B,
    const c10::optional<at::Tensor>& b_t_map,
    int64_t info_B_num_bits,
    int64_t info_B_mask,
    int8_t bounds_check_version
    )
{
    TORCH_CHECK(bounds_check_version == 1 || bounds_check_version == 2,
        "bounds_check_indices: bounds_check_version=", bounds_check_version, " is not support");

    bool env_enable_v2 = check_env_enable_v2();
    if (env_enable_v2) {
        bounds_check_version = 2;
    }

    TORCH_CHECK(
        bounds_check_mode >= 0 && bounds_check_mode <= 2,
        "bounds_check_indices: bounds_check_mode=", bounds_check_mode, " is not support");

    std::vector<at::Tensor> tensors = {rows_per_table, indices, offsets, warning};
    std::vector<std::string> names = {"rows_per_table", "indices", "offsets", "warning"};
    check_tensor_npu_device(tensors, names);

    TORCH_CHECK(indices.dim() == 1, "indices should be 1D");
    TORCH_CHECK(offsets.dim() == 1, "offsets should be 1D");
    TORCH_CHECK(rows_per_table.dim() == 1, "rows_per_table should be 1D");
    TORCH_CHECK(warning.dim() == 1, "warning should be 1D");

    TORCH_CHECK(
        indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
        "indices must have int32 or int64 dtype.");
    TORCH_CHECK(
        offsets.scalar_type() == indices.scalar_type(),
        "offsets dtype must match indices dtype.");
    const int32_t T = rows_per_table.size(0);
    const int32_t total_B = offsets.size(0) - 1;
    if (total_B <= 0 || T == 0) {
        return;
    }
    const int32_t B = total_B / T;

    bool vbe = B_offsets.has_value();
    if (vbe) {
        TORCH_CHECK(B_offsets->dim() == 1, "B_offsets should be 1D");
        TORCH_CHECK(max_B >= 0, "max_B must be non-negative in VBE mode");
    } else {
        TORCH_CHECK(
            offsets.size(0) == B * T + 1,
            "offsets size ", offsets.size(0), " is not equal to B (", B, ") * T (", T, ") + 1");
    }

    if (weights.has_value() && weights->numel() != 0) {
        TORCH_CHECK(
            weights->size(0) == indices.size(0),
            "weights size ", weights->size(0), " is not equal to indices size ", indices.size(0) );
    }

    if (bounds_check_mode == 1) {
        warning.zero_();
    }

    bool is_indices_contiguous = indices.is_contiguous();
    bool is_offsets_contiguous = offsets.is_contiguous();
    bool is_warning_contiguous = warning.is_contiguous();

    at::Tensor indices_c = is_indices_contiguous ? indices : indices.contiguous();
    at::Tensor offsets_c = is_offsets_contiguous ? offsets : offsets.contiguous();
    at::Tensor warning_c = is_warning_contiguous ? warning : warning.contiguous();

    if (bounds_check_version == 1) {
        EXEC_NPU_CMD(aclnnBoundsCheckIndicesV1,
                     rows_per_table,
                     indices_c,
                     offsets_c,
                     warning_c,
                     B_offsets,
                     bounds_check_mode,
                     max_B,
                     T,
                     B,
                     total_B,
                     vbe,
                     indices_c,
                     offsets_c,
                     warning_c);
    } else {
        if (vbe) {
            TORCH_CHECK(b_t_map.has_value(), "b_t_map should have value");
            TORCH_CHECK(b_t_map.value().dim() == 1, "b_t_map should be 1D");
            TORCH_CHECK(b_t_map.value().size(0) == total_B, "b_t_map size must equal total_B");
        }
        EXEC_NPU_CMD(aclnnBoundsCheckIndicesV2,
                 rows_per_table,
                 indices_c,
                 offsets_c,
                 warning_c,
                 B_offsets, 
                 b_t_map,
                 bounds_check_mode,
                 info_B_num_bits,
                 info_B_mask,
                 T,
                 B,
                 total_B,
                 vbe,
                 indices_c,
                 offsets_c,
                 warning_c);
    }
    if (!is_indices_contiguous) {
        indices.copy_(indices_c);
    }
    if (!is_offsets_contiguous) {
        offsets.copy_(offsets_c);
    }
    if (!is_warning_contiguous) {
        warning.copy_(warning_c);
    }
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def(
        "bounds_check_indices("
        "    Tensor rows_per_table, "
        "    Tensor(a!) indices, "
        "    Tensor(b!) offsets, "
        "    int bounds_check_mode, "
        "    Tensor(c!) warning, "
        "    Tensor(d!)? weights=None, "
        "    Tensor? B_offsets=None, "
        "    SymInt max_B=-1, "
        "    Tensor? b_t_map=None, "
        "    int info_B_num_bits=-1, "
        "    int info_B_mask=-1, "
        "    int bounds_check_version=1 "
        ") -> ()");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("bounds_check_indices", &bounds_check_indices_impl_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("bounds_check_indices", &bounds_check_indices_impl_npu);
}
