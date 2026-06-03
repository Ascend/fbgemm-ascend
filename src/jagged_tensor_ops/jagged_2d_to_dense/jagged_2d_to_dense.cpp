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
#include <torch/library.h>

#include "../../common/pytorch_npu_helper.hpp"
#include "../../common/common_utils.h"
#include "fbgemm_ascend/jagged_tensor_ops.h"

using namespace at;
using fbgemm_npu::EXPECTED_DIM_1D;
using fbgemm_npu::EXPECTED_DIM_2D;

at::Tensor jagged_2d_to_dense_npu(const at::Tensor& values, const at::Tensor& offsets, const int64_t max_lengths)
{
    return jagged_to_padded_dense_impl_v1(values, offsets, max_lengths, .0);
}

at::Tensor jagged_2d_to_dense_meta(const at::Tensor& values, const at::Tensor& offsets, const int64_t max_lengths)
{
    TORCH_CHECK(offsets.dim() == EXPECTED_DIM_1D, "offsets must be 1D, but got ", offsets.dim(), "D.");
    TORCH_CHECK(values.dim() == EXPECTED_DIM_2D, "values must be 2D, but got ", values.dim(), "D.");
    auto B = offsets.sym_size(0) - 1;
    auto D = values.size(-1);
    return at::empty_symint({B, max_lengths, D}, values.options().device(c10::kMeta));
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("jagged_2d_to_dense", TORCH_FN(jagged_2d_to_dense_npu));
}

TORCH_LIBRARY_IMPL(fbgemm, Meta, m)
{
    m.impl("jagged_2d_to_dense", TORCH_FN(jagged_2d_to_dense_meta));
}
