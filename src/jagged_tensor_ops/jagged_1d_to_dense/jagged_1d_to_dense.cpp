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

at::Tensor jagged_1d_to_dense_npu(at::Tensor values, at::Tensor offsets, c10::SymInt max_lengths,
                                  const int64_t padding_value)
{
    int64_t max_L = max_lengths.as_int_unchecked();
    TORCH_CHECK(max_L >= 0, "max_sequence_length must be non-negative, but got ", max_L);
    return jagged_to_padded_dense_impl_v1(values, offsets, max_L, padding_value);
}

at::Tensor jagged_1d_to_dense_meta(at::Tensor values, at::Tensor offsets, c10::SymInt max_lengths,
                                   const int64_t padding_value)
{
    TORCH_CHECK(offsets.dim() == 1, "offsets must be 1D, but got ", offsets.dim(), "D");
    auto B = offsets.sym_size(0) - 1;
    return at::empty_symint({B, max_lengths}, values.options().device(c10::kMeta));
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("jagged_1d_to_dense", TORCH_FN(jagged_1d_to_dense_npu));
}

TORCH_LIBRARY_IMPL(fbgemm, Meta, m)
{
    m.impl("jagged_1d_to_dense", TORCH_FN(jagged_1d_to_dense_meta));
}
