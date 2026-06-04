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

namespace {
at::Tensor lengths_row_to_offsets_npu(const at::Tensor& lengths_contig, int64_t t)
{
    auto row = lengths_contig.select(0, t).to(at::kLong).contiguous();
    auto zero = at::zeros({1}, row.options());
    auto cum = at::cumsum(row, 0);
    return at::cat({zero, cum}, 0);
}
}  // namespace

std::vector<at::Tensor> stacked_jagged_1d_to_dense_npu(at::Tensor values, at::Tensor lengths,
                                                       const std::vector<int64_t>& offset_per_key_intput,
                                                       const std::vector<int64_t>& max_lengths_per_key_input,
                                                       int64_t padding_value)
{
    check_tensor_dim(values, EXPECTED_DIM_1D, "values");
    check_tensor_dim(lengths, EXPECTED_DIM_2D, "lengths");
    const auto lengths_contig = lengths.contiguous();
    const int64_t t_keys = lengths_contig.size(0);
    c10::IntArrayRef offset_per_key(offset_per_key_intput.data(), offset_per_key_intput.size());
    c10::IntArrayRef max_lengths_per_key(max_lengths_per_key_input.data(), max_lengths_per_key_input.size());
    TORCH_CHECK(static_cast<int64_t>(offset_per_key.size()) == t_keys + 1, "offset_per_key must have length T+1 (",
                t_keys + 1, "), but got ", offset_per_key.size());
    TORCH_CHECK(static_cast<int64_t>(max_lengths_per_key.size()) == t_keys, "max_lengths_per_key must have length T (",
                t_keys, "), but got ", max_lengths_per_key.size());

    std::vector<at::Tensor> tensors = {values, lengths_contig};
    std::vector<std::string> names = {"values", "lengths"};
    check_tensor_npu_device(tensors, names);

    const at::OptionalDeviceGuard guard(device_of(values));
    std::vector<at::Tensor> padded_values_per_key;
    padded_values_per_key.reserve(static_cast<size_t>(t_keys));

    for (const auto t : c10::irange(t_keys)) {
        const int64_t seg_begin = offset_per_key[static_cast<size_t>(t)];
        const int64_t seg_end = offset_per_key[static_cast<size_t>(t + 1)];
        TORCH_CHECK(seg_end >= seg_begin, "offset_per_key[", t + 1, "] (", seg_end, ") must be >= offset_per_key[", t,
                    "] (", seg_begin, ")");
        const int64_t seg_len = seg_end - seg_begin;
        TORCH_CHECK(seg_len == lengths_contig.select(0, t).sum().item<int64_t>(), "values slice length (", seg_len,
                    ") must equal sum(lengths[", t, "])");

        const at::Tensor values_slice = values.narrow(0, seg_begin, seg_len);
        const at::Tensor offsets = lengths_row_to_offsets_npu(lengths_contig, t);
        const int64_t max_l = max_lengths_per_key[static_cast<size_t>(t)];
        padded_values_per_key.push_back(
            jagged_to_padded_dense_npu(values_slice, {offsets}, max_l, static_cast<double>(padding_value)));
    }

    return padded_values_per_key;
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("stacked_jagged_1d_to_dense", TORCH_FN(stacked_jagged_1d_to_dense_npu));
}
