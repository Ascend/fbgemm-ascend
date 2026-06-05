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

#include <tuple>

#include "../../common/pytorch_npu_helper.hpp"
#include "../../common/common_utils.h"
#include "fbgemm_ascend/jagged_tensor_ops.h"
#include "fbgemm_ascend/sparse_ops.h"

using namespace at;

namespace fbgemm_npu {

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> stacked_jagged_2d_to_dense_forward_npu(
    at::Tensor values, at::Tensor lengths, const std::vector<int64_t>& offset_per_key_intput,
    const std::vector<int64_t>& max_lengths_per_key_intput, int64_t padding_value)
{
    check_tensor_dim(values, EXPECTED_DIM_2D, "values");
    check_tensor_dim(lengths, EXPECTED_DIM_2D, "lengths");
    const auto lengths_contig = lengths.contiguous();
    const int64_t t_keys = lengths_contig.size(0);
    c10::IntArrayRef offset_per_key(offset_per_key_intput.data(), offset_per_key_intput.size());
    c10::IntArrayRef max_lengths_per_key(max_lengths_per_key_intput.data(), max_lengths_per_key_intput.size());
    TORCH_CHECK(static_cast<int64_t>(offset_per_key.size()) == t_keys + 1, "offset_per_key must have length T+1 (",
                t_keys + 1, "), but got ", offset_per_key.size());
    TORCH_CHECK(static_cast<int64_t>(max_lengths_per_key.size()) == t_keys, "max_lengths_per_key must have length T (",
                t_keys, "), but got ", max_lengths_per_key.size());

    std::vector<at::Tensor> tensors = {values, lengths_contig};
    std::vector<std::string> names = {"values", "lengths"};
    check_tensor_npu_device(tensors, names);

    const at::OptionalDeviceGuard guard(device_of(values));
    std::vector<at::Tensor> padded_values_per_key;
    std::vector<at::Tensor> offsets_tensor_per_key;
    padded_values_per_key.reserve(static_cast<size_t>(t_keys));
    offsets_tensor_per_key.reserve(static_cast<size_t>(t_keys));

    for (const auto t : c10::irange(t_keys)) {
        const int64_t seg_begin = offset_per_key[static_cast<size_t>(t)];
        const int64_t seg_end = offset_per_key[static_cast<size_t>(t + 1)];
        TORCH_CHECK(seg_end >= seg_begin, "offset_per_key[", t + 1, "] (", seg_end, ") must be >= offset_per_key[", t,
                    "] (", seg_begin, ")");
        const int64_t seg_len = seg_end - seg_begin;
        TORCH_CHECK(seg_len == lengths_contig.select(0, t).sum().item<int64_t>(), "values slice length (", seg_len,
                    ") must equal sum(lengths[", t, "])");

        const at::Tensor values_slice = values.narrow(0, seg_begin, seg_len);
        const at::Tensor key_lengths = lengths_contig.select(0, t).contiguous();
        const at::Tensor offsets = asynchronous_complete_cumsum_npu(key_lengths);
        offsets_tensor_per_key.push_back(offsets);
        const int64_t max_l = max_lengths_per_key[static_cast<size_t>(t)];
        padded_values_per_key.push_back(
            jagged_to_padded_dense_npu(values_slice, {offsets}, max_l, static_cast<double>(padding_value)));
    }

    return std::make_tuple(padded_values_per_key, offsets_tensor_per_key);
}

std::vector<at::Tensor> stacked_jagged_2d_to_dense_npu(at::Tensor values, at::Tensor lengths,
                                                       const std::vector<int64_t>& offset_per_key_intput,
                                                       const std::vector<int64_t>& max_lengths_per_key_intput,
                                                       int64_t padding_value)
{
    return std::get<0>(stacked_jagged_2d_to_dense_forward_npu(values, lengths, offset_per_key_intput,
                                                              max_lengths_per_key_intput, padding_value));
}

at::Tensor stacked_jagged_2d_to_dense_backward_npu(int64_t B, int64_t D, int64_t total_L,
                                                   const std::vector<at::Tensor>& grad_padded_values_per_key,
                                                   const std::vector<at::Tensor>& offsets_tensor_per_key,
                                                   const std::vector<int64_t>& offset_per_key_intput)
{
    c10::IntArrayRef offset_per_key(offset_per_key_intput.data(), offset_per_key_intput.size());
    const size_t t_keys = grad_padded_values_per_key.size();
    TORCH_CHECK(offsets_tensor_per_key.size() == t_keys, "offsets_tensor_per_key must have length T (", t_keys,
                "), but got ", offsets_tensor_per_key.size());
    TORCH_CHECK(offset_per_key.size() == t_keys + 1, "offset_per_key must have length T+1 (", t_keys + 1, "), but got ",
                offset_per_key.size());

    std::vector<at::Tensor> tensors;
    std::vector<std::string> names;
    tensors.reserve(t_keys * 2);
    names.reserve(t_keys * 2);
    for (const auto t : c10::irange(t_keys)) {
        tensors.push_back(grad_padded_values_per_key[t]);
        names.push_back("grad_padded_values_per_key[" + std::to_string(t) + "]");
        tensors.push_back(offsets_tensor_per_key[t]);
        names.push_back("offsets_tensor_per_key[" + std::to_string(t) + "]");
    }
    check_tensor_npu_device(tensors, names);

    TORCH_CHECK(total_L == offset_per_key[t_keys], "total_L (", total_L, ") must equal offset_per_key[T] (",
                offset_per_key[t_keys], ")");
    if (t_keys == 0) {
        return at::empty({total_L, D}, at::TensorOptions().device(c10::DeviceType::PrivateUse1));
    }
    const at::OptionalDeviceGuard guard(device_of(grad_padded_values_per_key[0]));
    std::vector<at::Tensor> grad_values_per_key;
    grad_values_per_key.reserve(t_keys);

    for (const auto t : c10::irange(t_keys)) {
        const int64_t seg_begin = offset_per_key[t];
        const int64_t seg_end = offset_per_key[t + 1];
        TORCH_CHECK(seg_end >= seg_begin, "offset_per_key[", t + 1, "] (", seg_end, ") must be >= offset_per_key[", t,
                    "] (", seg_begin, ")");
        const int64_t seg_len = seg_end - seg_begin;
        TORCH_CHECK(offsets_tensor_per_key[t].numel() == B + 1, "offsets_tensor_per_key[", t,
                    "] must have length B+1 (", B + 1, "), but got ", offsets_tensor_per_key[t].numel());
        TORCH_CHECK(grad_padded_values_per_key[t].size(0) == B, "grad_padded_values_per_key[", t,
                    "].size(0) must equal B (", B, ")");
        TORCH_CHECK(grad_padded_values_per_key[t].size(2) == D, "grad_padded_values_per_key[", t,
                    "].size(2) must equal D (", D, ")");
        grad_values_per_key.emplace_back(
            dense_to_jagged_impl(grad_padded_values_per_key[t], offsets_tensor_per_key[t], seg_len));
    }

    if (grad_values_per_key.empty()) {
        auto options = grad_padded_values_per_key.empty() ? at::TensorOptions().device(c10::DeviceType::PrivateUse1)
                                                          : grad_padded_values_per_key[0].options();
        return at::empty({total_L, D}, options);
    }
    return at::cat(grad_values_per_key, 0);
}

}  // namespace fbgemm_npu

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("stacked_jagged_2d_to_dense",
           torch::dispatch(c10::DispatchKey::PrivateUse1, TORCH_FN(fbgemm_npu::stacked_jagged_2d_to_dense_npu)));
    m.impl(
        "stacked_jagged_2d_to_dense_forward",
        torch::dispatch(c10::DispatchKey::PrivateUse1, TORCH_FN(fbgemm_npu::stacked_jagged_2d_to_dense_forward_npu)));
    m.impl(
        "stacked_jagged_2d_to_dense_backward",
        torch::dispatch(c10::DispatchKey::PrivateUse1, TORCH_FN(fbgemm_npu::stacked_jagged_2d_to_dense_backward_npu)));
}
