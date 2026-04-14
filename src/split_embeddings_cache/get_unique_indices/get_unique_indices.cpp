/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <limits>
#include <tuple>
#include <vector>

#include <ATen/DeviceGuard.h>
#include <torch/library.h>

#include "../../common/common_utils.h"
#include "../../common/pytorch_npu_helper.hpp"

using namespace at;

namespace
{
    constexpr int64_t EXPECTED_DIM_1D = 1;
    constexpr int64_t SORT_DIM = 0;
    constexpr int64_t MAX_INPUT_NUMEL = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    constexpr bool SORT_STABLE = true;
    constexpr bool SORT_DESCENDING = false;

    struct UniqueIndicesOutputs
    {
        Tensor unique_indices;
        Tensor unique_indices_length;
        c10::optional<Tensor> unique_indices_count;
        c10::optional<Tensor> inverse_indices;
    };

    void validate_get_unique_indices_inputs(const Tensor &linear_indices, int64_t max_indices)
    {
        TORCH_CHECK(linear_indices.defined(), "linear_indices tensor must be defined");

        std::vector<Tensor> tensors = {linear_indices};
        std::vector<std::string> names = {"linear_indices"};
        check_tensor_npu_device(tensors, names);

        check_tensor_dim(linear_indices, EXPECTED_DIM_1D, "linear_indices");
        TORCH_CHECK(linear_indices.scalar_type() == at::kInt || linear_indices.scalar_type() == at::kLong,
                    "linear_indices must have int32 or int64 dtype, but got ", linear_indices.scalar_type());
        TORCH_CHECK(linear_indices.numel() <= MAX_INPUT_NUMEL, "linear_indices numel must be <= ", MAX_INPUT_NUMEL,
                    ", but got ", linear_indices.numel());
        TORCH_CHECK(max_indices >= 0, "max_indices must be >= 0, but got ", max_indices);
    }

    UniqueIndicesOutputs run_get_unique_indices_impl(const Tensor &linear_indices, bool compute_count,
                                                     bool compute_inverse_indices)
    {
        auto linear_indices_contiguous = linear_indices.contiguous();

        // RLE requires sorted input values and stable permutation for inverse output.
        auto sorted_values = at::empty_like(linear_indices_contiguous);
        auto sorted_indices =
            at::empty(linear_indices_contiguous.sizes(), linear_indices_contiguous.options().dtype(at::kLong));
        EXEC_NPU_CMD(aclnnSort, linear_indices_contiguous, SORT_STABLE, SORT_DIM, SORT_DESCENDING, sorted_values,
                     sorted_indices);

        auto unique_values = at::empty_like(sorted_values);
        auto unique_counts = at::empty(sorted_values.sizes(), sorted_values.options().dtype(at::kInt));
        auto unique_length_tensor = at::empty({1}, sorted_values.options().dtype(at::kInt));

        EXEC_NPU_CMD(aclnnRunLengthEncode, sorted_values, compute_count, unique_values, unique_counts, unique_length_tensor);

        c10_npu::getCurrentNPUStream().synchronize();

        c10::optional<Tensor> count_output = compute_count ? c10::make_optional(unique_counts) : c10::nullopt;

        c10::optional<Tensor> inverse_output =
            compute_inverse_indices ? c10::make_optional(sorted_indices.to(at::kInt)) : c10::nullopt;

        return {unique_values, unique_length_tensor, count_output, inverse_output};
    }
} // namespace

std::tuple<Tensor, Tensor, c10::optional<Tensor>> get_unique_indices_impl_npu(const Tensor &linear_indices,
                                                                              int64_t max_indices, bool compute_count)
{
    const OptionalDeviceGuard guard(device_of(linear_indices));
    validate_get_unique_indices_inputs(linear_indices, max_indices);
    auto outputs = run_get_unique_indices_impl(linear_indices, compute_count, false);
    return std::make_tuple(outputs.unique_indices, outputs.unique_indices_length, outputs.unique_indices_count);
}

std::tuple<Tensor, Tensor, c10::optional<Tensor>, c10::optional<Tensor>> get_unique_indices_with_inverse_impl_npu(
    const Tensor &linear_indices, int64_t max_indices, bool compute_count, bool compute_inverse_indices)
{
    const OptionalDeviceGuard guard(device_of(linear_indices));
    validate_get_unique_indices_inputs(linear_indices, max_indices);
    auto outputs = run_get_unique_indices_impl(linear_indices, compute_count, compute_inverse_indices);
    return std::make_tuple(outputs.unique_indices, outputs.unique_indices_length, outputs.unique_indices_count,
                           outputs.inverse_indices);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("get_unique_indices", &get_unique_indices_impl_npu);
    m.impl("get_unique_indices_with_inverse", &get_unique_indices_with_inverse_impl_npu);
}
