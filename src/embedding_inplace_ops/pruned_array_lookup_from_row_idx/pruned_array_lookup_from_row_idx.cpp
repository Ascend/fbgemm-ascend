/**
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <limits>

#include <ATen/DeviceGuard.h>
#include <torch/library.h>

#include "../../common/common_utils.h"
#include "../../common/pytorch_npu_helper.hpp"

using namespace at;

namespace {

Tensor pruned_array_lookup_from_row_idx_npu(const Tensor& updateRowIndices, const Tensor& updateTableIndices,
                                            const Tensor& indexRemappings,
                                            const Tensor& indexRemappingsOffsets)
{
    const OptionalDeviceGuard guard(device_of(updateRowIndices));

    TORCH_CHECK(
        updateRowIndices.scalar_type() == ScalarType::Int || updateRowIndices.scalar_type() == ScalarType::Long,
        "update_row_indices must be int32 or int64, got ", updateRowIndices.scalar_type());
    TORCH_CHECK(updateTableIndices.scalar_type() == ScalarType::Int, "update_table_indices must be int32, got ",
                updateTableIndices.scalar_type());
    TORCH_CHECK(indexRemappings.scalar_type() == ScalarType::Int || indexRemappings.scalar_type() == ScalarType::Long,
                "index_remappings must be int32 or int64, got ", indexRemappings.scalar_type());
    TORCH_CHECK(indexRemappingsOffsets.scalar_type() == ScalarType::Long,
                "index_remappings_offsets must be int64, got ", indexRemappingsOffsets.scalar_type());

    check_tensor_dim(updateRowIndices, 1, "update_row_indices");
    check_tensor_dim(updateTableIndices, 1, "update_table_indices");
    check_tensor_dim(indexRemappings, 1, "index_remappings");
    check_tensor_dim(indexRemappingsOffsets, 1, "index_remappings_offsets");

    const int64_t numIndices = updateRowIndices.numel();
    TORCH_CHECK(updateTableIndices.numel() == numIndices,
                "update_table_indices must have the same length as update_row_indices");
    TORCH_CHECK(
        static_cast<uint64_t>(indexRemappings.size(0)) <
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
        "index_remappings length must be less than int64 max");

    if (numIndices == 0) {
        return empty_like(updateRowIndices);
    }

    auto rowContig = updateRowIndices.contiguous();
    auto tableContig = updateTableIndices.contiguous();
    auto remapContig = indexRemappings.contiguous();
    auto offContig = indexRemappingsOffsets.contiguous();
    Tensor denseIndices = empty_like(rowContig);

    EXEC_NPU_CMD(aclnnPrunedArrayLookupFromRowIdx, rowContig, tableContig, remapContig, offContig, denseIndices);
    return denseIndices;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("pruned_array_lookup_from_row_idx("
          "    Tensor updateRowIndices, "
          "    Tensor updateTableIndices, "
          "    Tensor indexRemappings, "
          "    Tensor indexRemappingsOffsets) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("pruned_array_lookup_from_row_idx", &pruned_array_lookup_from_row_idx_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("pruned_array_lookup_from_row_idx", &pruned_array_lookup_from_row_idx_npu);
}
