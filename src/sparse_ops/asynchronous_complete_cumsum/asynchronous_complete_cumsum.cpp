/**
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../../common/pytorch_npu_helper.hpp"
#include "../../common/common_utils.h"

at::Tensor asynchronous_complete_cumsum_npu(const at::Tensor &offset)
{
    const at::OptionalDeviceGuard guard(device_of(offset));

    std::vector<at::Tensor> tensors = {offset};
    std::vector<std::string> names = {"offset"};
    check_tensor_npu_device(tensors, names);

    auto offset_contig = offset.contiguous();
    const auto num_dims = offset.dim();
    TORCH_CHECK(num_dims == 1 || num_dims == 2,
        "asynchronous_complete_cumsum only supports 1D or 2D input, but got ", num_dims, "D");

    if (num_dims == 1) {
        int64_t offset_size = offset.size(0);
        TORCH_CHECK(offset_size >= 0 && offset_size < std::numeric_limits<int64_t>::max(),
            "offset.size(0) limit [0, ", std::numeric_limits<int64_t>::max(), "), but get ", offset_size, "\n");
        auto output = at::empty({offset_size + 1}, offset.options());
        if (offset_size == 0) {
            output.zero_();
            return output;
        }
        EXEC_NPU_CMD(aclnnAsynchronousCompleteCumsum, offset_contig, output);
        return output;
    } else {
        const auto num_vecs = offset.size(0);
        const auto num_entries = offset.size(1);
        auto output = at::zeros({num_vecs, num_entries + 1}, offset.options());

        if (offset.numel() == 0) {
            return output;
        }

        for (int64_t i = 0; i < num_vecs; i++) {
            auto row_in = offset_contig.select(0, i);
            auto row_out = output.select(0, i);
            EXEC_NPU_CMD(aclnnAsynchronousCompleteCumsum, row_in, row_out);
        }
        return output;
    }
}

at::Tensor asynchronous_inclusive_cumsum_npu(const at::Tensor &offset)
{
    if (offset.numel() == 0) {
        return at::empty_like(offset);
    }
    auto original_sizes = offset.sizes();
    auto flat_input = offset.contiguous().reshape({-1});
    auto complete_result = asynchronous_complete_cumsum_npu(flat_input);
    return complete_result.narrow(0, 1, flat_input.numel()).reshape(original_sizes);
}

at::Tensor asynchronous_exclusive_cumsum_npu(const at::Tensor &offset)
{
    if (offset.numel() == 0) {
        return at::empty_like(offset);
    }
    auto original_sizes = offset.sizes();
    auto flat_input = offset.contiguous().reshape({-1});
    auto complete_result = asynchronous_complete_cumsum_npu(flat_input);
    return complete_result.narrow(0, 0, flat_input.numel()).reshape(original_sizes);
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("asynchronous_inclusive_cumsum(Tensor offset) -> Tensor");
    m.def("asynchronous_exclusive_cumsum(Tensor offset) -> Tensor");
    m.def("asynchronous_complete_cumsum(Tensor offset) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("asynchronous_inclusive_cumsum", &asynchronous_inclusive_cumsum_npu);
    m.impl("asynchronous_exclusive_cumsum", &asynchronous_exclusive_cumsum_npu);
    m.impl("asynchronous_complete_cumsum", &asynchronous_complete_cumsum_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("asynchronous_inclusive_cumsum", &asynchronous_inclusive_cumsum_npu);
    m.impl("asynchronous_exclusive_cumsum", &asynchronous_exclusive_cumsum_npu);
    m.impl("asynchronous_complete_cumsum", &asynchronous_complete_cumsum_npu);
}
