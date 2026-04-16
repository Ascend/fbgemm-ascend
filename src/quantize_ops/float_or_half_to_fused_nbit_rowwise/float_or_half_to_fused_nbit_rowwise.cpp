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

#include "../../common/common_utils.h"
#include "../../common/pytorch_npu_helper.hpp"

using namespace at;

at::Tensor float_or_half_to_fused_nbit_rowwise_npu(const at::Tensor& input, int64_t bit_rate)
{
    TORCH_CHECK(input.dim() == 2, "input must be 2D, got ", input.dim(), "D");
    TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf,
                "input must be float32 or float16, got ", input.scalar_type());
    TORCH_CHECK(bit_rate == 1 || bit_rate == 2 || bit_rate == 4 || bit_rate == 8,
                "bit_rate must be 1, 2, 4, or 8, got ", bit_rate);

    std::vector<at::Tensor> tensors = {input};
    std::vector<std::string> names = {"input"};
    check_tensor_npu_device(tensors, names);

    auto inputConti = input.contiguous();
    int64_t nrows = inputConti.size(0);
    int64_t ncols = inputConti.size(1);

    constexpr int32_t kBitsPerByte = 8;
    int32_t numElemPerByte = kBitsPerByte / static_cast<int32_t>(bit_rate);
    TORCH_CHECK(ncols % (2 * numElemPerByte) == 0, "ncols must be a multiple of ", 2 * numElemPerByte,
                " for bit_rate=", bit_rate, ", got ncols=", ncols);

    int64_t outputColumns = (ncols + numElemPerByte - 1) / numElemPerByte + 2 * static_cast<int64_t>(sizeof(at::Half));

    auto output = at::empty({nrows, outputColumns}, inputConti.options().dtype(at::kByte));

    if (nrows == 0 || ncols == 0) {
        return output;
    }

    EXEC_NPU_CMD(aclnnFloatOrHalfToFusedNbitRowwise, inputConti, bit_rate, output);

    return output;
}

at::Tensor float_to_fused_nbit_rowwise_npu(const at::Tensor& input, int64_t bit_rate)
{
    return float_or_half_to_fused_nbit_rowwise_npu(input, bit_rate);
}

at::Tensor half_to_fused_nbit_rowwise_npu(const at::Tensor& input, int64_t bit_rate)
{
    return float_or_half_to_fused_nbit_rowwise_npu(input, bit_rate);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("FloatToFusedNBitRowwiseQuantizedSBHalf", &float_to_fused_nbit_rowwise_npu);
    m.impl("HalfToFusedNBitRowwiseQuantizedSBHalf", &half_to_fused_nbit_rowwise_npu);
    m.impl("FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf", &float_or_half_to_fused_nbit_rowwise_npu);
}
