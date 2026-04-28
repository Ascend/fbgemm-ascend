/**
 * @file group_index_select_dim0.cpp
 *
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../../common/pytorch_npu_helper.hpp"
#include "../../common/common_utils.h"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

constexpr uint32_t MAX_GROUP_NUM = 32;

// 为NPU设备注册前向实现
tensor_list group_index_select_dim0_forward_impl_npu(at::TensorList inputGroups, at::TensorList indicesGroups)
{
    TORCH_CHECK(inputGroups.size() >= 1, "inputGroups must contain group, but got ", inputGroups.size(), " group");
    TORCH_CHECK(indicesGroups.size() >= 1, "indicesGroups must contain group, but got ", indicesGroups.size(), " group");
    TORCH_CHECK(inputGroups.size() == indicesGroups.size(), "inputGroups and indicesGroups must have the same group num, got ", inputGroups.size(), " and ", indicesGroups.size());
    TORCH_CHECK(inputGroups.size() <= MAX_GROUP_NUM, "inputGroups must less than 32, but got ", inputGroups.size(), " group");

    for (int64_t i = 0; i < inputGroups.size(); ++i) {
        auto& input = inputGroups[i];
        auto& indices = indicesGroups[i];

        check_tensor_non_empty(input, "input");
        check_tensor_non_empty(indices, "indices");
        
        TORCH_CHECK(input.dim() >= 2 && input.dim() <= 3, "inputGroups[", i, "] must be at least 2D and at most 3D, but got ", input.dim(), "D");
        TORCH_CHECK(indices.dim() == 1, "indicesGroups[", i, "] must be 1D, but got ", indices.dim(), "D");

        int64_t num_rows = input.size(0);
        int64_t max_idx = indices.max().item<int64_t>();
        int64_t min_idx = indices.min().item<int64_t>();
        TORCH_CHECK(max_idx < num_rows, "indicesGroups[", i, "] has index ", max_idx, " >= num_rows ", num_rows);
        TORCH_CHECK(min_idx >= 0, "indicesGroups[", i, "] has index ", min_idx, " < 0 ");
    }

    tensor_list outputGroups;
    outputGroups.reserve(inputGroups.size());
    
    for (int64_t i = 0; i < inputGroups.size(); ++i) {
        auto& input = inputGroups[i];
        auto& indices = indicesGroups[i];
        
        std::vector<int64_t> outputShape;
        outputShape.push_back(indices.size(0));
        for (int64_t j = 1; j < input.dim(); ++j) {
            outputShape.push_back(input.size(j));
        }
        
        outputGroups.push_back(at::empty(outputShape, input.options()));
    }

    at::TensorList inputGroupsRef(inputGroups);
    at::TensorList indicesGroupsRef(indicesGroups);
    at::TensorList outputGroupsRef(outputGroups);
    int64_t groupNum = inputGroups.size();

    EXEC_NPU_CMD(aclnnGroupIndexSelectDim0, inputGroupsRef, indicesGroupsRef, groupNum, outputGroupsRef);
    return outputGroups;
}

tensor_list group_index_select_dim0_backward_impl_npu(at::TensorList inputGroups, at::TensorList indicesGroups, at::TensorList gradOutputs)
{
    TORCH_CHECK(gradOutputs.size() >= 1, "inputGroups must contain group, but got ", gradOutputs.size(), " group");

    for (int64_t i = 0; i < gradOutputs.size(); ++i) {
        auto& gradOutput = gradOutputs[i];

        check_tensor_non_empty(gradOutput, "gradOutput");
        
        TORCH_CHECK(gradOutput.dim() >= 2 && gradOutput.dim() <= 3, "gradOutputs[", i, "] must be at least 2D and less than 3D, but got ", gradOutput.dim(), "D");
    }

    tensor_list inputReturnGroups;
    inputReturnGroups.reserve(inputGroups.size());

    for (int64_t i = 0; i < inputGroups.size(); ++i) {
        auto& input = inputGroups[i];

        std::vector<int64_t> inputReturnShape;
        for (int64_t j = 0; j < input.dim(); ++j) {
            inputReturnShape.push_back(input.size(j));
        }
        inputReturnGroups.push_back(at::zeros(inputReturnShape, input.options()));
    }

    at::TensorList gradOutputs_ref(gradOutputs);
    at::TensorList indicesGroups_ref(indicesGroups);
    at::TensorList inputReturnGroupsRef(inputReturnGroups);
    int64_t groupNum = inputGroups.size();

    EXEC_NPU_CMD(aclnnGroupIndexSelectDim0Backward, gradOutputs_ref, indicesGroups_ref, groupNum, inputReturnGroupsRef);

    tensor_list grad_indices(indicesGroups.size());
    tensor_list result = inputReturnGroups;
    result.insert(result.end(), grad_indices.begin(), grad_indices.end());

    return result;
}

class GroupIndexSelectDim0 : public torch::autograd::Function<GroupIndexSelectDim0> {
public:
    static tensor_list forward(AutogradContext* ctx, at::TensorList inputGroups, at::TensorList indicesGroups)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        tensor_list all_saved;
        
        for (auto& t : inputGroups) {
            all_saved.push_back(t);
        }
        for (auto& t : indicesGroups) {
            all_saved.push_back(t);
        }
        ctx->save_for_backward(all_saved);

        return group_index_select_dim0_forward_impl_npu(inputGroups, indicesGroups);
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list gradOutputs)
    {
        auto saved = ctx->get_saved_variables();
        int64_t num_groups = gradOutputs.size();

        tensor_list inputGroups;
        tensor_list indicesGroups;
        for (int64_t i = 0; i < num_groups; i++) {
            inputGroups.push_back(saved[i]);
        }
        for (int64_t i = 0; i < num_groups; i++) {
            indicesGroups.push_back(saved[num_groups + i]);
        }
        return group_index_select_dim0_backward_impl_npu(inputGroups, indicesGroups, gradOutputs);
    }
};

tensor_list group_index_select_dim0_autograd(at::TensorList inputGroups, at::TensorList indicesGroups)
{
    return GroupIndexSelectDim0::apply(inputGroups, indicesGroups);
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("group_index_select_dim0(Tensor[] inputGroups, Tensor[] indicesGroups) -> Tensor[]");
    m.def("group_index_select_dim0_backward(Tensor[] inputGroups, Tensor[] indicesGroups, Tensor[] gradOutputs) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("group_index_select_dim0", &group_index_select_dim0_forward_impl_npu);
    m.impl("group_index_select_dim0_backward", &group_index_select_dim0_backward_impl_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("group_index_select_dim0", &group_index_select_dim0_forward_impl_npu);
    m.impl("group_index_select_dim0_backward", &group_index_select_dim0_backward_impl_npu);
}

TORCH_LIBRARY_IMPL(mxrec, AutogradPrivateUse1, m)
{
    m.impl("group_index_select_dim0", &group_index_select_dim0_autograd);
}

TORCH_LIBRARY_IMPL(fbgemm, AutogradPrivateUse1, m)
{
    m.impl("group_index_select_dim0", &group_index_select_dim0_autograd);
}
