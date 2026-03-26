#pragma once

#include <ATen/Tensor.h>

at::Tensor permute_pooled_embs_impl_npu(
    const at::Tensor& pooled_embs,
    const at::Tensor& offset_dim_list,
    const at::Tensor& permute_list,
    const at::Tensor& inv_offset_dim_list,
    const at::Tensor& inv_permute_list);
