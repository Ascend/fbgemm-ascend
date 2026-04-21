#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import sysconfig
import pytest
import torch
import torch_npu
import fbgemm_gpu
import fbgemm_ascend

torch.npu.config.allow_internal_format = False

DEVICE = "npu:0"


def get_golden(batch_size: int, csr_seg: torch.Tensor, values: torch.Tensor):
    """
    :param batch_size: 每行包含的元素个数
    :param csr_seg: 各分段长度的完整累积和，分段长度是指每个段所含的行数，csr_seg张量的大小是num_segments + 1，其中num_segments为段的数量
    :param values: 需要分段求和的张量，长度是batch_size的倍数
    :return: 分段求和的结果
    """
    y = torch.ops.fbgemm.segment_sum_csr(batch_size, csr_seg, values)
    return y


def get_op(batch_size: int, csr_seg: torch.Tensor, values: torch.Tensor, is_mxrec: bool):
    if is_mxrec:
        y = torch.ops.mxrec.segment_sum_csr(batch_size, csr_seg, values)
    else:
        y = torch.ops.fbgemm.segment_sum_csr(batch_size, csr_seg, values)
    return y


def generate_random_segment_sum_data(device, csr_type, v_type):
    """
    为 segment_sum_csr 算子生成随机测试数据

    参数:
        device: torch.device对象或字符串，如 "npu:0", "cpu", "cuda:0"
        csr_type: torch.dtype对象，csr_seg张量的数据类型（如 torch.int32）
        v_type: torch.dtype对象，values张量的数据类型（如 torch.float32）

    返回:
        batch_size: 批次数张量，shape=[1], dtype=int32
        csr_seg: CSR分段指针，shape=[num_segments+1], dtype=int32
        values: 数据张量，shape=[batch_size * total_elements_per_batch], dtype=dtype
    """

    batch_size_val = torch.randint(1, 33, (1,), device=device, dtype=csr_type).item()
    num_segments = torch.randint(2, 101, (1,), device=device, dtype=csr_type).item()
    segment_lengths = torch.randint(1, 101, (num_segments,), device=device, dtype=csr_type)

    csr_seg = torch.cat([
        torch.tensor([0], device=device, dtype=csr_type),
        segment_lengths.cumsum(dim=0)
    ], dim=0)
    csr_seg = csr_seg.to(torch.int32)

    total_elements_per_batch = csr_seg[-1].item()
    total_values_length = batch_size_val * total_elements_per_batch

    if v_type.is_floating_point:
        values = torch.empty(total_values_length, device=device, dtype=v_type).uniform_(-5, 5)
    else:
        values = torch.randint(-5, 6, (total_values_length,), device=device, dtype=v_type)

    batch_size = torch.tensor([batch_size_val], device=device, dtype=csr_type)

    return batch_size, csr_seg, values


@pytest.mark.parametrize("csr_type", [torch.int32, torch.int64])
@pytest.mark.parametrize("v_type", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_segment_sum_csr(csr_type, v_type, is_mxrec):
    torch.npu.set_device(DEVICE)
    batch_size, csr_seg, values = generate_random_segment_sum_data("cpu", csr_type, v_type)
    segment_sum_npu = get_op(batch_size.item(), csr_seg.to(DEVICE), values.to(DEVICE), is_mxrec)
    # golden用更高精度的cpu实现
    segment_sum_golden = get_golden(batch_size.item(), csr_seg, values.to(torch.float32)).to(v_type)

    if v_type == torch.float16:
        rtol, atol = 2 ** (-7), 2 ** (-7)
    elif v_type == torch.bfloat16:
        rtol, atol = 2 ** (-6), 2 ** (-6)
    else:
        rtol, atol = 2 ** (-9), 2 ** (-9)

    torch.testing.assert_close(segment_sum_npu.cpu(), segment_sum_golden, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
