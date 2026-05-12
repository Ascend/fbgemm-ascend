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

import numpy as np
import pytest
import torch
import torch_npu

import fbgemm_gpu
import fbgemm_ascend


def get_result(update_row_indices, update_table_indices, index_remappings, index_remappings_offsets):
    return torch.ops.fbgemm.pruned_array_lookup_from_row_idx(
        update_row_indices, update_table_indices, index_remappings, index_remappings_offsets
    )


@pytest.mark.parametrize("row_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("remap_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("device", ["npu:0"])
@pytest.mark.parametrize("T", [1, 5, 20])
@pytest.mark.parametrize("N", [0, 1, 10, 100, 1000, 10000, 128 * 1024 * 5 + 3])
def test_pruned_array_lookup_from_row_idx(row_dtype, remap_dtype, device, T, N):
    # Construct index_remappings and index_remappings_offsets
    offsets = [0]
    index_remappings_list = []
    current_offset = 0

    for t in range(T):
        # randomly decide if this table has capacity or not
        has_mapping = np.random.rand() > 0.5
        if has_mapping:
            capacity = np.random.randint(10, 100)
            # mapping values can be arbitrary int32/int64
            mapping = torch.randint(0, 1000, (capacity,), dtype=remap_dtype)
            index_remappings_list.append(mapping)
            current_offset += capacity
        offsets.append(current_offset)

    if len(index_remappings_list) > 0:
        index_remappings = torch.cat(index_remappings_list)
    else:
        index_remappings = torch.empty((0,), dtype=remap_dtype)

    index_remappings_offsets = torch.tensor(offsets, dtype=torch.int64)

    # Construct update_table_indices and update_row_indices
    update_table_indices = torch.randint(0, T, (N,), dtype=torch.int32)
    update_row_indices = torch.empty((N,), dtype=row_dtype)

    for i in range(N):
        t = update_table_indices[i].item()
        cap = offsets[t + 1] - offsets[t]
        if cap > 0:
            update_row_indices[i] = np.random.randint(0, cap)
        else:
            update_row_indices[i] = np.random.randint(0, 100)

    # Golden from CPU
    golden = get_result(update_row_indices, update_table_indices, index_remappings, index_remappings_offsets)

    # NPU Result
    result = get_result(
        update_row_indices.to(device),
        update_table_indices.to(device),
        index_remappings.to(device),
        index_remappings_offsets.to(device)
    ).cpu()

    assert torch.equal(result, golden)
