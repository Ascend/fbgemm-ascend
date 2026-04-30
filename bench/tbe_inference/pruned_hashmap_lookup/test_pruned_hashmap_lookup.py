#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging

import numpy as np
import pytest
import torch
import fbgemm_gpu  # type: ignore

DEVICE_ID = 0

if torch.cuda.is_available():
    torch.cuda.set_device(DEVICE_ID)
else:
    import torch_npu  # type: ignore
    import fbgemm_ascend
    torch_npu.npu.set_device(DEVICE_ID)

logging.basicConfig(level=logging.INFO)

# 装载因子，hash_table每个表大小=每个表的dense_indices_numel/LOAD_FACTOR，让hash_table中有空槽位
LOAD_FACTOR = 0.8
# 剪枝比例 用来计算稀疏索引和密集索引的值范围
PRUNING_RATIO = 0.5

TABLE_NUM_LIST = [1, 13, 31, 60, 73, 100]
BATCH_NUM_LIST = [1, 14, 73, 234, 1000]
LENGTH_LIST = [20, 55, 664]
PARAM_TYPES = [
    [torch.int32, torch.int32, torch.int64],
    [torch.int32, torch.int64, torch.int64],
    [torch.int64, torch.int32, torch.int64],
    [torch.int64, torch.int64, torch.int64],
]

def _assume(flag, msg: str = "Assumption failed"):
    if not flag:
        raise ValueError(msg)


def _get_device():
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return torch.npu.current_device()  # type: ignore


def _set_seed(seed: int = 32):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    else:
        torch.npu.manual_seed(seed)  # type: ignore


@pytest.mark.parametrize("table_num", TABLE_NUM_LIST)
@pytest.mark.parametrize("batch_num", BATCH_NUM_LIST)
@pytest.mark.parametrize("length", LENGTH_LIST)
@pytest.mark.parametrize("param_types", PARAM_TYPES)
def test_pruned_hashmap_lookup(
    table_num: int,
    batch_num: int,
    length: int,
    param_types: list[torch.dtype],
) -> None:
    _set_seed()
    logging.info(f"table_num:{table_num}, batch_num:{batch_num}, length:{length}, param_types:{param_types}")
    current_device = _get_device()
    indices_type, hash_table_type, hash_table_offsets_type = param_types
    is_valid_type = indices_type in [torch.int32, torch.int64] and hash_table_offsets_type in [torch.int64]

    # 稀疏索引的值的范围
    sparse_idx_range = int(batch_num * length / (1.0 - PRUNING_RATIO))
    if is_valid_type:
        idx_type_max = torch.iinfo(indices_type).max
        _assume(
            sparse_idx_range < idx_type_max,
            f"sparse_idx_range must be less than indices_type:{indices_type} max, "
            f"indices type max:{idx_type_max}, sparse_idx_range:{sparse_idx_range}.",
        )

    # 生成唯一的indices
    indices = torch.empty(size=(table_num, batch_num, length), dtype=indices_type)
    for t in range(table_num):
        np_table = np.random.choice(
            np.arange(sparse_idx_range, dtype=np.int64),
            size=(batch_num, length),
            replace=False,
        )
        indices[t] = torch.tensor(np_table, dtype=indices_type)
    indices = indices.view(-1)

    # 创建offsets
    offsets = torch.tensor([length * b_t for b_t in range(batch_num * table_num + 1)]).to(dtype=indices_type)

    # 生成致密索引
    dense_idx_range = int(batch_num * length / (1.0 - PRUNING_RATIO + 0.2))  # 致密索引范围小于稀疏索引范围
    dense_indices = (
        torch.randint(low=0, high=dense_idx_range, size=(table_num, batch_num, length)).view(-1).to(dtype=indices_type)
    )

    # 初始化hash_table和对应offsets
    # hash_table 中每个致密索引表的大小
    capacities = [int(batch_num * length / LOAD_FACTOR) for _ in range(table_num)]
    hash_table = torch.full(
        (sum(capacities), 2),
        -1,  # 填充-1
        dtype=hash_table_type,
    )
    hash_table_offsets = torch.tensor([0] + np.cumsum(capacities).tolist()).to(dtype=hash_table_offsets_type)

    # 将生成的dense_indices插入到hash_table中，调用fbgemm的CPU实现
    if is_valid_type:
        torch.ops.fbgemm.pruned_hashmap_insert(
            indices, dense_indices, offsets, hash_table, hash_table_offsets
        )

    indices = indices.to(current_device)
    dense_indices = dense_indices.to(current_device)
    offsets = offsets.to(current_device)
    hash_table = hash_table.to(current_device)
    hash_table_offsets = hash_table_offsets.to(current_device)

    # 查表致密索引
    dense_indices_lookup = torch.ops.fbgemm.pruned_hashmap_lookup(indices, offsets, hash_table, hash_table_offsets)

    # 验证结果
    torch.testing.assert_close(dense_indices, dense_indices_lookup)


@pytest.mark.parametrize(
    "param_types",
    [
        [torch.int32, torch.int32, torch.int32],
        [torch.int64, torch.int32, torch.int32],
        [torch.int64, torch.int32, torch.float],
        [torch.float, torch.int32, torch.float],
    ],
)
def test_invalid_dtype_param(
    param_types: list[torch.dtype],
):
    with pytest.raises(RuntimeError):
        test_pruned_hashmap_lookup(10, 10, 20, param_types)


def test_different_batch_length(has_empty_hash_table: bool = False):
    # 设定每个表3个batch
    batch_num_per_table = 3
    # indices长度 550，分为4个表，每个表index个数：120， 80， 200，150
    indices_num_per_table = [120, 80, 200, 150]
    offsets = torch.tensor([0, 50, 100, 120, 155, 175, 200, 300, 350, 400, 500, 530, 550], dtype=torch.int32)

    indices_list = []
    for i in indices_num_per_table:
        sparse_idx_range = int(i / (1 - PRUNING_RATIO))
        sub_indices = np.random.choice(
            np.arange(sparse_idx_range, dtype=np.int64),
            size=(i,),
            replace=False,
        )
        indices_list.append(torch.tensor(sub_indices, dtype=torch.int32))
    indices = torch.cat(indices_list, dim=0)

    dense_indices_list = []
    for i in indices_num_per_table:
        dense_idx_range = int(i / (1.0 - PRUNING_RATIO + 0.2))
        sub_indices = torch.randint(low=0, high=dense_idx_range, size=(i,)).view(-1).to(dtype=torch.int32)
        dense_indices_list.append(sub_indices)
    dense_indices = torch.cat(dense_indices_list, dim=0)

    capacities = [int(t_s / LOAD_FACTOR) + 1 for t_s in indices_num_per_table]
    if has_empty_hash_table:
        # 构造第三个hash_table为空的情况
        empty_hash_table_idx = 2
        capacities[empty_hash_table_idx] = 0
        # 处理空表的dense_indices，直接copy原始稀疏索引
        empty_table_idx_start = offsets[empty_hash_table_idx * batch_num_per_table]
        empty_table_idx_end = offsets[(empty_hash_table_idx + 1) * batch_num_per_table]
        dense_indices[empty_table_idx_start:empty_table_idx_end] = indices[empty_table_idx_start:empty_table_idx_end]

    hash_table = torch.full(
        (sum(capacities), 2),
        -1,  # 填充-1
        dtype=torch.int32,
    )
    hash_table_offsets = torch.tensor([0] + np.cumsum(capacities).tolist()).to(dtype=torch.int64)

    torch.ops.fbgemm.pruned_hashmap_insert(indices, dense_indices, offsets, hash_table, hash_table_offsets)
    current_device = _get_device()

    data_list = [indices, dense_indices, offsets, hash_table, hash_table_offsets]
    indices, dense_indices, offsets, hash_table, hash_table_offsets = [data.to(current_device) for data in data_list]

    # 查表致密索引
    dense_indices_lookup = torch.ops.fbgemm.pruned_hashmap_lookup(indices, offsets, hash_table, hash_table_offsets)

    # 验证结果
    torch.testing.assert_close(dense_indices, dense_indices_lookup)


def test_empty_hash_table():
    test_different_batch_length(has_empty_hash_table=True)


def test_diff_batch_per_table():
    # 测试表之间的batch个数不一样场景
    with pytest.raises(RuntimeError):
        indices_num_per_table = [120, 80, 200, 150]
        # 构造表之间的batch个数不一样的场景。下面10个batch，不能均分给4个表
        offsets = torch.tensor([0, 50, 100, 120, 200, 300, 350, 400, 500, 530, 550], dtype=torch.int32)
        indices_num: int = int(offsets[-1].item())
        indices = torch.randint(low=0, high=1000, size=(indices_num,), dtype=torch.int32)
        capacities = [100, 100, 100, 100]
        hash_table = torch.full(
            (sum(capacities), 2),
            -1,  # 填充-1
            dtype=torch.int32,
        )
        dense_indices = torch.randint(low=0, high=1000, size=(indices_num,), dtype=torch.int32)
        hash_table_offsets = torch.tensor([0] + np.cumsum(capacities).tolist()).to(dtype=torch.int64)

        current_device = _get_device()
        data_list = [indices, dense_indices, offsets, hash_table, hash_table_offsets]
        indices, dense_indices, offsets, hash_table, hash_table_offsets = [
            data.to(current_device)
            for data in data_list
        ]
        _ = torch.ops.fbgemm.pruned_hashmap_lookup(indices, offsets, hash_table, hash_table_offsets)


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
