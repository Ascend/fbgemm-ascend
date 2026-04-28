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
#        limitations under the License.
# ==============================================================================

"""
linearize_cache_indices算子的测试用例
"""

import torch
import logging
import pytest
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)d in %(funcName)s] %(levelname)s: %(message)s"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.propagate = False


if torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
    DEVICE = "cuda:0"
    import fbgemm_gpu
    import fbgemm_gpu.sparse_ops
elif torch.npu.is_available():
    import torch_npu

    DEVICE_TYPE = "npu"
    DEVICE = "npu:0"
    torch.npu.config.allow_internal_format = False
    import fbgemm_ascend
else:
    raise RuntimeError("Neither CUDA nor NPU is available")

logger.info(f"Using device type: {DEVICE_TYPE}, device: {DEVICE}")


def device_sync(device_type):
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "npu":
        torch.npu.synchronize()


def linearize_cache_indices_device(
    cache_hash_size_cumsum,
    indices,
    offsets,
    B_offsets,
    max_B,
    indices_base_offset,
    device_type,
    device,
):
    if device_type == "npu":
        import torch_npu

        torch.npu.set_device(device)
        torch.npu.config.allow_internal_format = False
    elif device_type == "cuda":
        torch.cuda.set_device(device)

    cache_hash_size_cumsum = cache_hash_size_cumsum.to(device)
    indices = indices.to(device)
    offsets = offsets.to(device)
    B_offsets_device = B_offsets.to(device) if B_offsets is not None else None
    B_offsets_param = B_offsets_device if B_offsets_device is not None else None

    result = torch.ops.fbgemm.linearize_cache_indices(
        cache_hash_size_cumsum,
        indices,
        offsets,
        B_offsets_param,
        max_B,
        indices_base_offset,
    )
    device_sync(device_type)
    return result.cpu()


def execute_linearize_cache_indices_ref(
    hash_size_cumsum, indices, offsets, B_offsets=None, max_B=-1
):
    T = hash_size_cumsum.numel() - 1
    B_offsets_ = None

    if B_offsets is not None:
        assert max_B > 0, "Invalid max_B"
        B_offsets_ = B_offsets.cpu().tolist()
        use_vbe = True
    else:
        B = (offsets.numel() - 1) // T
        use_vbe = False

    offsets_ = offsets.cpu().tolist()
    max_offset = hash_size_cumsum[-1].to(indices.dtype)
    linear_cache_indices = indices.detach().clone()

    if use_vbe:
        table_offsets_cuda = []
        for t in range(1, T):
            b_offset_idx = B_offsets_[t]
            table_offsets_cuda.append(offsets_[b_offset_idx])

        for idx in range(len(indices)):
            left, right = 0, len(table_offsets_cuda)
            while left != right:
                middle = left + (right - left) // 2
                if table_offsets_cuda[middle] <= idx:
                    left = middle + 1
                else:
                    right = middle
            table_idx = left

            hash_size_offset = hash_size_cumsum[table_idx]
            if hash_size_offset >= 0 and indices[idx] >= 0:
                linear_cache_indices[idx] = indices[idx] + hash_size_offset
            else:
                linear_cache_indices[idx] = max_offset
    else:
        for t in range(T):
            hash_size_offset = hash_size_cumsum[t]
            indices_start = offsets_[t * B]
            indices_end = offsets_[(t + 1) * B]
            if hash_size_offset >= 0:
                linear_cache_indices[indices_start:indices_end] += hash_size_offset
            else:
                linear_cache_indices[indices_start:indices_end] = max_offset

    pruned_pos = (indices < 0).nonzero(as_tuple=True)
    if len(pruned_pos) > 0:
        linear_cache_indices[pruned_pos] = max_offset
    return linear_cache_indices


def compare_tensors(cpu_tensor, device_tensor, name="Tensor", atol=1e-4, rtol=1e-4):
    if cpu_tensor.shape != device_tensor.shape:
        logger.error(
            f"✗ {name} shape不一致: CPU={cpu_tensor.shape}, Device={device_tensor.shape}"
        )
        return False
    if torch.allclose(cpu_tensor, device_tensor, rtol=rtol, atol=atol):
        return True
    else:
        max_diff = (cpu_tensor - device_tensor).abs().max()
        logger.error(f"✗ {name}与CPU不一致, 最大差异: {max_diff}")
        return False


def gen_test_data(
    num_tables,
    batch_size,
    num_indices=None,
    use_vbe=False,
    pruned_ratio=0.0,
    cached_ratio=1.0,
    hash_scale=100,
    seed=42,
):
    """统一的数据生成函数

    Args:
        num_tables: 表数量 T
        batch_size: 批次大小 B (use_vbe时为每表平均batch)
        num_indices: 索引数量，默认T*B
        use_vbe: 是否使用VBE模式
        pruned_ratio: 剪枝比例 [0, 1]
        cached_ratio: 缓存比例 [0, 1]，<1表示部分表未缓存
        hash_scale: hash偏移量的缩放因子
        seed: 随机种子

    Returns:
        hash_size_cumsum, indices, offsets, B_offsets, max_B
    """
    random.seed(seed)

    if num_indices is None:
        num_indices = num_tables * batch_size

    T = num_tables
    B = num_indices // T if not use_vbe else batch_size

    # 生成 hash_size_cumsum
    num_cached = int(T * cached_ratio)
    hash_size_cumsum = [i * hash_scale if i < num_cached else -1 for i in range(T)]
    hash_size_cumsum.append(T * hash_scale)  # max_offset

    # 生成 indices
    if use_vbe:
        indices = list(range(num_indices))
    else:
        indices = [i % 50 for i in range(num_indices)]

    # 计算 B
    if batch_size is not None:
        B = batch_size
    else:
        if num_indices is not None and num_indices > 0:
            B = max(1, num_indices // T)
        else:
            B = 1

    # 确保 num_indices 有值
    if num_indices is None:
        num_indices = T * B

    # 生成 offsets
    if use_vbe:
        offsets = [0]
        for t in range(T):
            offsets.append(offsets[-1] + B + t % 3)
        offsets[-1] = num_indices
    else:
        offsets = [i * B for i in range(T + 1)]
        offsets[-1] = num_indices

    # 生成 B_offsets (VBE模式)
    B_offsets = list(range(T + 1)) if use_vbe else None
    max_B = B + 2 if use_vbe else -1

    # 剪枝处理
    if pruned_ratio > 0:
        num_pruned = int(num_indices * pruned_ratio)
        for _ in range(num_pruned):
            indices[random.randint(0, len(indices) - 1)] = -1

    return hash_size_cumsum, indices, offsets, B_offsets, max_B


@pytest.mark.parametrize("num_tables", [4])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("num_indices", [None, 100000, 1000000])
@pytest.mark.parametrize("use_vbe", [False, True])
@pytest.mark.parametrize("pruned_ratio", [0.0, 1.0])
@pytest.mark.parametrize("cached_ratio", [1.0])
@pytest.mark.parametrize("hash_scale", [100, 1000000])
@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("offsets_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("b_offsets_dtype", [torch.int32, torch.int64])
def test_linearize_cache_indices(
    num_tables,
    batch_size,
    num_indices,
    use_vbe,
    pruned_ratio,
    cached_ratio,
    hash_scale,
    seed,
    indices_dtype,
    offsets_dtype,
    b_offsets_dtype,
):
    if use_vbe and batch_size is None:
        batch_size = 4

    hash_size_cumsum, indices, offsets, B_offsets, max_B = gen_test_data(
        num_tables,
        batch_size,
        num_indices,
        use_vbe,
        pruned_ratio,
        cached_ratio,
        hash_scale,
        seed,
    )

    hash_size_cumsum_t = torch.tensor(hash_size_cumsum, dtype=torch.int64, device="cpu")
    indices_t = torch.tensor(indices, dtype=indices_dtype, device="cpu")
    offsets_t = torch.tensor(offsets, dtype=offsets_dtype, device="cpu")
    B_offsets_t = (
        torch.tensor(B_offsets, dtype=b_offsets_dtype, device="cpu")
        if B_offsets is not None
        else None
    )

    output_ref = execute_linearize_cache_indices_ref(
        hash_size_cumsum_t, indices_t, offsets_t, B_offsets_t, max_B
    )
    output_ref = output_ref.to(torch.int64)
    output_test = linearize_cache_indices_device(
        hash_size_cumsum_t,
        indices_t,
        offsets_t,
        B_offsets_t,
        max_B,
        0,
        DEVICE_TYPE,
        DEVICE,
    )
    assert compare_tensors(output_ref, output_test)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
