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
"""
emb_inplace_update 算子测试用例。

昇腾 SIMT 暂不支持 UVM，仅覆盖 placement == DEVICE 的写入路径。
非 DEVICE 的记录由 kernel 静默跳过。
"""

import itertools
import logging
import random

import numpy as np
import pytest
import torch
import fbgemm_ascend  # noqa: F401

DEVICE = "npu:0"
logging.getLogger().setLevel(logging.INFO)

# ============================================================
# SparseType 枚举（与 FBGEMM embedding_common.h 对齐）
# ============================================================

SPARSE_TYPE_FP32 = 0
SPARSE_TYPE_FP16 = 1
SPARSE_TYPE_INT8 = 2
SPARSE_TYPE_INT4 = 3
SPARSE_TYPE_INT2 = 4
SPARSE_TYPE_BF16 = 5
SPARSE_TYPE_FP8 = 6

PLACEMENT_DEVICE = 0

# ============================================================
# 测试参数
# ============================================================

WEIGHT_TY_LIST = [
    SPARSE_TYPE_FP32,
    SPARSE_TYPE_FP16,
    SPARSE_TYPE_BF16,
    SPARSE_TYPE_INT8,
    SPARSE_TYPE_INT4,
    SPARSE_TYPE_INT2,
    SPARSE_TYPE_FP8,
]
ROW_ALIGNMENT_LIST = [16, 32, 64]
ROW_IDX_DTYPE_LIST = [torch.int32, torch.int64]


# ============================================================
# 工具函数
# ============================================================


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def unpadded_row_size_in_bytes(D, weight_ty):
    if weight_ty == SPARSE_TYPE_FP32:
        return D * 4
    elif weight_ty in (SPARSE_TYPE_FP16, SPARSE_TYPE_BF16):
        return D * 2
    elif weight_ty == SPARSE_TYPE_FP8:
        return D
    elif weight_ty == SPARSE_TYPE_INT8:
        return D + 4
    elif weight_ty == SPARSE_TYPE_INT4:
        return D // 2 + 4
    elif weight_ty == SPARSE_TYPE_INT2:
        return D // 4 + 4
    return 0


def padded_row_size_in_bytes(D, weight_ty, row_alignment):
    r = unpadded_row_size_in_bytes(D, weight_ty)
    if row_alignment <= 1:
        return r
    return ((r + row_alignment - 1) // row_alignment) * row_alignment


# ============================================================
# CPU 参考实现
# ============================================================


def emb_inplace_update_cpu_reference(
    dev_weights,
    weights_placements,
    weights_offsets,
    weights_tys,
    D_offsets,
    update_weights,
    update_table_indices,
    update_row_indices,
    update_offsets,
    row_alignment,
):
    """纯 Python 参考实现。仅处理 placement==DEVICE 的记录，其余跳过。"""
    dev_w = dev_weights.clone()
    N = update_row_indices.numel()

    for n in range(N):
        t = int(update_table_indices[n].item())
        placement = int(weights_placements[t].item())
        if placement != PLACEMENT_DEVICE:
            continue

        r = int(update_row_indices[n].item())
        D = int(D_offsets[t + 1].item()) - int(D_offsets[t].item())
        ty = int(weights_tys[t].item())
        Db = padded_row_size_in_bytes(D, ty, row_alignment)

        src_off = int(update_offsets[n].item())
        dst_off = int(weights_offsets[t].item()) + Db * r
        dev_w[dst_off : dst_off + Db] = update_weights[src_off : src_off + Db]

    return dev_w


# ============================================================
# 测试数据生成
# ============================================================


def generate_test_data(T, D_range, rows_range, N, weight_ty_list, row_alignment, row_idx_dtype=torch.int32):
    """生成一组完整的测试输入（CPU tensor）。"""
    D_offsets_list = [0]
    weights_offsets_list = []
    weights_tys_list = []
    dev_weights_size = 0
    table_rows = []

    for t in range(T):
        ty = weight_ty_list[t % len(weight_ty_list)]
        weights_tys_list.append(ty)

        D = random.randint(D_range[0], D_range[1])
        if ty == SPARSE_TYPE_INT4:
            D = max(D - D % 2, 2)
        elif ty == SPARSE_TYPE_INT2:
            D = max(D - D % 4, 4)
        D_offsets_list.append(D_offsets_list[-1] + D)

        total_rows = random.randint(rows_range[0], rows_range[1])
        table_rows.append(total_rows)
        Db = padded_row_size_in_bytes(D, ty, row_alignment)

        weights_offsets_list.append(dev_weights_size)
        dev_weights_size += Db * total_rows

    dev_weights_size = max(dev_weights_size, 1)
    dev_weights = torch.randint(0, 256, (dev_weights_size,), dtype=torch.uint8)
    uvm_weights = torch.zeros(1, dtype=torch.uint8)

    update_table_indices_list = []
    update_row_indices_list = []
    update_offsets_list = [0]
    update_bytes_total = 0

    if N > 0:
        all_pairs = [(t, r) for t in range(T) for r in range(table_rows[t])]
        actual_n = min(N, len(all_pairs))
        sampled_pairs = random.sample(all_pairs, actual_n)
        random.shuffle(sampled_pairs)

        for t, r in sampled_pairs:
            update_table_indices_list.append(t)
            update_row_indices_list.append(r)
            D = D_offsets_list[t + 1] - D_offsets_list[t]
            ty = weights_tys_list[t]
            Db = padded_row_size_in_bytes(D, ty, row_alignment)
            update_bytes_total += Db
            update_offsets_list.append(update_bytes_total)

    update_weights = torch.randint(0, 256, (max(update_bytes_total, 1),), dtype=torch.uint8)

    return {
        "dev_weights": dev_weights,
        "uvm_weights": uvm_weights,
        "weights_placements": torch.full((T,), PLACEMENT_DEVICE, dtype=torch.int32),
        "weights_offsets": torch.tensor(weights_offsets_list, dtype=torch.int64),
        "weights_tys": torch.tensor(weights_tys_list, dtype=torch.uint8),
        "D_offsets": torch.tensor(D_offsets_list, dtype=torch.int32),
        "update_weights": update_weights,
        "update_table_indices": torch.tensor(update_table_indices_list, dtype=torch.int32),
        "update_row_indices": torch.tensor(update_row_indices_list, dtype=row_idx_dtype),
        "update_offsets": torch.tensor(update_offsets_list, dtype=torch.int64),
        "row_alignment": row_alignment,
    }


# ============================================================
# 核心测试逻辑
# ============================================================


def run_test(data):
    """对比 CPU 参考实现与 NPU 算子输出（逐字节比对 dev_weights）。"""
    # CPU oracle
    dev_ref = emb_inplace_update_cpu_reference(
        data["dev_weights"],
        data["weights_placements"],
        data["weights_offsets"],
        data["weights_tys"],
        data["D_offsets"],
        data["update_weights"],
        data["update_table_indices"],
        data["update_row_indices"],
        data["update_offsets"],
        data["row_alignment"],
    )

    # 搬到 NPU
    dev_w_npu = data["dev_weights"].clone().to(DEVICE)
    uvm_w_npu = data["uvm_weights"].clone().to(DEVICE)
    placements_npu = data["weights_placements"].to(DEVICE)
    offsets_npu = data["weights_offsets"].to(DEVICE)
    tys_npu = data["weights_tys"].to(DEVICE)
    d_offsets_npu = data["D_offsets"].to(DEVICE)
    upd_w_npu = data["update_weights"].to(DEVICE)
    upd_t_npu = data["update_table_indices"].to(DEVICE)
    upd_r_npu = data["update_row_indices"].to(DEVICE)
    upd_off_npu = data["update_offsets"].to(DEVICE)

    # 调用 NPU 算子
    torch.ops.fbgemm.emb_inplace_update(
        dev_w_npu,
        uvm_w_npu,
        placements_npu,
        offsets_npu,
        tys_npu,
        d_offsets_npu,
        upd_w_npu,
        upd_t_npu,
        upd_r_npu,
        upd_off_npu,
        data["row_alignment"],
    )

    # 搬回 CPU 比对
    dev_w_result = dev_w_npu.cpu()
    assert torch.equal(dev_w_result, dev_ref), (
        f"dev_weights mismatch!\n  diff positions: {torch.nonzero(dev_w_result != dev_ref).flatten().tolist()[:20]}"
    )


# ============================================================
# 基本功能测试
# ============================================================


@pytest.mark.parametrize("weight_ty", WEIGHT_TY_LIST)
def test_all_sparse_types(weight_ty):
    """覆盖所有量化类型。"""
    set_seed(300 + weight_ty)
    D_min = 4 if weight_ty in (SPARSE_TYPE_INT2,) else (2 if weight_ty == SPARSE_TYPE_INT4 else 4)
    data = generate_test_data(
        T=2,
        D_range=(D_min, 32),
        rows_range=(10, 50),
        N=20,
        weight_ty_list=[weight_ty],
        row_alignment=16,
    )
    run_test(data)


@pytest.mark.parametrize("row_alignment", ROW_ALIGNMENT_LIST)
def test_row_alignment_variants(row_alignment):
    """覆盖不同 row_alignment。"""
    set_seed(400 + row_alignment)
    data = generate_test_data(
        T=4,
        D_range=(4, 32),
        rows_range=(10, 50),
        N=30,
        weight_ty_list=[SPARSE_TYPE_INT8, SPARSE_TYPE_FP16],
        row_alignment=row_alignment,
    )
    run_test(data)


@pytest.mark.parametrize("row_idx_dtype", ROW_IDX_DTYPE_LIST)
def test_row_idx_dtype(row_idx_dtype):
    """覆盖 int32/int64 行索引类型。"""
    set_seed(500)
    data = generate_test_data(
        T=3,
        D_range=(8, 16),
        rows_range=(10, 30),
        N=15,
        weight_ty_list=[SPARSE_TYPE_FP32],
        row_alignment=16,
        row_idx_dtype=row_idx_dtype,
    )
    run_test(data)


def test_multi_table_mixed_dtype():
    """多表混合量化类型。"""
    set_seed(102)
    data = generate_test_data(
        T=5,
        D_range=(4, 64),
        rows_range=(50, 200),
        N=100,
        weight_ty_list=[
            SPARSE_TYPE_FP32,
            SPARSE_TYPE_FP16,
            SPARSE_TYPE_INT8,
            SPARSE_TYPE_INT4,
            SPARSE_TYPE_INT2,
        ],
        row_alignment=16,
    )
    run_test(data)


# ============================================================
# 边界测试
# ============================================================


def test_empty_update_n_zero():
    """N=0 时 dev_weights 不被修改。"""
    set_seed(200)
    data = generate_test_data(
        T=3,
        D_range=(8, 32),
        rows_range=(10, 50),
        N=0,
        weight_ty_list=[SPARSE_TYPE_FP16],
        row_alignment=16,
    )
    run_test(data)


def test_single_record():
    """单条更新记录。"""
    set_seed(201)
    data = generate_test_data(
        T=1,
        D_range=(4, 4),
        rows_range=(1, 1),
        N=1,
        weight_ty_list=[SPARSE_TYPE_INT8],
        row_alignment=16,
    )
    run_test(data)


def test_mixed_placement_skip_non_device():
    """非 DEVICE 表的记录应被静默跳过，DEVICE 表正常更新。"""
    set_seed(600)
    T = 4
    data = generate_test_data(
        T=T,
        D_range=(8, 32),
        rows_range=(10, 50),
        N=30,
        weight_ty_list=[SPARSE_TYPE_FP16],
        row_alignment=16,
    )
    # 将部分表的 placement 设为非 DEVICE
    data["weights_placements"] = torch.tensor([0, 1, 0, 2], dtype=torch.int32)
    run_test(data)


# ============================================================
# 大规模测试
# ============================================================


def test_large_scale():
    """大规模：T=32, N=10000。"""
    set_seed(700)
    data = generate_test_data(
        T=32,
        D_range=(8, 128),
        rows_range=(100, 1000),
        N=10000,
        weight_ty_list=[
            SPARSE_TYPE_FP32,
            SPARSE_TYPE_FP16,
            SPARSE_TYPE_INT8,
            SPARSE_TYPE_INT4,
        ],
        row_alignment=16,
    )
    run_test(data)


def test_large_D():
    """大行宽：D=512, FP32 → Db=2048B。"""
    set_seed(701)
    data = generate_test_data(
        T=2,
        D_range=(512, 512),
        rows_range=(10, 10),
        N=5,
        weight_ty_list=[SPARSE_TYPE_FP32],
        row_alignment=32,
    )
    run_test(data)


# ============================================================
# 大维度 D 上限探测
# ============================================================

# D 取值逐步翻倍，探测 NPU 支持的最大维度
LARGE_D_LIST = [256, 512, 1024, 2048, 4096, 8192, 16384]


@pytest.mark.parametrize("D", LARGE_D_LIST)
def test_large_D_limit(D):
    """探测大维度 D 的上限。FP32 下 Db = D*4，对齐到 16。"""
    set_seed(800 + D)
    data = generate_test_data(
        T=1,
        D_range=(D, D),
        rows_range=(2, 2),
        N=2,
        weight_ty_list=[SPARSE_TYPE_FP32],
        row_alignment=16,
    )
    run_test(data)


@pytest.mark.parametrize("D", LARGE_D_LIST)
def test_large_D_limit_fp16(D):
    """探测大维度 D 的上限。FP16 下 Db = D*2，对齐到 16。"""
    set_seed(900 + D)
    data = generate_test_data(
        T=1,
        D_range=(D, D),
        rows_range=(2, 2),
        N=2,
        weight_ty_list=[SPARSE_TYPE_FP16],
        row_alignment=16,
    )
    run_test(data)


# ============================================================
# 多表 × 多类型 × 多维度 排列组合随机测试
# ============================================================

# 表数量
COMBO_T_LIST = [1, 4, 8, 16, 32]
# 特征维度范围
COMBO_D_RANGES = [(4, 16), (16, 64), (64, 256), (128, 512)]
# 更新记录数
COMBO_N_LIST = [1, 10, 100, 500, 2000]
# row_alignment
COMBO_ALIGNMENT_LIST = [16, 32]

# 生成排列组合参数（取部分代表性组合，避免用例爆炸）
COMBO_PARAMS = list(itertools.product(COMBO_T_LIST, COMBO_D_RANGES, COMBO_N_LIST, COMBO_ALIGNMENT_LIST))
# 从全组合中随机采样 100 个用例
random.seed(12345)
COMBO_PARAMS_SAMPLED = random.sample(COMBO_PARAMS, min(100, len(COMBO_PARAMS)))


@pytest.mark.parametrize("T,D_range,N,row_alignment", COMBO_PARAMS_SAMPLED)
def test_combo_random_multi_table(T, D_range, N, row_alignment):
    """多表 × 多类型 × 多维度排列组合：每张表随机分配量化类型和维度。"""
    set_seed(1000 + T * 100 + N)
    data = generate_test_data(
        T=T,
        D_range=D_range,
        rows_range=(5, 100),
        N=N,
        weight_ty_list=WEIGHT_TY_LIST,  # 7 种类型轮流分配
        row_alignment=row_alignment,
    )
    run_test(data)


@pytest.mark.parametrize("T", COMBO_T_LIST)
def test_all_types_per_table(T):
    """每张表分配不同量化类型，验证混合场景。"""
    set_seed(2000 + T)
    data = generate_test_data(
        T=T,
        D_range=(8, 128),
        rows_range=(10, 50),
        N=min(T * 5, 200),
        weight_ty_list=WEIGHT_TY_LIST,
        row_alignment=16,
    )
    run_test(data)


@pytest.mark.parametrize("weight_ty", WEIGHT_TY_LIST)
@pytest.mark.parametrize("T", [1, 8, 32])
def test_single_type_multi_table(weight_ty, T):
    """同一量化类型 × 不同表数量。"""
    set_seed(3000 + weight_ty * 100 + T)
    D_min = 4 if weight_ty in (SPARSE_TYPE_INT2,) else (2 if weight_ty == SPARSE_TYPE_INT4 else 4)
    data = generate_test_data(
        T=T,
        D_range=(D_min, 64),
        rows_range=(10, 100),
        N=min(T * 10, 500),
        weight_ty_list=[weight_ty],
        row_alignment=16,
    )
    run_test(data)


@pytest.mark.parametrize("row_idx_dtype", ROW_IDX_DTYPE_LIST)
@pytest.mark.parametrize("T", [4, 16])
@pytest.mark.parametrize("row_alignment", [16, 32])
def test_dtype_table_alignment_combo(row_idx_dtype, T, row_alignment):
    """行索引类型 × 表数量 × 对齐 三维组合。"""
    set_seed(4000 + T * 10 + row_alignment)
    data = generate_test_data(
        T=T,
        D_range=(8, 128),
        rows_range=(10, 50),
        N=min(T * 8, 300),
        weight_ty_list=WEIGHT_TY_LIST,
        row_alignment=row_alignment,
        row_idx_dtype=row_idx_dtype,
    )
    run_test(data)


@pytest.mark.parametrize("round_idx", range(100))
def test_stress_random_100_rounds(round_idx):
    """压力测试：100 轮随机参数，每轮一个独立用例。"""
    set_seed(5000 + round_idx)
    T = random.choice([1, 2, 4, 8, 16, 32])
    D_lo = random.choice([4, 8, 16, 32, 64])
    D_hi = D_lo * random.choice([1, 2, 4, 8])
    N = random.choice([0, 1, 5, 20, 100, 500])
    row_alignment = random.choice([16, 32, 64])
    ty_count = random.randint(1, len(WEIGHT_TY_LIST))
    weight_ty_list = random.sample(WEIGHT_TY_LIST, ty_count)

    data = generate_test_data(
        T=T,
        D_range=(D_lo, D_hi),
        rows_range=(2, 50),
        N=N,
        weight_ty_list=weight_ty_list,
        row_alignment=row_alignment,
        row_idx_dtype=random.choice(ROW_IDX_DTYPE_LIST),
    )
    run_test(data)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
