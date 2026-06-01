#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
import logging
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401

import fbgemm_ascend  # noqa: F401
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import EmbeddingLocation, PoolingMode
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.utils import generate_requests, round_up

DEVICE = "npu:0"
INT8_QPARAMS_BYTES = 8

# 配置日志
logging.getLogger().setLevel(logging.INFO)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, 'npu'):
        torch.npu.manual_seed_all(seed)


set_seed(10000)


def _print_debug(golden: torch.Tensor, test: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-4) -> None:
    golden_flat = golden.detach().cpu().flatten()
    test_flat = test.detach().cpu().flatten()
    diff = test_flat - golden_flat

    # 使用容差判断真正不匹配的元素（而不是简单的 != 比较）
    # 使用与 torch.testing.assert_close 相同的逻辑
    abs_diff = diff.abs()
    rel_diff = abs_diff / (golden_flat.abs() + rtol)
    mismatch_mask = (abs_diff > atol) & (rel_diff > rtol)
    mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False).squeeze(-1)

    if mismatch_indices.numel() > 0:
        num_mismatches = mismatch_indices.numel()
        logging.info(
            "Found %s mismatched elements (total: %s, rtol=%s, atol=%s)",
            num_mismatches,
            golden_flat.numel(),
            rtol,
            atol,
        )

        # 按顺序打印前20个错误
        max_print = min(20, num_mismatches)
        print_indices = mismatch_indices[:max_print]

        logging.info("Mismatched elements (position, golden_value, test_value, abs_diff, rel_diff):")
        for idx in print_indices:
            idx_val = idx.item()
            golden_val = golden_flat[idx_val].item()
            test_val = test_flat[idx_val].item()
            abs_diff_val = abs_diff[idx_val].item()
            rel_diff_val = rel_diff[idx_val].item()
            logging.info(
                "  [%s]: golden=%.6f, test=%.6f, abs_diff=%.6e, rel_diff=%.6e",
                idx_val,
                golden_val,
                test_val,
                abs_diff_val,
                rel_diff_val,
            )

        if num_mismatches > max_print:
            logging.info("... and %s more mismatches (not shown)", num_mismatches - max_print)
    else:
        logging.info("No mismatches found (within tolerance)")


def _print_nonfinite_mismatches(
    golden: torch.Tensor,
    test: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    max_print: int = 20,
) -> None:
    gold = golden.detach().cpu()
    test_cpu = test.detach().cpu()
    close_mask = torch.isclose(test_cpu, gold, rtol=rtol, atol=atol, equal_nan=True)
    mismatch = torch.nonzero(~close_mask, as_tuple=False)

    if mismatch.numel() == 0:
        logging.info("No mismatches found by isclose(equal_nan=True)")
        return

    logging.info(
        "Found %s mismatches with equal_nan=True (showing first %s)",
        mismatch.shape[0],
        min(max_print, mismatch.shape[0]),
    )

    for idx_tensor in mismatch[:max_print]:
        idx = tuple(idx_tensor.tolist())
        gv = gold[idx]
        tv = test_cpu[idx]
        logging.info(
            "mismatch %s: golden=%s test=%s "
            "gold.isnan=%s test.isnan=%s "
            "gold.isposinf=%s test.isposinf=%s "
            "gold.isneginf=%s test.isneginf=%s "
            "gold.isfinite=%s test.isfinite=%s",
            idx,
            gv.item(),
            tv.item(),
            torch.isnan(gv).item(),
            torch.isnan(tv).item(),
            torch.isposinf(gv).item(),
            torch.isposinf(tv).item(),
            torch.isneginf(gv).item(),
            torch.isneginf(tv).item(),
            torch.isfinite(gv).item(),
            torch.isfinite(tv).item(),
        )


def _extract_fused8bit_rowwise_scales(quantized: torch.Tensor) -> torch.Tensor:
    quantized_cpu = quantized.detach().cpu().contiguous()
    qparams = quantized_cpu[:, -INT8_QPARAMS_BYTES:].contiguous().view(torch.float32)
    return qparams[:, 0].contiguous()


def _build_int8_golden_output_with_atol(
    fp32_output: torch.Tensor,
    ref_module: IntNBitTableBatchedEmbeddingBagsCodegen,
    pooling_mode: PoolingMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp32_output = fp32_output.cpu().to(torch.float32).contiguous()

    if pooling_mode == PoolingMode.NONE:
        quantized = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(fp32_output.contiguous()).contiguous()
        dequantized = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(quantized).contiguous()
        scales = _extract_fused8bit_rowwise_scales(quantized)
        return dequantized, scales.unsqueeze(1).expand_as(dequantized).contiguous()

    d_offsets = ref_module.D_offsets.cpu().to(torch.int64)
    dequantized = []
    atols = []
    for t in range(d_offsets.numel() - 1):
        d_start = int(d_offsets[t].item())
        d_end = int(d_offsets[t + 1].item())
        table = fp32_output[:, d_start:d_end].contiguous()
        quantized = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(table).contiguous()
        dequantized_table = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(quantized).contiguous()
        scales = _extract_fused8bit_rowwise_scales(quantized)
        dequantized.append(dequantized_table)
        atols.append(scales.unsqueeze(1).expand_as(dequantized_table).contiguous())
    return torch.cat(dequantized, dim=1), torch.cat(atols, dim=1)


def _dequantize_int8_test_output_with_atol(
    output: torch.Tensor,
    ref_module: IntNBitTableBatchedEmbeddingBagsCodegen,
    pooling_mode: PoolingMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_cpu = output.detach().cpu().contiguous()

    if pooling_mode == PoolingMode.NONE:
        dequantized = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(output_cpu).contiguous()
        scales = _extract_fused8bit_rowwise_scales(output_cpu)
        return dequantized, scales.unsqueeze(1).expand_as(dequantized).contiguous()

    d_offsets = ref_module.D_offsets.cpu().to(torch.int64)
    split_sizes = []
    for t in range(d_offsets.numel() - 1):
        dim = int(d_offsets[t + 1].item() - d_offsets[t].item())
        split_sizes.append(dim + INT8_QPARAMS_BYTES)
    per_table = torch.split(output_cpu, split_sizes, dim=1)
    dequantized = []
    atols = []
    for table in per_table:
        dequantized_table = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(table.contiguous()).contiguous()
        scales = _extract_fused8bit_rowwise_scales(table)
        dequantized.append(dequantized_table)
        atols.append(scales.unsqueeze(1).expand_as(dequantized_table).contiguous())
    return torch.cat(dequantized, dim=1), torch.cat(atols, dim=1)


def _assert_close_with_tensor_atol(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol_tensor: torch.Tensor,
    rtol: float,
    equal_nan: bool = True,
) -> None:
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    atol_cpu = atol_tensor.detach().cpu()

    abs_diff = (actual_cpu - expected_cpu).abs()
    tol_tensor = atol_cpu + rtol * expected_cpu.abs()
    close_mask = abs_diff <= tol_tensor

    finite_equal = torch.isfinite(actual_cpu) & torch.isfinite(expected_cpu) & close_mask
    posinf_equal = torch.isposinf(actual_cpu) & torch.isposinf(expected_cpu)
    neginf_equal = torch.isneginf(actual_cpu) & torch.isneginf(expected_cpu)
    if equal_nan:
        nan_equal = torch.isnan(actual_cpu) & torch.isnan(expected_cpu)
    else:
        nan_equal = torch.zeros_like(close_mask, dtype=torch.bool)

    all_close = finite_equal | posinf_equal | neginf_equal | nan_equal
    if torch.all(all_close):
        return

    mismatch = torch.nonzero(~all_close, as_tuple=False)
    max_diff = abs_diff[~all_close].max().item()
    raise AssertionError(
        "Tensor-likes are not close!\n\n"
        f"Mismatched elements: {mismatch.shape[0]} / {expected_cpu.numel()} "
        f"({100.0 * mismatch.shape[0] / expected_cpu.numel():.1f}%)\n"
        f"Greatest absolute difference: {max_diff}"
    )


def call_operator(
    op: IntNBitTableBatchedEmbeddingBagsCodegen,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    per_sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.fbgemm.int_nbit_split_embedding_codegen_lookup_function(
        dev_weights=op.weights_dev,
        uvm_weights=op.weights_uvm,
        weights_placements=op.weights_placements,
        weights_offsets=op.weights_offsets,
        weights_tys=op.weights_tys,
        D_offsets=op.D_offsets,
        total_D=op.total_D,
        max_int2_D=op.max_int2_D,
        max_int4_D=op.max_int4_D,
        max_int8_D=op.max_int8_D,
        max_float16_D=op.max_float16_D,
        max_float32_D=op.max_float32_D,
        indices=indices,
        offsets=offsets,
        pooling_mode=int(op.pooling_mode),
        indice_weights=per_sample_weights,
        output_dtype=op.output_dtype,
        lxu_cache_weights=None,
        lxu_cache_locations=None,
        row_alignment=op.row_alignment,
        max_float8_D=op.max_float8_D,
        fp8_exponent_bits=op.fp8_exponent_bits,
        fp8_exponent_bias=op.fp8_exponent_bias,
    )


def _build_module(
    *,
    embedding_specs: List[Tuple[int, int, SparseType]],
    pooling_mode: PoolingMode,
    output_dtype: SparseType,
    indices_dtype: torch.dtype,
    device: str,
) -> IntNBitTableBatchedEmbeddingBagsCodegen:
    is_cpu = device.startswith("cpu")
    location = EmbeddingLocation.HOST if is_cpu else EmbeddingLocation.DEVICE
    specs = [("table", E, D, wty, location) for (E, D, wty) in embedding_specs]
    return IntNBitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=specs,
        device=device,
        pooling_mode=pooling_mode,
        output_dtype=output_dtype,
        indices_dtype=indices_dtype,
    )


def _sync_random_weights(
    ref_module: IntNBitTableBatchedEmbeddingBagsCodegen,
    test_module: IntNBitTableBatchedEmbeddingBagsCodegen,
    weight_types: List[SparseType],
) -> None:
    ref_module.fill_random_weights()
    test_module.fill_random_weights()
    ref_split = ref_module.split_embedding_weights()
    test_split = test_module.split_embedding_weights()
    for (ref_w, ref_scale), (test_w, test_scale), weight_ty in zip(ref_split, test_split, weight_types):
        if weight_ty == SparseType.FP16:
            ref_w.copy_(
                torch.empty_like(ref_w.view(torch.float16), dtype=torch.float16).uniform_(-1.0, 1.0).view(torch.uint8)
            )
        elif weight_ty == SparseType.FP32:
            ref_w.copy_(
                torch.empty_like(ref_w.view(torch.float32), dtype=torch.float32).uniform_(-1.0, 1.0).view(torch.uint8)
            )
        test_w.copy_(ref_w)
        if ref_scale is not None and test_scale is not None:
            test_scale.copy_(ref_scale)


def _copy_module_weights(
    src_module: IntNBitTableBatchedEmbeddingBagsCodegen,
    dst_module: IntNBitTableBatchedEmbeddingBagsCodegen,
) -> None:
    src_split = src_module.split_embedding_weights()
    dst_split = dst_module.split_embedding_weights()
    for (src_w, src_scale), (dst_w, dst_scale) in zip(src_split, dst_split):
        dst_w.copy_(src_w)
        if src_scale is not None and dst_scale is not None:
            dst_scale.copy_(src_scale)


def _print_shadow_fp32_debug(
    *,
    config: "TestConfig",
    embedding_specs: List[Tuple[int, int, SparseType]],
    indices: torch.Tensor,
    offsets: torch.Tensor,
    per_sample_weights: Optional[torch.Tensor],
    ref_module: IntNBitTableBatchedEmbeddingBagsCodegen,
    test_module: IntNBitTableBatchedEmbeddingBagsCodegen,
    rtol: float,
    atol: float,
    max_print: int = 20,
) -> None:
    shadow_ref = _build_module(
        embedding_specs=embedding_specs,
        pooling_mode=config.pooling_mode,
        output_dtype=SparseType.FP32,
        indices_dtype=config.indices_dtype,
        device="cpu",
    )
    shadow_test = _build_module(
        embedding_specs=embedding_specs,
        pooling_mode=config.pooling_mode,
        output_dtype=SparseType.FP32,
        indices_dtype=config.indices_dtype,
        device=DEVICE,
    )
    shadow_ref.fill_random_weights()
    shadow_test.fill_random_weights()
    _copy_module_weights(ref_module, shadow_ref)
    _copy_module_weights(test_module, shadow_test)

    shadow_golden = shadow_ref(
        indices=indices,
        offsets=offsets,
        per_sample_weights=per_sample_weights,
    ).cpu()
    shadow_test_out = call_operator(
        shadow_test,
        indices.to(device=DEVICE),
        offsets.to(device=DEVICE),
        per_sample_weights.to(device=DEVICE) if per_sample_weights is not None else None,
    ).cpu()

    close_mask = torch.isclose(shadow_test_out, shadow_golden, rtol=rtol, atol=atol, equal_nan=True)
    mismatch = torch.nonzero(~close_mask, as_tuple=False)
    logging.info(
        "Shadow FP32 mismatches: %d / %d",
        mismatch.shape[0],
        shadow_golden.numel(),
    )
    for idx_tensor in mismatch[:max_print]:
        idx = tuple(idx_tensor.tolist())
        gv = shadow_golden[idx]
        tv = shadow_test_out[idx]
        logging.info(
            "shadow_fp32 %s: golden=%s test=%s "
            "gold.isnan=%s test.isnan=%s "
            "gold.isposinf=%s test.isposinf=%s "
            "gold.isneginf=%s test.isneginf=%s "
            "gold.isfinite=%s test.isfinite=%s",
            idx,
            gv.item(),
            tv.item(),
            torch.isnan(gv).item(),
            torch.isnan(tv).item(),
            torch.isposinf(gv).item(),
            torch.isposinf(tv).item(),
            torch.isneginf(gv).item(),
            torch.isneginf(tv).item(),
            torch.isfinite(gv).item(),
            torch.isfinite(tv).item(),
        )


def _generate_workload(
    *,
    B: int,
    T: int,
    L: List[int],  # 每张表的bag长度: [L1, L2, ...]
    tables: List[Tuple[int, int]],  # 每张表的shape: [(E1, D1), (E2, D2), ...]
    weighted: bool,
    emulate_pruning: bool = False,
    indices_dtype: torch.dtype,
):
    """
    生成workload数据，支持每张表使用不同的bag长度

    Args:
        B: batch size
        T: 表的数量
        L: 每张表的bag长度列表，长度必须等于T
        tables: 每张表的shape列表
        weighted: 是否使用权重
        emulate_pruning: 是否模拟剪枝
        indices_dtype: indices的数据类型
    """
    assert len(L) == T, f"L的长度({len(L)})必须等于表的数量({T})"
    assert len(tables) == T, f"tables的长度({len(tables)})必须等于表的数量({T})"

    # 为每张表分别生成请求，然后合并
    all_indices_list = []
    all_weights_list = [] if weighted else None

    # 累积的indices数量，用于调整offsets
    current_indices_offset = 0

    for t in range(T):
        E_t, D_t = tables[t]
        L_t = L[t]

        # 为当前表生成请求
        requests = generate_requests(
            1,
            B,
            1,
            L_t,
            E_t,
            reuse=0.1,
            weighted=weighted,
            emulate_pruning=emulate_pruning,
            use_cpu=True,
            deterministic_output=True,  # 确保生成可重复的测试数据
        )

        for req in requests:
            if weighted:
                indices_t, offsets_t, per_sample_weights_t = req.unpack_3()
            else:
                indices_t, offsets_t = req.unpack_2()
                per_sample_weights_t = None

            all_indices_list.append(indices_t)

            if weighted and per_sample_weights_t is not None:
                all_weights_list.append(per_sample_weights_t)

            # 更新累积的indices数量
            current_indices_offset += len(indices_t)

    # 合并所有表的数据
    all_indices = torch.cat(all_indices_list).to(dtype=indices_dtype)

    # 合并offsets：TBE格式是 [0, L1, 2*L1, ..., B*L1, B*L1+L2, B*L1+2*L2, ..., B*L1+B*L2, ...]
    # 即：每张表有B+1个offsets（包括起始的0），最后一张表的最后一个offset是总长度
    all_offsets = [0]
    for t in range(T):
        L_t = L[t]
        # 每张表的offsets（除了第一个0）
        for b in range(1, B + 1):
            all_offsets.append(all_offsets[-1] + L_t)
    all_offsets = torch.tensor(all_offsets, dtype=indices_dtype)

    if weighted and all_weights_list is not None:
        all_weights = torch.cat(all_weights_list)
    else:
        all_weights = None

    yield (all_indices, all_offsets, all_weights)


def _generate_table_config(
    table_num: int,
    L: int,
    weights_ty: SparseType,
    is_nobag: bool = False,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    生成随机表配置

    Args:
        table_num: 表的数量
        L: bag长度的取值范围
        is_nobag: 是否为nobag模式（nobag模式下所有表的D维度必须相同）

    Returns:
        (tables, L_list): tables是(E, D)的列表，L_list是每张表的bag长度列表
    """
    tables = []
    L_list = []

    if weights_ty == SparseType.FP32:
        max_d_base = 512
    elif weights_ty == SparseType.INT8:
        max_d_base = 1023
    elif weights_ty == SparseType.INT4:
        max_d_base = 1022
    elif weights_ty == SparseType.INT2:
        max_d_base = 1020
    else:
        max_d_base = 1024

    # nobag模式下，所有表的D维度必须相同，在循环外生成
    if is_nobag:
        D_base = random.randint(1, max_d_base)
        D = D_base * 4
    else:
        D = 0  # 避免 pylint 警告 possibly-used-before-assignment

    for _ in range(table_num):
        # 行数：1-20000之间随机选择
        E = random.randint(1, 20000)

        # 列数：1-1024之间随机选择一个数乘以4（确保是4的倍数，符合FP8对齐要求）
        if not is_nobag:
            D_base = random.randint(1, max_d_base)
            D = D_base * 4

        tables.append((E, D))

        # bag长度：从1到L之间随机选择
        L_t = random.randint(1, L)
        L_list.append(L_t)

    return tables, L_list


def _generate_mixed_table_config(
    weight_types: List[SparseType],
    L: int,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    tables = []
    L_list = []

    for weight_ty in weight_types:
        if weight_ty == SparseType.FP32:
            max_d_base = 512
        elif weight_ty == SparseType.INT8:
            max_d_base = 1023
        elif weight_ty == SparseType.INT4:
            max_d_base = 1022
        elif weight_ty == SparseType.INT2:
            max_d_base = 1020
        else:
            max_d_base = 1024

        E = random.randint(1, 20000)
        D = random.randint(1, max_d_base) * 4
        tables.append((E, D))
        L_list.append(random.randint(1, L))

    return tables, L_list


@dataclass
class TestConfig:
    """测试配置参数"""

    pooling_mode: PoolingMode
    weighted: bool
    weights_ty: SparseType
    indices_dtype: torch.dtype
    output_dtype: SparseType
    B: int
    table_num: Optional[int] = None
    L: Optional[int] = None
    tables: Optional[List[Tuple[int, int]]] = None
    L_list: Optional[List[int]] = None
    weights_ty_list: Optional[List[SparseType]] = None


def _run_multi_table_test(config: TestConfig) -> None:
    """
    执行多表测试的公共逻辑

    Args:
        config: 测试配置参数
    """
    if config.weights_ty_list is not None:
        assert config.pooling_mode != PoolingMode.NONE, (
            "Mixed weights bench only covers pooled modes to match community coverage"
        )

    # 如果提供了预定义的表配置，使用它们；否则随机生成
    if config.tables is not None and config.L_list is not None:
        T = len(config.tables)
        assert len(config.L_list) == T, f"L_list的长度({len(config.L_list)})必须等于表的数量({T})"
        tables = config.tables
        L_list = config.L_list
    else:
        assert config.table_num is not None and config.L is not None, (
            "必须提供table_num和L（随机生成）或tables和L_list（预定义）"
        )
        if config.weights_ty_list is not None:
            tables, L_list = _generate_mixed_table_config(config.weights_ty_list, config.L)
        else:
            is_nobag = config.pooling_mode == PoolingMode.NONE
            tables, L_list = _generate_table_config(
                config.table_num,
                config.L,
                config.weights_ty,
                is_nobag=is_nobag,
            )
        T = len(tables)

    active_weight_types = config.weights_ty_list if config.weights_ty_list is not None else [config.weights_ty] * T
    assert len(active_weight_types) == T, f"weights_ty_list的长度({len(active_weight_types)})必须等于表的数量({T})"

    # 生成embedding_specs，对D进行对齐处理
    embedding_specs = []
    for (E, D), weight_ty in zip(tables, active_weight_types):
        D_aligned = round_up(D, max(weight_ty.align_size(), config.output_dtype.align_size()))
        embedding_specs.append((E, D_aligned, weight_ty))

    # 构建模块
    ref_output_dtype = (
        SparseType.FP32 if config.output_dtype in (SparseType.INT8, SparseType.BF16) else config.output_dtype
    )
    ref_module = _build_module(
        embedding_specs=embedding_specs,
        pooling_mode=config.pooling_mode,
        output_dtype=ref_output_dtype,
        indices_dtype=config.indices_dtype,
        device="cpu",
    )
    test_module = _build_module(
        embedding_specs=embedding_specs,
        pooling_mode=config.pooling_mode,
        output_dtype=config.output_dtype,
        indices_dtype=config.indices_dtype,
        device=DEVICE,
    )

    # 同步权重
    _sync_random_weights(ref_module, test_module, active_weight_types)

    # 执行测试
    for indices, offsets, per_sample_weights in _generate_workload(
        B=config.B, T=T, L=L_list, tables=tables, weighted=config.weighted, indices_dtype=config.indices_dtype
    ):
        psw = per_sample_weights
        golden_out = ref_module(
            indices=indices,
            offsets=offsets,
            per_sample_weights=psw,
        )
        projected_test_out = test_out = None  # keep scope explicit
        if config.output_dtype == SparseType.INT8:
            golden_out, golden_atol = _build_int8_golden_output_with_atol(golden_out, ref_module, config.pooling_mode)
        elif config.output_dtype == SparseType.BF16:
            golden_out = golden_out.to(torch.bfloat16)
        indices_npu = indices.to(device=DEVICE)
        offsets_npu = offsets.to(device=DEVICE)
        psw_npu = psw.to(device=DEVICE) if psw is not None else None
        test_out = call_operator(test_module, indices_npu, offsets_npu, psw_npu)
        if config.output_dtype == SparseType.INT8:
            projected_test_out, test_atol = _dequantize_int8_test_output_with_atol(
                test_out, ref_module, config.pooling_mode
            )
            assert golden_out.shape == projected_test_out.shape
        else:
            assert golden_out.shape == test_out.shape

        # 计算容差
        tol = (
            1e-3
            if config.output_dtype == SparseType.FP16
            else 2**-7
            if config.output_dtype == SparseType.BF16
            else 1e-4
            if config.output_dtype == SparseType.FP32
            else 1e-2
            if config.output_dtype == SparseType.INT8
            else 0
        )

        try:
            if config.output_dtype == SparseType.INT8:
                _assert_close_with_tensor_atol(
                    projected_test_out,
                    golden_out.cpu(),
                    torch.maximum(golden_atol, test_atol),
                    rtol=tol,
                    equal_nan=True,
                )
            else:
                torch.testing.assert_close(test_out.cpu(), golden_out.cpu(), rtol=tol, atol=tol, equal_nan=True)
        except AssertionError as err:
            debug_test = projected_test_out if config.output_dtype == SparseType.INT8 else test_out
            _print_debug(golden_out, debug_test, rtol=tol, atol=tol)
            if config.output_dtype != SparseType.INT8:
                _print_nonfinite_mismatches(golden_out, test_out, rtol=tol, atol=tol)
            if config.output_dtype == SparseType.BF16 and config.weights_ty in (SparseType.FP16, SparseType.FP32):
                _print_shadow_fp32_debug(
                    config=config,
                    embedding_specs=embedding_specs,
                    indices=indices,
                    offsets=offsets,
                    per_sample_weights=psw,
                    ref_module=ref_module,
                    test_module=test_module,
                    rtol=1e-4,
                    atol=1e-4,
                )
            raise err


POOLING_CASES = [
    (PoolingMode.SUM, False, torch.int32),
    (PoolingMode.SUM, True, torch.int32),
    (PoolingMode.SUM, False, torch.int64),
    (PoolingMode.SUM, True, torch.int64),
    (PoolingMode.MEAN, False, torch.int32),
    (PoolingMode.MEAN, True, torch.int32),
    (PoolingMode.MEAN, False, torch.int64),
    (PoolingMode.MEAN, True, torch.int64),
]

NOBAG_CASES = [
    (PoolingMode.NONE, False, torch.int32),
]

FORWARD_CASES = POOLING_CASES + NOBAG_CASES


@pytest.mark.parametrize("forward_case", FORWARD_CASES)
@pytest.mark.parametrize("output_dtype", [SparseType.FP16, SparseType.FP32, SparseType.BF16, SparseType.INT8])
@pytest.mark.parametrize(
    "weights_ty", [SparseType.FP8, SparseType.FP16, SparseType.FP32, SparseType.INT8, SparseType.INT4, SparseType.INT2]
)
@pytest.mark.parametrize("table_num", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_random_multi_table_forward(
    forward_case: Tuple[PoolingMode, bool, torch.dtype],
    output_dtype: SparseType,
    weights_ty: SparseType,
    table_num: int,
) -> None:
    """
    统一的多表测试用例，支持bag和nobag模式

    - 表的个数由table_num指定
    - 每张表的shape随机生成：行数在1-20000之间，列数是1-1024之间随机数乘以4（确保是4的倍数）
    - 每张表的bag长度从1-L之间随机选择
    - nobag模式下，所有表的D维度必须相同
    - 只参数化有效的(pooling_mode, weighted, indices_dtype)组合
    """
    default_B = 64
    default_L = 100
    pooling_mode, weighted, indices_dtype = forward_case

    if weights_ty == SparseType.INT2 and output_dtype == SparseType.FP32:
        pytest.skip("INT2 + FP32 output matches GPU test skip")

    config = TestConfig(
        pooling_mode=pooling_mode,
        weighted=weighted,
        weights_ty=weights_ty,
        indices_dtype=indices_dtype,
        output_dtype=output_dtype,
        B=default_B,
        table_num=table_num,
        L=default_L,
    )
    _run_multi_table_test(config)


MIXED_WEIGHT_CASES = [
    (SparseType.FP32, SparseType.FP16),
    (SparseType.FP8, SparseType.INT8),
    (SparseType.FP32, SparseType.FP16, SparseType.FP8, SparseType.INT8),
    (SparseType.FP16, SparseType.FP8, SparseType.INT8, SparseType.INT4),
    (SparseType.FP32, SparseType.FP16, SparseType.FP8, SparseType.INT8, SparseType.INT4, SparseType.INT2),
    (SparseType.FP16, SparseType.FP32, SparseType.FP8, SparseType.INT8, SparseType.INT4, SparseType.FP16),
    (
        SparseType.FP32,
        SparseType.FP16,
        SparseType.FP8,
        SparseType.INT8,
        SparseType.INT4,
        SparseType.FP16,
        SparseType.FP32,
        SparseType.INT8,
    ),
    (
        SparseType.FP16,
        SparseType.FP8,
        SparseType.INT8,
        SparseType.INT4,
        SparseType.FP32,
        SparseType.FP16,
        SparseType.INT8,
        SparseType.INT2,
    ),
]


@pytest.mark.parametrize("forward_case", POOLING_CASES)
@pytest.mark.parametrize("output_dtype", [SparseType.FP16, SparseType.FP32, SparseType.BF16, SparseType.INT8])
@pytest.mark.parametrize("mixed_weight_case", MIXED_WEIGHT_CASES)
def test_random_multi_table_forward_mixed_weights(
    forward_case: Tuple[PoolingMode, bool, torch.dtype],
    output_dtype: SparseType,
    mixed_weight_case: Tuple[SparseType, ...],
) -> None:
    """
    Mixed weights bench only covers pooled modes to match community coverage.
    """
    pooling_mode, weighted, indices_dtype = forward_case

    if output_dtype == SparseType.FP32 and SparseType.INT2 in mixed_weight_case:
        pytest.skip("Mixed INT2 + FP32 output matches GPU/community skip")

    config = TestConfig(
        pooling_mode=pooling_mode,
        weighted=weighted,
        weights_ty=SparseType.INT8,
        weights_ty_list=list(mixed_weight_case),
        indices_dtype=indices_dtype,
        output_dtype=output_dtype,
        B=64,
        table_num=len(mixed_weight_case),
        L=100,
    )
    _run_multi_table_test(config)
