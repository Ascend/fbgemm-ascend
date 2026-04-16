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

from pathlib import Path
import sysconfig
from typing import Optional
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch_npu
import fbgemm_gpu
import fbgemm_ascend

torch.npu.config.allow_internal_format = False
DEVICE = "npu:0"
REPEAT_TIMES = 1
# 是否测试另外两个指定类型的算子
ENABLE_TYPED_NBIT_OP_DISPATCH = True


@dataclass
class _ToleranceStats:
    quant_total: int = 0
    quant_abs_diff_nonzero: int = 0
    quant_abs_diff_eq1: int = 0
    scale_total: int = 0
    scale_in_tol_not_equal: int = 0
    bias_total: int = 0
    bias_in_tol_not_equal: int = 0


_STATS = _ToleranceStats()


# CPU 参考实现
def fused_nbit_rowwise_quantize_ref(
    input_data: np.ndarray, bit_rate: int
) -> np.ndarray:
    """
    FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf 的 NumPy 参考实现。

    算法与 CUDA 内核 _float_to_fusednbitrowwise_cuda_kernel 相同：
      1. 逐行求 min / max
      2. min 转 float16 再转回 float32
      3. scale = range / (2^bit_rate - 1)，转 float16
      4. 量化：q = round((x - min) / scale)，截断到 [0, 2^bit_rate - 1]
      5. 多值打包：num_elem_per_byte = 8 / bit_rate
    """
    assert input_data.ndim == 2, "输入必须为二维数组"
    nrows, ncols = input_data.shape
    num_elem_per_byte = 8 // bit_rate
    emb_bytes = (ncols + num_elem_per_byte - 1) // num_elem_per_byte
    output_columns = emb_bytes + 4
    max_quant = (1 << bit_rate) - 1

    output = np.zeros((nrows, output_columns), dtype=np.uint8)

    for row_idx in range(nrows):
        row = input_data[row_idx].astype(np.float32)
        min_val = float(np.min(row))
        max_val = float(np.max(row))

        min_fp16 = float(np.float16(min_val))
        range_val = max_val - min_fp16

        if range_val == 0.0:
            scale_fp16 = np.float16(1.0)
        else:
            scale_fp16 = np.float16(range_val / max_quant)

        scale_float = float(scale_fp16)
        if scale_float == 0.0:
            scale_fp16 = np.float16(1.0)
            scale_float = 1.0

        inv_scale = 1.0 / scale_float
        if np.isinf(inv_scale):
            scale_fp16 = np.float16(1.0)
            inv_scale = 1.0

        output[row_idx, emb_bytes : emb_bytes + 2] = np.array(
            [scale_fp16], dtype=np.float16
        ).view(np.uint8)
        output[row_idx, emb_bytes + 2 : emb_bytes + 4] = np.array(
            [np.float16(min_fp16)], dtype=np.float16
        ).view(np.uint8)

        q_row = np.clip(np.round((row - min_fp16) * inv_scale), 0, max_quant).astype(
            np.uint8
        )
        for col in range(ncols):
            q = int(q_row[col])
            byte_idx = col // num_elem_per_byte
            bit_off = (col % num_elem_per_byte) * bit_rate
            if bit_off == 0:
                output[row_idx, byte_idx] = q
            else:
                output[row_idx, byte_idx] |= q << bit_off

    return output


# NPU 算子调用包装器
def quantize_npu(
    input_tensor: torch.Tensor,
    bit_rate: int,
    device: str,
) -> np.ndarray:
    """NPU 算子调用"""
    torch.npu.set_device(device)
    npu_tensor = input_tensor.to(device)

    if ENABLE_TYPED_NBIT_OP_DISPATCH:
        if input_tensor.dtype == torch.float32:
            op_name = "FloatToFusedNBitRowwiseQuantizedSBHalf"
        elif input_tensor.dtype == torch.float16:
            op_name = "HalfToFusedNBitRowwiseQuantizedSBHalf"
        else:
            raise TypeError("仅支持 float32/float16")
    else:
        op_name = "FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf"

    op = getattr(torch.ops.fbgemm, op_name)
    torch.npu.synchronize()
    for _ in range(REPEAT_TIMES):
        result = op(npu_tensor, bit_rate)
        torch.npu.synchronize()
    return result.cpu().numpy()


# 容差比较：量化值允许误差<=1；scale/bias 允许 isclose(rtol=1e-3, atol=1e-5)
def _unpack_quantized_values(
    packed_u8: np.ndarray,
    bit_rate: int,
    ncols: int,
) -> np.ndarray:
    """将每行 packed quant bytes 解包为 (nrows, ncols) 的 uint8 量化值矩阵。"""
    if packed_u8.size == 0:
        return np.empty((packed_u8.shape[0], ncols), dtype=np.uint8)

    if bit_rate == 8:
        unpacked = packed_u8
    elif bit_rate == 4:
        lo = packed_u8 & 0x0F
        hi = (packed_u8 >> 4) & 0x0F
        unpacked = np.empty(
            (packed_u8.shape[0], packed_u8.shape[1] * 2), dtype=np.uint8
        )
        unpacked[:, 0::2] = lo
        unpacked[:, 1::2] = hi
    elif bit_rate == 2:
        v0 = packed_u8 & 0x03
        v1 = (packed_u8 >> 2) & 0x03
        v2 = (packed_u8 >> 4) & 0x03
        v3 = (packed_u8 >> 6) & 0x03
        unpacked = np.empty(
            (packed_u8.shape[0], packed_u8.shape[1] * 4), dtype=np.uint8
        )
        unpacked[:, 0::4] = v0
        unpacked[:, 1::4] = v1
        unpacked[:, 2::4] = v2
        unpacked[:, 3::4] = v3
    else:
        raise ValueError(f"不支持 bit_rate={bit_rate}，仅支持 2/4/8")

    return unpacked[:, :ncols]


def _extract_scale_bias_fp16(
    out_u8: np.ndarray,
    emb_bytes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """从 fused 输出中提取 scale/bias（均为 float16，返回 float32 便于比较）。"""
    nrows = out_u8.shape[0]
    scale = (
        out_u8[:, emb_bytes : emb_bytes + 2]
        .copy()
        .view(np.float16)
        .reshape(nrows)
        .astype(np.float32)
    )
    bias = (
        out_u8[:, emb_bytes + 2 : emb_bytes + 4]
        .copy()
        .view(np.float16)
        .reshape(nrows)
        .astype(np.float32)
    )
    return scale, bias


def assert_quantized_close(
    npu_out: np.ndarray,
    ref_out: np.ndarray,
    bit_rate: int,
    ncols: int,
    *,
    scale_rtol: float = 1e-3,
    scale_atol: float = 1e-5,
) -> None:
    """比较 NPU 输出与 CPU 参考输出：量化值误差<=1；scale/bias 使用 isclose 容差。"""
    assert npu_out.shape == ref_out.shape, (
        f"形状不匹配: NPU {npu_out.shape} vs REF {ref_out.shape}"
    )

    num_elem_per_byte = 8 // bit_rate
    emb_bytes = (ncols + num_elem_per_byte - 1) // num_elem_per_byte

    npu_q = _unpack_quantized_values(npu_out[:, :emb_bytes], bit_rate, ncols).astype(
        np.int16
    )
    ref_q = _unpack_quantized_values(ref_out[:, :emb_bytes], bit_rate, ncols).astype(
        np.int16
    )
    q_abs_diff = np.abs(npu_q - ref_q)

    total_q = int(q_abs_diff.size)
    q_nonzero = int(np.count_nonzero(q_abs_diff))
    q_eq1 = int(np.count_nonzero(q_abs_diff == 1))
    q_gt1 = int(np.count_nonzero(q_abs_diff > 1))
    _STATS.quant_total += total_q
    _STATS.quant_abs_diff_nonzero += q_nonzero
    _STATS.quant_abs_diff_eq1 += q_eq1

    if q_gt1 != 0:
        r, c = np.argwhere(q_abs_diff > 1)[0]
        raise AssertionError(
            f"量化值误差超过 1：bit_rate={bit_rate}, row={int(r)}, col={int(c)}, "
            f"NPU={int(npu_q[r, c])}, REF={int(ref_q[r, c])}, abs_diff={int(q_abs_diff[r, c])}"
        )

    npu_scale, npu_bias = _extract_scale_bias_fp16(npu_out, emb_bytes)
    ref_scale, ref_bias = _extract_scale_bias_fp16(ref_out, emb_bytes)

    scale_close = np.isclose(
        npu_scale, ref_scale, rtol=scale_rtol, atol=scale_atol, equal_nan=True
    )
    bias_close = np.isclose(
        npu_bias, ref_bias, rtol=scale_rtol, atol=scale_atol, equal_nan=True
    )

    # 统计“在容许误差内但不完全相等”的频率
    scale_not_equal = npu_scale != ref_scale
    bias_not_equal = npu_bias != ref_bias
    if npu_scale.size > 0:
        scale_in_tol_not_equal = int(np.count_nonzero(scale_close & scale_not_equal))
        bias_in_tol_not_equal = int(np.count_nonzero(bias_close & bias_not_equal))
        _STATS.scale_total += int(npu_scale.size)
        _STATS.scale_in_tol_not_equal += scale_in_tol_not_equal
        _STATS.bias_total += int(npu_bias.size)
        _STATS.bias_in_tol_not_equal += bias_in_tol_not_equal

    if not np.all(scale_close):
        idx = int(np.flatnonzero(~scale_close)[0])
        raise AssertionError(
            f"scale 超出容差：idx={idx}, NPU={float(npu_scale[idx])}, REF={float(ref_scale[idx])}, "
            f"abs_diff={float(abs(npu_scale[idx] - ref_scale[idx]))}"
        )

    if not np.all(bias_close):
        idx = int(np.flatnonzero(~bias_close)[0])
        raise AssertionError(
            f"bias 超出容差：idx={idx}, NPU={float(npu_bias[idx])}, REF={float(ref_bias[idx])}, "
            f"abs_diff={float(abs(npu_bias[idx] - ref_bias[idx]))}"
        )


def _print_tolerance_stats_once() -> None:
    if _STATS.quant_total == 0 and _STATS.scale_total == 0 and _STATS.bias_total == 0:
        return

    def _rate(n: int, d: int) -> float:
        return float(n) / float(d) if d else 0.0

    print(
        "\n========== FloatOrHalfToFusedNBitRowwise tolerance stats (all cases) =========="
    )
    print(
        f"[quant] abs_diff>0: {_STATS.quant_abs_diff_nonzero}/{_STATS.quant_total} "
        f"({_rate(_STATS.quant_abs_diff_nonzero, _STATS.quant_total):.6e}), "
        f"abs_diff==1: {_STATS.quant_abs_diff_eq1}/{_STATS.quant_total} "
        f"({_rate(_STATS.quant_abs_diff_eq1, _STATS.quant_total):.6e})"
    )
    print(
        f"[scale] in_tol_not_equal: {_STATS.scale_in_tol_not_equal}/{_STATS.scale_total} "
        f"({_rate(_STATS.scale_in_tol_not_equal, _STATS.scale_total):.6e}), "
        "rtol=1e-3 atol=1e-5"
    )
    print(
        f"[bias ] in_tol_not_equal: {_STATS.bias_in_tol_not_equal}/{_STATS.bias_total} "
        f"({_rate(_STATS.bias_in_tol_not_equal, _STATS.bias_total):.6e}), "
        "rtol=1e-3 atol=1e-5"
    )
    print(
        "==========================================================================\n"
    )


@pytest.fixture(scope="session", autouse=True)
def _print_stats_at_end_of_session():
    yield
    _print_tolerance_stats_once()


# ---------------------------------------------------------------------------
# 测试场景定义（供 test_correctness 参数化使用）
# ---------------------------------------------------------------------------

# 每条记录：(bit_rate, dtype_str, nrows, ncols_mult, lo, hi, fill, seed, sample_rows)
#   fill != None       → 全填充常数（range=0 边界情况）；fill == None → 随机数据
#   sample_rows == None → 验证全部行；sample_rows == int → 仅抽样前 N 行
_CORRECTNESS_CASES = [
    # --- range=0：scale 置 1，量化字节全 0 ---
    pytest.param(
        2, "float32", 4, 4, None, None, 0.0, None, None, id="all_same_zero_2bit"
    ),
    pytest.param(
        4, "float32", 4, 4, None, None, 1.0, None, None, id="all_same_pos_4bit"
    ),
    pytest.param(
        8, "float32", 4, 4, None, None, -3.14, None, None, id="all_same_neg_8bit"
    ),
    pytest.param(
        4, "float32", 4, 4, None, None, 65504.0, None, None, id="all_same_fp16max_4bit"
    ),
    # --- 单行 float32 / float16 ---
    pytest.param(
        2, "float32", 1, 8, -10.0, 10.0, None, 1001, None, id="single_row_fp32_2bit"
    ),
    pytest.param(
        4, "float32", 1, 8, -10.0, 10.0, None, 1001, None, id="single_row_fp32_4bit"
    ),
    pytest.param(
        2, "float16", 1, 8, -5.0, 5.0, None, 1002, None, id="single_row_fp16_2bit"
    ),
    pytest.param(
        4, "float16", 1, 8, -5.0, 5.0, None, 1002, None, id="single_row_fp16_4bit"
    ),
    # --- 纯负数 ---
    pytest.param(
        2, "float32", 6, 8, -100.0, -0.01, None, 2001, None, id="negative_2bit"
    ),
    pytest.param(
        4, "float32", 6, 8, -100.0, -0.01, None, 2001, None, id="negative_4bit"
    ),
    # --- 正负混合 ---
    pytest.param(
        2, "float32", 8, 16, -50.0, 50.0, None, 3001, None, id="mixed_sign_2bit"
    ),
    pytest.param(
        4, "float32", 8, 16, -50.0, 50.0, None, 3001, None, id="mixed_sign_4bit"
    ),
    # --- 接近 float16 最大值 ---
    pytest.param(
        2, "float32", 4, 4, 60000.0, 65000.0, None, 5001, None, id="near_fp16max_2bit"
    ),
    pytest.param(
        4, "float32", 4, 4, 60000.0, 65000.0, None, 5001, None, id="near_fp16max_4bit"
    ),
    # --- 极小值 ---
    pytest.param(
        2, "float32", 4, 4, 1e-6, 1e-4, None, 5002, None, id="very_small_2bit"
    ),
    pytest.param(
        4, "float32", 4, 4, 1e-6, 1e-4, None, 5002, None, id="very_small_4bit"
    ),
    # --- 多种形状 × bit_rate ---
    pytest.param(2, "float32", 1, 1, -1.0, 1.0, None, 4210, None, id="shape_1_1_2bit"),
    pytest.param(
        4, "float32", 10, 1, -1.0, 1.0, None, 4410, None, id="shape_10_1_4bit"
    ),
    pytest.param(
        8, "float32", 13, 3, -1.0, 1.0, None, 4810, None, id="shape_13_3_8bit"
    ),
    pytest.param(2, "float32", 63, 7, -1.0, 1.0, None, 4263, None, id="shape_63_2bit"),
    pytest.param(
        4, "float32", 101, 17, -1.0, 1.0, None, 4417, None, id="shape_101_4bit"
    ),
    pytest.param(
        4, "float16", 77, 1000, -3.0, 3.0, None, 42, None, id="big_ncols_fp16_4bit"
    ),
    pytest.param(
        2, "float32", 1024, 16, -1.0, 1.0, None, 6200, None, id="fp32_2bit_1k"
    ),
    pytest.param(
        2, "float16", 1024, 16, -1.0, 1.0, None, 6200, None, id="fp16_2bit_1k"
    ),
    pytest.param(
        4, "float32", 1024, 16, -1.0, 1.0, None, 6400, None, id="fp32_4bit_1k"
    ),
    pytest.param(
        4, "float16", 1024, 16, -1000, 1000, None, 6400, None, id="fp16_4bit_1k"
    ),
    pytest.param(
        4, "float32", 65536, 2, -1.0, 1.0, None, 2333, None, id="small_ncols_4bit_64k"
    ),
    # --- 大规模：抽样验证 ---
    pytest.param(
        2, "float32", 65536, 64, -10, 10, None, 6265, 20000, id="fp32_2bit_64k"
    ),
    pytest.param(
        2, "float16", 65536, 64, -1.0, 1.0, None, 6265, 20000, id="fp16_2bit_64k"
    ),
    pytest.param(
        4, "float32", 65536, 64, -1.0, 1.0, None, 6465, 20000, id="fp32_4bit_64k"
    ),
    pytest.param(
        4, "float16", 65536, 64, -1.0, 1.0, None, 6465, 20000, id="fp16_4bit_64k"
    ),
    pytest.param(
        4,
        "float16",
        10000000,
        16,
        -1.0,
        1.0,
        None,
        42,
        100000,
        id="fp16_4bit_10m",
    ),
]


class TestFloatOrHalfToFusedNbitRowwise:
    """FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf 算子测试类"""

    @staticmethod
    def _ncols_for(bit_rate: int, base: int = 1) -> int:
        return 2 * (8 // bit_rate) * base

    # 1. 空输入（nrows=0 或 ncols=0）
    @pytest.mark.parametrize("empty_dim", ["rows", "cols", "both"])
    @pytest.mark.parametrize("bit_rate", [2, 4, 8])
    @pytest.mark.parametrize("device", [DEVICE])
    def test_empty_input(self, device, bit_rate, empty_dim):
        """空输入：nrows=0 或 ncols=0 时输出形状正确且不崩溃"""
        num_elem_per_byte = 8 // bit_rate

        if empty_dim == "rows":
            ncols = self._ncols_for(bit_rate, base=4)
            inp = torch.empty(0, ncols, dtype=torch.float32)
            expected = (0, (ncols + num_elem_per_byte - 1) // num_elem_per_byte + 4)
        elif empty_dim == "cols":
            nrows = 8
            inp = torch.empty(nrows, 0, dtype=torch.float32)
            expected = (nrows, 4)
        else:
            inp = torch.empty(0, 0, dtype=torch.float32)
            expected = (0, 4)

        npu_out = quantize_npu(inp, bit_rate, device)

        assert npu_out.shape == expected, (
            f"空{empty_dim}输出形状错误: {npu_out.shape}, 期望 {expected}"
        )

    # 2. 正确性：覆盖各种数据场景
    @pytest.mark.parametrize(
        "bit_rate,dtype_str,nrows,ncols_mult,lo,hi,fill,seed,sample_rows",
        _CORRECTNESS_CASES,
    )
    @pytest.mark.parametrize("device", [DEVICE])
    def test_correctness(
        self,
        device: str,
        bit_rate: int,
        dtype_str: str,
        nrows: int,
        ncols_mult: int,
        lo: Optional[float],
        hi: Optional[float],
        fill: Optional[float],
        seed: Optional[int],
        sample_rows: Optional[int],
    ):
        """
        各种数据场景下 NPU 输出与 CPU 参考实现符合精度要求。
        误差有极小概率出现且NPU和GPU都不可避免，FBGEMM测试代码有误。
        sample_rows=None 时验证全部行；sample_rows=N 时仅抽样前 N 行（大规模场景）。
        """
        ncols = self._ncols_for(bit_rate, base=ncols_mult)

        if fill is not None:
            data_fp32 = np.full((nrows, ncols), fill, dtype=np.float32)
        else:
            np.random.seed(seed)
            data_fp32 = np.random.uniform(lo, hi, size=(nrows, ncols)).astype(
                np.float32
            )

        if dtype_str == "float16":
            data_fp16 = data_fp32.astype(np.float16)
            data_for_ref = data_fp16.astype(np.float32)
            inp = torch.from_numpy(data_fp16)
        else:
            data_for_ref = data_fp32
            inp = torch.from_numpy(data_fp32)

        npu_out = quantize_npu(inp, bit_rate, device)

        n = sample_rows if sample_rows is not None else nrows
        ref = fused_nbit_rowwise_quantize_ref(data_for_ref[:n], bit_rate)
        assert_quantized_close(npu_out[:n], ref, bit_rate, ncols)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
