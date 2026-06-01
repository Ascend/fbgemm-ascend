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

#ifndef INT_NBIT_SPLIT_EMBEDDING_COMMON_H
#define INT_NBIT_SPLIT_EMBEDDING_COMMON_H

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

// 常量定义
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int DATA_TYPE_INT32 = 0;
constexpr int DATA_TYPE_INT64 = 1;

constexpr int USE_QUEUE_NUM = 2;
constexpr int FLOAT_ALIGNMENT = 8;
constexpr int ALIGN = 32;
constexpr int64_t MAX_INDICES_ONE_BLOCK = 1024;
constexpr int64_t INT8_QPARAMS_BYTES = 8;
constexpr float INT8_QPARAM_SCALE_DIVISOR = 255.0f;
constexpr float INT8_QPARAM_EPS = 1.0e-8f;
constexpr uint8_t FP8_SIGN_MASK = 0x80;
constexpr uint8_t FP8_BODY_MASK = 0x7F;

// FP8反量化相关常量
constexpr uint32_t FP8_SIGN_SHIFT = 24U;         // FP8符号位的左移量（用于位打包）
constexpr uint32_t FP8_BODY_SHIFT_OFFSET = 16U;  // FP8 body shift计算中的偏移量
constexpr uint32_t FP8_EXPONENT_MAX = 254U;      // FP8 exponent的最大值（用于multiplier计算）
constexpr uint32_t FLOAT32_EXPONENT_BIAS = 23U;  // IEEE 754 float32的指数位偏移量
constexpr int32_t INT_DEQUANT_SIMT_MAX_THREADS = 1024;
constexpr int32_t U32B = 4;
constexpr int32_t I8_PAD = 4;
constexpr int32_t I8_N = 4;
constexpr uint32_t I8_M = 0xFFU;
constexpr int32_t I8_S8 = 8;
constexpr int32_t I8_S16 = 16;
constexpr int32_t I8_S24 = 24;
constexpr int32_t I4_PAD = 8;
constexpr int32_t I4_N = 8;
constexpr uint32_t I4_M = 0xFU;
constexpr int32_t I4_S4 = 4;
constexpr int32_t I4_S8 = 8;
constexpr int32_t I4_S12 = 12;
constexpr int32_t I4_S16 = 16;
constexpr int32_t I4_S20 = 20;
constexpr int32_t I4_S24 = 24;
constexpr int32_t I4_S28 = 28;
constexpr int32_t I2_PAD = 16;
constexpr int32_t I2_N = 16;
constexpr uint32_t I2_M = 0x3U;
constexpr int32_t I2_S2 = 2;
constexpr int32_t I2_S4 = 4;
constexpr int32_t I2_S6 = 6;
constexpr int32_t I2_S8 = 8;
constexpr int32_t I2_S10 = 10;
constexpr int32_t I2_S12 = 12;
constexpr int32_t I2_S14 = 14;
constexpr int32_t I2_S16 = 16;
constexpr int32_t I2_S18 = 18;
constexpr int32_t I2_S20 = 20;
constexpr int32_t I2_S22 = 22;
constexpr int32_t I2_S24 = 24;
constexpr int32_t I2_S26 = 26;
constexpr int32_t I2_S28 = 28;
constexpr int32_t I2_S30 = 30;

// PoolingMode枚举
enum class PoolingMode {
    SUM = 0,
    MEAN = 1,
    NONE = 2
};

enum class PlacementType : uint8_t {
    DEVICE = 0,
    MANAGED = 1,
    MANAGED_CACHING = 2,
    HOST = 3
};

// SparseType枚举
enum class SparseType : uint8_t {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
    INT4 = 3,
    INT2 = 4,
    BF16 = 5,
    FP8 = 6,
    INVALID = 7
};

__aicore__ inline int64_t GetInputRowBytes(int64_t embedDim, int64_t rowAlignment, SparseType weightType)
{
    int64_t inputBytes = 0;
    switch (weightType) {
        case SparseType::FP32:
            inputBytes = embedDim * static_cast<int64_t>(sizeof(float));
            break;
        case SparseType::FP16:
            inputBytes = embedDim * static_cast<int64_t>(sizeof(half));
            break;
        case SparseType::FP8:
            inputBytes = embedDim;
            break;
        case SparseType::INT8:
            inputBytes = embedDim + static_cast<int64_t>(2 * sizeof(half));
            break;
        case SparseType::INT4:
            inputBytes = embedDim / 2 + static_cast<int64_t>(2 * sizeof(half));
            break;
        case SparseType::INT2:
            inputBytes = embedDim / 4 + static_cast<int64_t>(2 * sizeof(half));
            break;
        default:
            inputBytes = 0;
            break;
    }
    return (inputBytes + rowAlignment - 1) / rowAlignment * rowAlignment;
}

__aicore__ inline int64_t GetAlignedInputRowBytes(int64_t embedDim, int64_t rowAlignment, SparseType weightType)
{
    return AlignTo32(GetInputRowBytes(embedDim, rowAlignment, weightType));
}

__aicore__ inline int64_t GetAlignedFloatRowBytes(int64_t embedDim)
{
    return AlignTo32(embedDim * static_cast<int64_t>(sizeof(float)));
}

__aicore__ inline int64_t GetAlignedFloatRowElems(int64_t embedDim)
{
    return GetAlignedFloatRowBytes(embedDim) / static_cast<int64_t>(sizeof(float));
}

__aicore__ inline int64_t GetDecodedFloatRowElems(int64_t embedDim, int64_t rowAlignment, SparseType weightType)
{
    switch (weightType) {
        case SparseType::FP8:
            return GetAlignedInputRowBytes(embedDim, rowAlignment, weightType);
        case SparseType::FP16:
            return GetAlignedFloatRowElems(embedDim);
        case SparseType::FP32:
            return GetAlignedFloatRowElems(embedDim);
        case SparseType::INT8:
            return GetAlignedFloatRowElems(embedDim);
        case SparseType::INT4:
            return GetAlignedFloatRowElems(embedDim);
        case SparseType::INT2:
            return GetAlignedFloatRowElems(embedDim);
        default:
            return 0;
    }
}

__aicore__ inline int64_t GetDecodedFloatRowBytes(int64_t embedDim, int64_t rowAlignment, SparseType weightType)
{
    return GetDecodedFloatRowElems(embedDim, rowAlignment, weightType) * static_cast<int64_t>(sizeof(float));
}

__aicore__ inline int32_t GetIntDequantThreadNum(int64_t totalWords)
{
    if (totalWords <= 32) {
        return 32;
    }
    if (totalWords >= INT_DEQUANT_SIMT_MAX_THREADS) {
        return INT_DEQUANT_SIMT_MAX_THREADS;
    }
    return static_cast<int32_t>(((totalWords + 31) / 32) * 32);
}

// Args结构体
struct Args {
    GM_ADDR devWeights;
    GM_ADDR uvmWeights;
    GM_ADDR lxuCacheWeights;
    GM_ADDR weightsPlacements;
    GM_ADDR weightsOffsets;
    GM_ADDR weightsTys;
    GM_ADDR dOffsets;
    GM_ADDR indices;
    GM_ADDR offsets;
    GM_ADDR lxuCacheLocations;
    GM_ADDR offsetPerKey;  // 每张表在offsets中的起始位置
    GM_ADDR indiceWeights;
    GM_ADDR out;
    GM_ADDR tiling;
    GM_ADDR workspace;
};

// ========== 数据拷贝函数 ==========

template <typename T>
__aicore__ inline void CpGm2Local(const LocalTensor<T>& lt, const GlobalTensor<T>& gt, int64_t len)
{
    uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
    uint32_t unAlignLen = len * sizeof(T) - alignLen;

    DataCopy(lt, gt, alignLen / sizeof(T));
    if (unAlignLen != 0) {
        const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
        const DataCopyPadExtParams<T> dataCopyPadExtParams{false, 0, 0, 0};
        DataCopyPad(lt[alignLen / sizeof(T)], gt[alignLen / sizeof(T)], dataCopyExtParams, dataCopyPadExtParams);
    }
}

template <typename T>
__aicore__ inline void CpLocal2Gm(const GlobalTensor<T>& gt, const LocalTensor<T>& lt, int64_t len)
{
    uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
    uint32_t unAlignLen = len * sizeof(T) - alignLen;

    DataCopy(gt, lt, alignLen / sizeof(T));
    if (unAlignLen != 0) {
        const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
        DataCopyPad(gt[alignLen / sizeof(T)], lt[alignLen / sizeof(T)], dataCopyExtParams);
    }
}

// ========== FP8反量化公共函数 ==========

// 初始化FP8参数
__aicore__ inline void InitFp8Params(int64_t fp8ExponentBits, int64_t fp8ExponentBias, uint32_t& fp8BodyShift,
                                     float& fp8Multiplier)
{
    fp8BodyShift = static_cast<uint32_t>(fp8ExponentBits + FP8_BODY_SHIFT_OFFSET);
    union {
        uint32_t u32;
        float f32;
    } multUnion;
    multUnion.u32 = static_cast<uint32_t>(FP8_EXPONENT_MAX - fp8ExponentBias) << FLOAT32_EXPONENT_BIAS;
    fp8Multiplier = multUnion.f32;
}

// FP8到FP32的位打包转换
template <typename MaskBuf, typename TmpByteBuf, typename SignUint32Buf>
__aicore__ inline void FP8U8ToFP32Bitpack(const LocalTensor<uint8_t>& src, LocalTensor<float>& dst,
                                          int64_t elementCount, uint32_t fp8BodyShift, float fp8Multiplier,
                                          MaskBuf& fp8MaskBuf, TmpByteBuf& fp8TmpByteBuf,
                                          SignUint32Buf& fp8SignUint32Buf)
{
    if (elementCount <= 0) {
        return;
    }
    LocalTensor<uint8_t> maskLt = fp8MaskBuf.template Get<uint8_t>();
    LocalTensor<uint8_t> tmpU8Lt = fp8TmpByteBuf.template Get<uint8_t>();
    LocalTensor<uint32_t> signBitsLt = fp8SignUint32Buf.template Get<uint32_t>();
    LocalTensor<uint32_t> dstBitsLt = dst.ReinterpretCast<uint32_t>();

    Duplicate<uint8_t>(maskLt, FP8_SIGN_MASK, elementCount);
    And(tmpU8Lt, src, maskLt, elementCount);
    Cast(signBitsLt, tmpU8Lt, RoundMode::CAST_NONE, elementCount);
    ShiftLeft(signBitsLt, signBitsLt, FP8_SIGN_SHIFT, elementCount);

    Duplicate<uint8_t>(maskLt, FP8_BODY_MASK, elementCount);
    And(tmpU8Lt, src, maskLt, elementCount);
    Cast(dstBitsLt, tmpU8Lt, RoundMode::CAST_NONE, elementCount);
    ShiftLeft(dstBitsLt, dstBitsLt, fp8BodyShift, elementCount);
    LocalTensor<float> dstFloatLt = dstBitsLt.ReinterpretCast<float>();
    Muls(dstFloatLt, dstFloatLt, fp8Multiplier, elementCount);
    Or(dstBitsLt, dstBitsLt, signBitsLt, elementCount);
}

namespace IntNBitSplitEmbeddingSimt {

__simt_vf__ __aicore__ LAUNCH_BOUND(INT_DEQUANT_SIMT_MAX_THREADS) inline void Int8DequantRowsSimt(
    __local_mem__ uint8_t* input, __local_mem__ float* output, int32_t rowsThisBatch, int32_t embedDim,
    int32_t alignedInputRowBytes, int32_t decodedFloatRowElems)
{
    const int32_t wordsPerRow = alignedInputRowBytes / U32B;
    for (int32_t globalWordIdx = threadIdx.x; globalWordIdx < rowsThisBatch * wordsPerRow;
         globalWordIdx += blockDim.x) {
        const int32_t rowIdx = globalWordIdx / wordsPerRow;
        const int32_t wordIdx = globalWordIdx % wordsPerRow;

        __local_mem__ uint8_t* rowBytes = input + rowIdx * alignedInputRowBytes;
        __local_mem__ half* scaleBias = reinterpret_cast<__local_mem__ half*>(rowBytes);
        const float scale = static_cast<float>(scaleBias[0]);
        const float bias = static_cast<float>(scaleBias[1]);

        if (wordIdx == 0) {
            continue;
        }
        const uint32_t packedVals = reinterpret_cast<__local_mem__ uint32_t*>(rowBytes)[wordIdx];
        const int32_t outputBase = wordIdx * I8_N - I8_PAD;
        if (outputBase >= embedDim) {
            continue;
        }

        const float value0 = static_cast<float>(packedVals & I8_M) * scale + bias;
        const float value1 = static_cast<float>((packedVals >> I8_S8) & I8_M) * scale + bias;
        const float value2 = static_cast<float>((packedVals >> I8_S16) & I8_M) * scale + bias;
        const float value3 = static_cast<float>((packedVals >> I8_S24) & I8_M) * scale + bias;

        __local_mem__ float* outputRow = output + rowIdx * decodedFloatRowElems;
        if (outputBase >= 0 && outputBase + I8_N <= embedDim) {
            reinterpret_cast<__local_mem__ float4*>(outputRow)[outputBase / I8_N] =
                make_float4(value0, value1, value2, value3);
            continue;
        }

        const int32_t tailCount = embedDim - outputBase;
        if (tailCount == 3) {
            reinterpret_cast<__local_mem__ float3*>(outputRow + outputBase)[0] = make_float3(value0, value1, value2);
        } else if (tailCount == 2) {
            reinterpret_cast<__local_mem__ float2*>(outputRow + outputBase)[0] = make_float2(value0, value1);
        } else {
            outputRow[outputBase] = value0;
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(INT_DEQUANT_SIMT_MAX_THREADS) inline void Int4DequantRowsSimt(
    __local_mem__ uint8_t* input, __local_mem__ float* output, int32_t rowsThisBatch, int32_t embedDim,
    int32_t alignedInputRowBytes, int32_t decodedFloatRowElems)
{
    const int32_t wordsPerRow = alignedInputRowBytes / U32B;
    for (int32_t globalWordIdx = threadIdx.x; globalWordIdx < rowsThisBatch * wordsPerRow;
         globalWordIdx += blockDim.x) {
        const int32_t rowIdx = globalWordIdx / wordsPerRow;
        const int32_t wordIdx = globalWordIdx % wordsPerRow;

        __local_mem__ uint8_t* rowBytes = input + rowIdx * alignedInputRowBytes;
        __local_mem__ half* scaleBias = reinterpret_cast<__local_mem__ half*>(rowBytes);
        const float scale = static_cast<float>(scaleBias[0]);
        const float bias = static_cast<float>(scaleBias[1]);

        if (wordIdx == 0) {
            continue;
        }
        const uint32_t packedVals = reinterpret_cast<__local_mem__ uint32_t*>(rowBytes)[wordIdx];
        const int32_t outputBase = wordIdx * I4_N - I4_PAD;
        if (outputBase >= embedDim) {
            continue;
        }

        const float value0 = static_cast<float>(packedVals & I4_M) * scale + bias;
        const float value1 = static_cast<float>((packedVals >> I4_S4) & I4_M) * scale + bias;
        const float value2 = static_cast<float>((packedVals >> I4_S8) & I4_M) * scale + bias;
        const float value3 = static_cast<float>((packedVals >> I4_S12) & I4_M) * scale + bias;
        const float value4 = static_cast<float>((packedVals >> I4_S16) & I4_M) * scale + bias;
        const float value5 = static_cast<float>((packedVals >> I4_S20) & I4_M) * scale + bias;
        const float value6 = static_cast<float>((packedVals >> I4_S24) & I4_M) * scale + bias;
        const float value7 = static_cast<float>((packedVals >> I4_S28) & I4_M) * scale + bias;

        __local_mem__ float* outputRow = output + rowIdx * decodedFloatRowElems;
        const int32_t validCount = embedDim - outputBase;
        if (outputBase >= 0 && validCount >= I4_N) {
            reinterpret_cast<__local_mem__ float4*>(outputRow)[outputBase / I8_N] =
                make_float4(value0, value1, value2, value3);
            reinterpret_cast<__local_mem__ float4*>(outputRow)[outputBase / I8_N + 1] =
                make_float4(value4, value5, value6, value7);
            continue;
        }

        if (validCount > I8_N) {
            reinterpret_cast<__local_mem__ float4*>(outputRow)[outputBase / I8_N] =
                make_float4(value0, value1, value2, value3);
            __local_mem__ float* tailRow = outputRow + outputBase + I8_N;
            const int32_t tailCount = validCount - I8_N;
            if (tailCount == 3) {
                reinterpret_cast<__local_mem__ float3*>(tailRow)[0] = make_float3(value4, value5, value6);
            } else if (tailCount == 2) {
                reinterpret_cast<__local_mem__ float2*>(tailRow)[0] = make_float2(value4, value5);
            } else {
                tailRow[0] = value4;
            }
            continue;
        }

        __local_mem__ float* tailRow = outputRow + outputBase;
        if (validCount >= I8_N) {
            reinterpret_cast<__local_mem__ float4*>(tailRow)[0] = make_float4(value0, value1, value2, value3);
        } else if (validCount == 3) {
            reinterpret_cast<__local_mem__ float3*>(tailRow)[0] = make_float3(value0, value1, value2);
        } else if (validCount == 2) {
            reinterpret_cast<__local_mem__ float2*>(tailRow)[0] = make_float2(value0, value1);
        } else {
            tailRow[0] = value0;
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(INT_DEQUANT_SIMT_MAX_THREADS) inline void Int2DequantRowsSimt(
    __local_mem__ uint8_t* input, __local_mem__ float* output, int32_t rowsThisBatch, int32_t embedDim,
    int32_t alignedInputRowBytes, int32_t decodedFloatRowElems)
{
    const int32_t wordsPerRow = alignedInputRowBytes / U32B;
    for (int32_t globalWordIdx = threadIdx.x; globalWordIdx < rowsThisBatch * wordsPerRow;
         globalWordIdx += blockDim.x) {
        const int32_t rowIdx = globalWordIdx / wordsPerRow;
        const int32_t wordIdx = globalWordIdx % wordsPerRow;

        __local_mem__ uint8_t* rowBytes = input + rowIdx * alignedInputRowBytes;
        __local_mem__ half* scaleBias = reinterpret_cast<__local_mem__ half*>(rowBytes);
        const float scale = static_cast<float>(scaleBias[0]);
        const float bias = static_cast<float>(scaleBias[1]);

        if (wordIdx == 0) {
            continue;
        }
        const uint32_t packedVals = reinterpret_cast<__local_mem__ uint32_t*>(rowBytes)[wordIdx];
        const int32_t outputBase = wordIdx * I2_N - I2_PAD;
        if (outputBase >= embedDim) {
            continue;
        }

        const float value0 = static_cast<float>(packedVals & I2_M) * scale + bias;
        const float value1 = static_cast<float>((packedVals >> I2_S2) & I2_M) * scale + bias;
        const float value2 = static_cast<float>((packedVals >> I2_S4) & I2_M) * scale + bias;
        const float value3 = static_cast<float>((packedVals >> I2_S6) & I2_M) * scale + bias;
        const float value4 = static_cast<float>((packedVals >> I2_S8) & I2_M) * scale + bias;
        const float value5 = static_cast<float>((packedVals >> I2_S10) & I2_M) * scale + bias;
        const float value6 = static_cast<float>((packedVals >> I2_S12) & I2_M) * scale + bias;
        const float value7 = static_cast<float>((packedVals >> I2_S14) & I2_M) * scale + bias;
        const float value8 = static_cast<float>((packedVals >> I2_S16) & I2_M) * scale + bias;
        const float value9 = static_cast<float>((packedVals >> I2_S18) & I2_M) * scale + bias;
        const float value10 = static_cast<float>((packedVals >> I2_S20) & I2_M) * scale + bias;
        const float value11 = static_cast<float>((packedVals >> I2_S22) & I2_M) * scale + bias;
        const float value12 = static_cast<float>((packedVals >> I2_S24) & I2_M) * scale + bias;
        const float value13 = static_cast<float>((packedVals >> I2_S26) & I2_M) * scale + bias;
        const float value14 = static_cast<float>((packedVals >> I2_S28) & I2_M) * scale + bias;
        const float value15 = static_cast<float>((packedVals >> I2_S30) & I2_M) * scale + bias;

        __local_mem__ float* outputRow = output + rowIdx * decodedFloatRowElems;
        const int32_t validCount = embedDim - outputBase;
        if (outputBase >= 0 && validCount >= I2_N) {
            reinterpret_cast<__local_mem__ float4*>(outputRow)[outputBase / I8_N] =
                make_float4(value0, value1, value2, value3);
            reinterpret_cast<__local_mem__ float4*>(outputRow)[outputBase / I8_N + 1] =
                make_float4(value4, value5, value6, value7);
            reinterpret_cast<__local_mem__ float4*>(outputRow)[outputBase / I8_N + 2] =
                make_float4(value8, value9, value10, value11);
            reinterpret_cast<__local_mem__ float4*>(outputRow)[outputBase / I8_N + 3] =
                make_float4(value12, value13, value14, value15);
            continue;
        }

        if (validCount > 3 * I8_N) {
            reinterpret_cast<__local_mem__ float4*>(outputRow)[outputBase / I8_N] =
                make_float4(value0, value1, value2, value3);
            reinterpret_cast<__local_mem__ float4*>(outputRow)[outputBase / I8_N + 1] =
                make_float4(value4, value5, value6, value7);
            reinterpret_cast<__local_mem__ float4*>(outputRow)[outputBase / I8_N + 2] =
                make_float4(value8, value9, value10, value11);
            __local_mem__ float* tailRow = outputRow + outputBase + 3 * I8_N;
            const int32_t tailCount = validCount - 3 * I8_N;
            if (tailCount >= I8_N) {
                reinterpret_cast<__local_mem__ float4*>(tailRow)[0] = make_float4(value12, value13, value14, value15);
            } else if (tailCount == 3) {
                reinterpret_cast<__local_mem__ float3*>(tailRow)[0] = make_float3(value12, value13, value14);
            } else if (tailCount == 2) {
                reinterpret_cast<__local_mem__ float2*>(tailRow)[0] = make_float2(value12, value13);
            } else {
                tailRow[0] = value12;
            }
            continue;
        }

        if (validCount > 2 * I8_N) {
            reinterpret_cast<__local_mem__ float4*>(outputRow)[outputBase / I8_N] =
                make_float4(value0, value1, value2, value3);
            reinterpret_cast<__local_mem__ float4*>(outputRow)[outputBase / I8_N + 1] =
                make_float4(value4, value5, value6, value7);
            __local_mem__ float* tailRow = outputRow + outputBase + 2 * I8_N;
            const int32_t tailCount = validCount - 2 * I8_N;
            if (tailCount >= I8_N) {
                reinterpret_cast<__local_mem__ float4*>(tailRow)[0] = make_float4(value8, value9, value10, value11);
            } else if (tailCount == 3) {
                reinterpret_cast<__local_mem__ float3*>(tailRow)[0] = make_float3(value8, value9, value10);
            } else if (tailCount == 2) {
                reinterpret_cast<__local_mem__ float2*>(tailRow)[0] = make_float2(value8, value9);
            } else {
                tailRow[0] = value8;
            }
            continue;
        }

        if (validCount > I8_N) {
            reinterpret_cast<__local_mem__ float4*>(outputRow)[outputBase / I8_N] =
                make_float4(value0, value1, value2, value3);
            __local_mem__ float* tailRow = outputRow + outputBase + I8_N;
            const int32_t tailCount = validCount - I8_N;
            if (tailCount >= I8_N) {
                reinterpret_cast<__local_mem__ float4*>(tailRow)[0] = make_float4(value4, value5, value6, value7);
            } else if (tailCount == 3) {
                reinterpret_cast<__local_mem__ float3*>(tailRow)[0] = make_float3(value4, value5, value6);
            } else if (tailCount == 2) {
                reinterpret_cast<__local_mem__ float2*>(tailRow)[0] = make_float2(value4, value5);
            } else {
                tailRow[0] = value4;
            }
            continue;
        }

        __local_mem__ float* tailRow = outputRow + outputBase;
        if (validCount >= I8_N) {
            reinterpret_cast<__local_mem__ float4*>(tailRow)[0] = make_float4(value0, value1, value2, value3);
        } else if (validCount == 3) {
            reinterpret_cast<__local_mem__ float3*>(tailRow)[0] = make_float3(value0, value1, value2);
        } else if (validCount == 2) {
            reinterpret_cast<__local_mem__ float2*>(tailRow)[0] = make_float2(value0, value1);
        } else {
            tailRow[0] = value0;
        }
    }
}

}  // namespace IntNBitSplitEmbeddingSimt

#endif  // INT_NBIT_SPLIT_EMBEDDING_COMMON_H
