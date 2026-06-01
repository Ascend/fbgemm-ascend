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

#ifndef INT_NBIT_SPLIT_EMBEDDING_POOLING_KERNEL_H
#define INT_NBIT_SPLIT_EMBEDDING_POOLING_KERNEL_H

#include <algorithm>
#include <type_traits>
#include "common.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace IntNBitSplitEmbeddingPooling {

// IndexType: int32_t 或 int64_t，表示 indices 和 offsets 的数据类型
// OutputType: float, half, bfloat16_t, uint8_t，表示输出数据类型
template <typename IndexType, typename OutputType>
class IntNBitSplitEmbeddingPoolingKernel {
public:
    __aicore__ inline IntNBitSplitEmbeddingPoolingKernel(Args& args)
    {
        GET_TILING_DATA(tilingData, args.tiling);

        // 数据准备：初始化所有参数
        InitShapeParams(tilingData);
        InitTiling(tilingData);
        InitGmParams(args);
        InitUbParams(tilingData);
    }

    __aicore__ inline void Compute()
    {
        int64_t maxDecodedFloatRowBytes = GetDecodedFloatRowBytes(maxD, rowAlignment, SparseType::FP8);
        indicesNumOneBlock = queInBytes / maxDecodedFloatRowBytes;
        if (indicesNumOneBlock >= MAX_INDICES_ONE_BLOCK) {
            indicesNumOneBlock = MAX_INDICES_ONE_BLOCK;
        }

        // 计算当前core的任务范围
        int64_t tableIndex = offsetOfThisCore / batchs;
        int64_t batchIndex = offsetOfThisCore % batchs;
        int64_t thisOffsetIndex = tableIndex * batchs + batchIndex;
        int64_t startIndices = static_cast<int64_t>(offsetGT.GetValue(thisOffsetIndex));

        // 遍历当前core的所有bag任务
        for (int64_t loop = 0; loop < lenOfThisCore; loop++) {
            int64_t endIndices = static_cast<int64_t>(offsetGT.GetValue(thisOffsetIndex + 1));
            int32_t thisLen = endIndices - startIndices;

            if (thisLen <= 0) {
                startIndices = endIndices;
                thisOffsetIndex++;
                continue;
            }

            // 获取当前bag的参数
            tableIndex = thisOffsetIndex / batchs;
            int64_t thisWeightOffset = weightOffsetGT.GetValue(tableIndex);
            int64_t outBatchInd = thisOffsetIndex % outDim0;
            int64_t outEmbedOffset = dOffsetGT.GetValue(tableIndex);
            int64_t outOffset = outBatchInd * outDim1 + outEmbedOffset;
            if constexpr (std::is_same_v<OutputType, uint8_t>) {
                outOffset += tableIndex * INT8_QPARAMS_BYTES;
            }
            int64_t embedDim = dOffsetGT.GetValue(tableIndex + 1) - outEmbedOffset;

            // 处理当前bag（可能分批处理）
            Process(thisLen, startIndices, embedDim, thisWeightOffset, outOffset, tableIndex, isWeighted);

            startIndices = endIndices;
            thisOffsetIndex++;
        }
    }

private:
    // ========== 数据准备：初始化参数 ==========

    __aicore__ inline void InitShapeParams(const IntNbitSplitEmbeddingCodegenLookupFunctionTilingData& tilingData)
    {
        devWeightsDim0 = tilingData.devWeightsDim0;
        weightsOffsetsDim0 = tilingData.weightsOffsetsDim0;
        dOffsetsDim0 = tilingData.dOffsetsDim0;
        indicesDim0 = tilingData.indicesDim0;
        offsetsDim0 = tilingData.offsetsDim0;
        outDim0 = tilingData.outDim0;
        outDim1 = tilingData.outDim1;
        maxD = tilingData.maxD;
        batchs = (offsetsDim0 - 1) / weightsOffsetsDim0;

        poolMode = tilingData.poolMode;

        // FP8相关参数
        fp8ExponentBits = tilingData.fp8ExponentBits;
        fp8ExponentBias = tilingData.fp8ExponentBias;
        rowAlignment = tilingData.rowAlignment;

        // Weighted模式标志
        isWeighted = tilingData.isWeighted;

        // 输出数据类型
        outputDtype = tilingData.outputDtype;

        // 初始化FP8参数（使用公共函数）
        InitFp8Params(fp8ExponentBits, fp8ExponentBias, fp8BodyShift, fp8Multiplier);
    }

    __aicore__ inline void InitTiling(const IntNbitSplitEmbeddingCodegenLookupFunctionTilingData& tilingData)
    {
        splitBaseLen = tilingData.splitBaseLen;
        tailSplitIndex = tilingData.tailSplitIndex;

        if (GetBlockIdx() >= tailSplitIndex) {
            lenOfThisCore = splitBaseLen;
            offsetOfThisCore = tailSplitIndex * (splitBaseLen + 1) + (GetBlockIdx() - tailSplitIndex) * splitBaseLen;
        } else {
            lenOfThisCore = splitBaseLen + 1;
            offsetOfThisCore = GetBlockIdx() * (splitBaseLen + 1);
        }
    }

    __aicore__ inline void InitGmParams(const Args& args)
    {
        // 权重使用uint8_t类型（量化数据）
        devWeightsGT.SetGlobalBuffer((__gm__ uint8_t*)args.devWeights, devWeightsDim0);
        indicesGT.SetGlobalBuffer((__gm__ IndexType*)args.indices, indicesDim0);
        offsetGT.SetGlobalBuffer((__gm__ IndexType*)args.offsets, offsetsDim0);
        dOffsetGT.SetGlobalBuffer((__gm__ int32_t*)args.dOffsets, dOffsetsDim0);
        weightOffsetGT.SetGlobalBuffer((__gm__ int64_t*)args.weightsOffsets, weightsOffsetsDim0);
        weightsTysGT.SetGlobalBuffer((__gm__ uint8_t*)args.weightsTys, weightsOffsetsDim0);
        outGT.SetGlobalBuffer((__gm__ OutputType*)args.out, outDim0 * outDim1);

        // 条件初始化：只有当isWeighted为true时才初始化
        if (isWeighted) {
            indiceWeightsGT.SetGlobalBuffer((__gm__ float*)args.indiceWeights, indicesDim0);
        }

        ASCENDC_ASSERT(static_cast<int64_t>(offsetGT.GetValue(offsetsDim0 - 1)) == indicesDim0,
                       "The last element in offsets must be equal to indices size");
    }

    __aicore__ inline void InitUbParams(const IntNbitSplitEmbeddingCodegenLookupFunctionTilingData& tilingData)
    {
        ubCanUsed = tilingData.ubCanUsed;

        // 步骤1：先分配queIndices（固定大小，类型与indices/offsets一致）
        int64_t queIndicesBytes = MAX_INDICES_ONE_BLOCK * sizeof(IndexType);
        int64_t remainingUb1 = ubCanUsed - queIndicesBytes;

        // 如果支持weighted，需要分配queWeights（存储indice_weights）
        int64_t queWeightsBytes = 0;
        if (isWeighted) {
            queWeightsBytes = MAX_INDICES_ONE_BLOCK * sizeof(float);
            remainingUb1 = remainingUb1 - queWeightsBytes;
        }

        // 步骤2：分配queOut（只需要一个embedding向量）
        int64_t queOutBytes = GetAlignedFloatRowBytes(maxD);
        int64_t remainingUb2 = remainingUb1 - queOutBytes;

        // 如果OutputType不是float，需要额外分配newqueOut（OutputType类型，用于输出）
        int64_t newqueOutBytes = 0;
        if constexpr (!std::is_same_v<OutputType, float>) {
            if constexpr (std::is_same_v<OutputType, uint8_t>) {
                newqueOutBytes = AlignTo32(maxD + INT8_QPARAMS_BYTES);
            } else {
                newqueOutBytes = AlignTo32(maxD * sizeof(OutputType));
            }
            remainingUb2 = remainingUb2 - newqueOutBytes;
        }

        // 步骤3：计算tmpQue和queIn的内存分配
        int64_t maxFp8InputRowBytes = GetAlignedInputRowBytes(maxD, rowAlignment, SparseType::FP8);
        int64_t maxFp16InputRowBytes = GetAlignedInputRowBytes(maxD, rowAlignment, SparseType::FP16);
        int64_t maxInt8InputRowBytes = GetAlignedInputRowBytes(maxD, rowAlignment, SparseType::INT8);
        int64_t maxInt4InputRowBytes = GetAlignedInputRowBytes(maxD, rowAlignment, SparseType::INT4);
        int64_t maxInt2InputRowBytes = GetAlignedInputRowBytes(maxD, rowAlignment, SparseType::INT2);
        int64_t maxTmpInputRowBytes =
            Std::max(Std::max(maxFp8InputRowBytes, maxFp16InputRowBytes),
                     Std::max(maxInt8InputRowBytes, Std::max(maxInt4InputRowBytes, maxInt2InputRowBytes)));
        int64_t maxFp8DecodedRowBytes = GetDecodedFloatRowBytes(maxD, rowAlignment, SparseType::FP8);
        int64_t maxFp16DecodedRowBytes = GetDecodedFloatRowBytes(maxD, rowAlignment, SparseType::FP16);
        int64_t maxFp32DecodedRowBytes = GetDecodedFloatRowBytes(maxD, rowAlignment, SparseType::FP32);
        int64_t maxInt8DecodedRowBytes = GetDecodedFloatRowBytes(maxD, rowAlignment, SparseType::INT8);
        int64_t maxInt4DecodedRowBytes = GetDecodedFloatRowBytes(maxD, rowAlignment, SparseType::INT4);
        int64_t maxInt2DecodedRowBytes = GetDecodedFloatRowBytes(maxD, rowAlignment, SparseType::INT2);
        int64_t maxDecodedRowBytes = Std::max(
            Std::max(maxFp8DecodedRowBytes, maxFp16DecodedRowBytes),
            Std::max(maxFp32DecodedRowBytes,
                     Std::max(maxInt8DecodedRowBytes, Std::max(maxInt4DecodedRowBytes, maxInt2DecodedRowBytes))));
        int64_t maxScratchInputRowBytes = Std::max(
            maxFp8InputRowBytes, Std::max(maxInt8InputRowBytes, Std::max(maxInt4InputRowBytes, maxInt2InputRowBytes)));

        int64_t bytesPerIndex = maxDecodedRowBytes + maxTmpInputRowBytes + maxScratchInputRowBytes * 6;
        int64_t maxN = remainingUb2 / bytesPerIndex;
        int64_t indicesNumOneBlockCalc = Std::min(maxN, MAX_INDICES_ONE_BLOCK);

        int64_t fp8TmpByteBytes = 0;
        int64_t fp8MaskBytes = 0;
        int64_t minInt8QuantTmpBytes = 0;
        int64_t minInt8QuantMaskBytes = 0;
        int64_t minInt8QuantSignBytes = 0;
        if constexpr (std::is_same_v<OutputType, uint8_t>) {
            // INT8输出在CopyOut阶段会对整条FP32结果做rowwise quantize，
            // 这里的临时buffer至少要能容纳一整行float结果及中间转换数据。
            minInt8QuantTmpBytes = GetAlignedFloatRowBytes(maxD);
            minInt8QuantMaskBytes = GetAlignedFloatRowBytes(maxD);
            minInt8QuantSignBytes = AlignTo32(maxD * sizeof(int16_t));
        }

        while (true) {
            tmpQueBytes = indicesNumOneBlockCalc * maxTmpInputRowBytes;
            queInBytes = indicesNumOneBlockCalc * maxDecodedRowBytes;
            fp8TmpByteBytes = indicesNumOneBlockCalc * maxScratchInputRowBytes;
            fp8MaskBytes = indicesNumOneBlockCalc * maxScratchInputRowBytes;
            fp8SignUint32Bytes = indicesNumOneBlockCalc * maxScratchInputRowBytes * sizeof(uint32_t);

            if constexpr (std::is_same_v<OutputType, uint8_t>) {
                fp8TmpByteBytes = Std::max(fp8TmpByteBytes, minInt8QuantTmpBytes);
                fp8MaskBytes = Std::max(fp8MaskBytes, minInt8QuantMaskBytes);
                fp8SignUint32Bytes = Std::max(fp8SignUint32Bytes, minInt8QuantSignBytes);
            }

            int64_t totalBytes = tmpQueBytes + queInBytes + fp8TmpByteBytes + fp8MaskBytes + fp8SignUint32Bytes;
            if (totalBytes <= remainingUb2 || indicesNumOneBlockCalc <= 1) {
                break;
            }
            indicesNumOneBlockCalc--;
        }

        // 步骤4：分配内存
        pipe.InitBuffer(queIndices, 1, queIndicesBytes);
        pipe.InitBuffer(queOut, 1, queOutBytes);
        if constexpr (!std::is_same_v<OutputType, float>) {
            pipe.InitBuffer(newqueOut, 1, newqueOutBytes);
        }
        pipe.InitBuffer(tmpQue, 1, tmpQueBytes);
        pipe.InitBuffer(queIn, 1, queInBytes);
        pipe.InitBuffer(fp8MaskBuf, AlignTo32(fp8MaskBytes));
        pipe.InitBuffer(fp8TmpByteBuf, AlignTo32(fp8TmpByteBytes));
        pipe.InitBuffer(fp8SignUint32Buf, AlignTo32(fp8SignUint32Bytes));
        if (isWeighted) {
            pipe.InitBuffer(queWeights, 1, queWeightsBytes);
        }
    }

    // ========== CopyIn：读取权重并反量化 ==========

    __aicore__ inline void CopyInWithDequantize(int64_t startIndices, int64_t thisLen, int64_t embedDim,
                                                int64_t thisWeightOffset, int64_t tableIndex)
    {
        // 获取当前表的量化类型
        uint8_t weightType = weightsTysGT.GetValue(tableIndex);
        if (weightType == static_cast<uint8_t>(SparseType::FP8)) {
            CopyInFP8(startIndices, thisLen, embedDim, thisWeightOffset);
        } else if (weightType == static_cast<uint8_t>(SparseType::FP16)) {
            CopyInFP16(startIndices, thisLen, embedDim, thisWeightOffset);
        } else if (weightType == static_cast<uint8_t>(SparseType::FP32)) {
            CopyInFP32(startIndices, thisLen, embedDim, thisWeightOffset);
        } else if (weightType == static_cast<uint8_t>(SparseType::INT8)) {
            CopyInINT8(startIndices, thisLen, embedDim, thisWeightOffset);
        } else if (weightType == static_cast<uint8_t>(SparseType::INT4)) {
            CopyInINT4(startIndices, thisLen, embedDim, thisWeightOffset);
        } else if (weightType == static_cast<uint8_t>(SparseType::INT2)) {
            CopyInINT2(startIndices, thisLen, embedDim, thisWeightOffset);
        } else {
            ascendc_assert(false, "Unsupported weight type: %x.\n", weightType);
        }
    }

    __aicore__ inline void CopyInFP8(int64_t startIndices, int64_t thisLen, int64_t embedDim, int64_t thisWeightOffset)
    {
        int64_t rowBytes = GetInputRowBytes(embedDim, rowAlignment, SparseType::FP8);
        int64_t alignedInputRowBytes = AlignTo32(rowBytes);
        int64_t decodedFloatRowElems = GetDecodedFloatRowElems(embedDim, rowAlignment, SparseType::FP8);

        // 步骤1：先读取indices到queIndices
        LocalTensor<IndexType> indicesLt = queIndices.AllocTensor<IndexType>();
        CpGm2Local(indicesLt, indicesGT[startIndices], thisLen);
        queIndices.EnQue(indicesLt);
        indicesLt = queIndices.DeQue<IndexType>();

        // 如果是weighted模式，读取indiceWeights到queWeights
        if (isWeighted) {
            LocalTensor<float> weightsLt = queWeights.AllocTensor<float>();
            CpGm2Local(weightsLt, indiceWeightsGT[startIndices], thisLen);
            queWeights.EnQue(weightsLt);
        }

        // 步骤2：DataCopy量化数据到tmpQue（每行对齐，从GM读取rowBytes字节）
        LocalTensor<uint8_t> quantizedLt = tmpQue.AllocTensor<uint8_t>();
        Duplicate<uint8_t>(quantizedLt, 0, thisLen * alignedInputRowBytes);
        for (int64_t i = 0; i < thisLen; ++i) {
            int64_t thisInd = static_cast<int64_t>(indicesLt.GetValue(i));
            if (thisInd < 0) {
                continue;
            }
            int64_t indWeightOffset = thisInd * rowBytes + thisWeightOffset;
            // 从GM读取rowBytes字节（实际存储长度），写入UB的alignedInputRowBytes对齐位置
            CpGm2Local(quantizedLt[i * alignedInputRowBytes], devWeightsGT[indWeightOffset], rowBytes);
        }
        tmpQue.EnQue(quantizedLt);
        quantizedLt = tmpQue.DeQue<uint8_t>();

        // 步骤3：手写FP8 → FP32转换
        LocalTensor<float> floatLt = queIn.AllocTensor<float>();
        int64_t totalElements = thisLen * decodedFloatRowElems;
        FP8U8ToFP32Bitpack(quantizedLt, floatLt, totalElements, fp8BodyShift, fp8Multiplier, fp8MaskBuf, fp8TmpByteBuf,
                           fp8SignUint32Buf);
        for (int64_t i = 0; i < thisLen; ++i) {
            if (static_cast<int64_t>(indicesLt.GetValue(i)) < 0) {
                Duplicate<float>(floatLt[i * decodedFloatRowElems], 0.0f, decodedFloatRowElems);
            }
        }

        queIn.EnQue(floatLt);
        tmpQue.FreeTensor(quantizedLt);
        queIndices.FreeTensor(indicesLt);
    }

    __aicore__ inline void CopyInFP16(int64_t startIndices, int64_t thisLen, int64_t embedDim, int64_t thisWeightOffset)
    {
        int64_t rowBytes = GetInputRowBytes(embedDim, rowAlignment, SparseType::FP16);
        int64_t alignedInputRowBytes = AlignTo32(rowBytes);
        int64_t decodedFloatRowElems = GetDecodedFloatRowElems(embedDim, rowAlignment, SparseType::FP16);
        int64_t gmHalfRowElems = rowBytes / static_cast<int64_t>(sizeof(half));
        int64_t alignedHalfRowElems = alignedInputRowBytes / static_cast<int64_t>(sizeof(half));

        LocalTensor<IndexType> indicesLt = queIndices.AllocTensor<IndexType>();
        CpGm2Local(indicesLt, indicesGT[startIndices], thisLen);
        queIndices.EnQue(indicesLt);
        indicesLt = queIndices.DeQue<IndexType>();

        if (isWeighted) {
            LocalTensor<float> weightsLt = queWeights.AllocTensor<float>();
            CpGm2Local(weightsLt, indiceWeightsGT[startIndices], thisLen);
            queWeights.EnQue(weightsLt);
        }

        LocalTensor<uint8_t> rawLt = tmpQue.AllocTensor<uint8_t>();
        Duplicate<uint8_t>(rawLt, 0, thisLen * alignedInputRowBytes);
        for (int64_t i = 0; i < thisLen; ++i) {
            int64_t thisInd = static_cast<int64_t>(indicesLt.GetValue(i));
            if (thisInd < 0) {
                continue;
            }
            int64_t indWeightOffset = thisInd * rowBytes + thisWeightOffset;
            CpGm2Local(rawLt[i * alignedInputRowBytes], devWeightsGT[indWeightOffset], rowBytes);
        }
        tmpQue.EnQue(rawLt);
        rawLt = tmpQue.DeQue<uint8_t>();

        LocalTensor<float> floatLt = queIn.AllocTensor<float>();
        LocalTensor<half> halfLt = rawLt.ReinterpretCast<half>();
        constexpr int64_t CAST_REPEAT_LIMIT = 255;
        int64_t dstRepStride64 = (decodedFloatRowElems * sizeof(float)) / DATA_ALIGN_BYTES;
        int64_t srcRepStride64 = (alignedHalfRowElems * sizeof(half)) / DATA_ALIGN_BYTES;
        if (gmHalfRowElems <= 64 && thisLen <= CAST_REPEAT_LIMIT && dstRepStride64 <= 255 && srcRepStride64 <= 255) {
            Cast(floatLt, halfLt, RoundMode::CAST_NONE, static_cast<uint64_t>(gmHalfRowElems),
                 static_cast<uint8_t>(thisLen),
                 UnaryRepeatParams{1, 1, static_cast<uint8_t>(dstRepStride64), static_cast<uint8_t>(srcRepStride64)});
        } else if (dstRepStride64 <= 255 && srcRepStride64 <= 255) {
            uint8_t dstRepStride = static_cast<uint8_t>(dstRepStride64);
            uint8_t srcRepStride = static_cast<uint8_t>(srcRepStride64);
            for (int64_t rowBase = 0; rowBase < thisLen; rowBase += CAST_REPEAT_LIMIT) {
                uint8_t repeatTime = static_cast<uint8_t>(Std::min<int64_t>(CAST_REPEAT_LIMIT, thisLen - rowBase));
                int64_t rowElemOffsetDst = rowBase * decodedFloatRowElems;
                int64_t rowElemOffsetSrc = rowBase * alignedHalfRowElems;
                for (int64_t colBase = 0; colBase < gmHalfRowElems; colBase += 64) {
                    uint64_t mask = static_cast<uint64_t>(Std::min<int64_t>(64, gmHalfRowElems - colBase));
                    LocalTensor<float> floatTileLt = floatLt[rowElemOffsetDst + colBase];
                    LocalTensor<half> halfTileLt = halfLt[rowElemOffsetSrc + colBase];
                    Cast(floatTileLt, halfTileLt, RoundMode::CAST_NONE, mask, repeatTime,
                         UnaryRepeatParams{1, 1, dstRepStride, srcRepStride});
                }
            }
        } else {
            for (int64_t i = 0; i < thisLen; ++i) {
                LocalTensor<float> floatRowLt = floatLt[i * decodedFloatRowElems];
                LocalTensor<half> halfRowLt = halfLt[i * alignedHalfRowElems];
                Cast(floatRowLt, halfRowLt, RoundMode::CAST_NONE, gmHalfRowElems);
            }
        }
        for (int64_t i = 0; i < thisLen; ++i) {
            if (static_cast<int64_t>(indicesLt.GetValue(i)) < 0) {
                Duplicate<float>(floatLt[i * decodedFloatRowElems], 0.0f, decodedFloatRowElems);
            }
        }

        queIn.EnQue(floatLt);
        tmpQue.FreeTensor(rawLt);
        queIndices.FreeTensor(indicesLt);
    }

    __aicore__ inline void CopyInFP32(int64_t startIndices, int64_t thisLen, int64_t embedDim, int64_t thisWeightOffset)
    {
        int64_t rowBytes = GetInputRowBytes(embedDim, rowAlignment, SparseType::FP32);
        int64_t decodedFloatRowElems = GetDecodedFloatRowElems(embedDim, rowAlignment, SparseType::FP32);
        int64_t decodedFloatRowBytes = decodedFloatRowElems * static_cast<int64_t>(sizeof(float));

        LocalTensor<IndexType> indicesLt = queIndices.AllocTensor<IndexType>();
        CpGm2Local(indicesLt, indicesGT[startIndices], thisLen);
        queIndices.EnQue(indicesLt);
        indicesLt = queIndices.DeQue<IndexType>();

        if (isWeighted) {
            LocalTensor<float> weightsLt = queWeights.AllocTensor<float>();
            CpGm2Local(weightsLt, indiceWeightsGT[startIndices], thisLen);
            queWeights.EnQue(weightsLt);
        }

        LocalTensor<float> floatLt = queIn.AllocTensor<float>();
        LocalTensor<uint8_t> floatBytesLt = floatLt.ReinterpretCast<uint8_t>();
        for (int64_t i = 0; i < thisLen; ++i) {
            int64_t thisInd = static_cast<int64_t>(indicesLt.GetValue(i));
            if (thisInd < 0) {
                continue;
            }
            int64_t indWeightOffset = thisInd * rowBytes + thisWeightOffset;
            CpGm2Local(floatBytesLt[i * decodedFloatRowBytes], devWeightsGT[indWeightOffset], rowBytes);
        }
        for (int64_t i = 0; i < thisLen; ++i) {
            if (static_cast<int64_t>(indicesLt.GetValue(i)) < 0) {
                Duplicate<float>(floatLt[i * decodedFloatRowElems], 0.0f, decodedFloatRowElems);
            }
        }

        queIn.EnQue(floatLt);
        queIndices.FreeTensor(indicesLt);
    }

    __aicore__ inline void CopyInINT8(int64_t startIndices, int64_t thisLen, int64_t embedDim, int64_t thisWeightOffset)
    {
        int64_t rowBytes = GetInputRowBytes(embedDim, rowAlignment, SparseType::INT8);
        int64_t alignedInputRowBytes = AlignTo32(rowBytes);
        int64_t decodedFloatRowElems = GetDecodedFloatRowElems(embedDim, rowAlignment, SparseType::INT8);

        LocalTensor<IndexType> indicesLt = queIndices.AllocTensor<IndexType>();
        CpGm2Local(indicesLt, indicesGT[startIndices], thisLen);
        queIndices.EnQue(indicesLt);
        indicesLt = queIndices.DeQue<IndexType>();

        if (isWeighted) {
            LocalTensor<float> weightsLt = queWeights.AllocTensor<float>();
            CpGm2Local(weightsLt, indiceWeightsGT[startIndices], thisLen);
            queWeights.EnQue(weightsLt);
        }

        LocalTensor<uint8_t> rawLt = tmpQue.AllocTensor<uint8_t>();
        Duplicate<uint8_t>(rawLt, 0, thisLen * alignedInputRowBytes);
        for (int64_t i = 0; i < thisLen; ++i) {
            int64_t thisInd = static_cast<int64_t>(indicesLt.GetValue(i));
            if (thisInd < 0) {
                continue;
            }
            int64_t indWeightOffset = thisInd * rowBytes + thisWeightOffset;
            CpGm2Local(rawLt[i * alignedInputRowBytes], devWeightsGT[indWeightOffset], rowBytes);
        }
        tmpQue.EnQue(rawLt);
        rawLt = tmpQue.DeQue<uint8_t>();

        LocalTensor<float> floatLt = queIn.AllocTensor<float>();
        __local_mem__ uint8_t* rawPtr = reinterpret_cast<__local_mem__ uint8_t*>(rawLt.GetPhyAddr());
        __local_mem__ float* floatPtr = reinterpret_cast<__local_mem__ float*>(floatLt.GetPhyAddr());
        int32_t wordsPerRow = static_cast<int32_t>(alignedInputRowBytes / static_cast<int64_t>(sizeof(uint32_t)));
        int32_t threadNum = GetIntDequantThreadNum(thisLen * static_cast<int64_t>(wordsPerRow));
        asc_vf_call<IntNBitSplitEmbeddingSimt::Int8DequantRowsSimt>(
            dim3{static_cast<uint32_t>(threadNum), 1U, 1U}, rawPtr, floatPtr, static_cast<int32_t>(thisLen),
            static_cast<int32_t>(embedDim), static_cast<int32_t>(alignedInputRowBytes),
            static_cast<int32_t>(decodedFloatRowElems));
        for (int64_t i = 0; i < thisLen; ++i) {
            if (static_cast<int64_t>(indicesLt.GetValue(i)) < 0) {
                Duplicate<float>(floatLt[i * decodedFloatRowElems], 0.0f, decodedFloatRowElems);
            }
        }

        queIn.EnQue(floatLt);
        tmpQue.FreeTensor(rawLt);
        queIndices.FreeTensor(indicesLt);
    }

    __aicore__ inline void CopyInINT4(int64_t startIndices, int64_t thisLen, int64_t embedDim, int64_t thisWeightOffset)
    {
        int64_t rowBytes = GetInputRowBytes(embedDim, rowAlignment, SparseType::INT4);
        int64_t alignedInputRowBytes = AlignTo32(rowBytes);
        int64_t decodedFloatRowElems = GetDecodedFloatRowElems(embedDim, rowAlignment, SparseType::INT4);

        LocalTensor<IndexType> indicesLt = queIndices.AllocTensor<IndexType>();
        CpGm2Local(indicesLt, indicesGT[startIndices], thisLen);
        queIndices.EnQue(indicesLt);
        indicesLt = queIndices.DeQue<IndexType>();

        if (isWeighted) {
            LocalTensor<float> weightsLt = queWeights.AllocTensor<float>();
            CpGm2Local(weightsLt, indiceWeightsGT[startIndices], thisLen);
            queWeights.EnQue(weightsLt);
        }

        LocalTensor<uint8_t> rawLt = tmpQue.AllocTensor<uint8_t>();
        Duplicate<uint8_t>(rawLt, 0, thisLen * alignedInputRowBytes);
        for (int64_t i = 0; i < thisLen; ++i) {
            int64_t thisInd = static_cast<int64_t>(indicesLt.GetValue(i));
            if (thisInd < 0) {
                continue;
            }
            int64_t indWeightOffset = thisInd * rowBytes + thisWeightOffset;
            CpGm2Local(rawLt[i * alignedInputRowBytes], devWeightsGT[indWeightOffset], rowBytes);
        }
        tmpQue.EnQue(rawLt);
        rawLt = tmpQue.DeQue<uint8_t>();

        LocalTensor<float> floatLt = queIn.AllocTensor<float>();
        __local_mem__ uint8_t* rawPtr = reinterpret_cast<__local_mem__ uint8_t*>(rawLt.GetPhyAddr());
        __local_mem__ float* floatPtr = reinterpret_cast<__local_mem__ float*>(floatLt.GetPhyAddr());
        int32_t wordsPerRow = static_cast<int32_t>(alignedInputRowBytes / static_cast<int64_t>(sizeof(uint32_t)));
        int32_t threadNum = GetIntDequantThreadNum(thisLen * static_cast<int64_t>(wordsPerRow));
        asc_vf_call<IntNBitSplitEmbeddingSimt::Int4DequantRowsSimt>(
            dim3{static_cast<uint32_t>(threadNum), 1U, 1U}, rawPtr, floatPtr, static_cast<int32_t>(thisLen),
            static_cast<int32_t>(embedDim), static_cast<int32_t>(alignedInputRowBytes),
            static_cast<int32_t>(decodedFloatRowElems));
        for (int64_t i = 0; i < thisLen; ++i) {
            if (static_cast<int64_t>(indicesLt.GetValue(i)) < 0) {
                Duplicate<float>(floatLt[i * decodedFloatRowElems], 0.0f, decodedFloatRowElems);
            }
        }

        queIn.EnQue(floatLt);
        tmpQue.FreeTensor(rawLt);
        queIndices.FreeTensor(indicesLt);
    }

    __aicore__ inline void CopyInINT2(int64_t startIndices, int64_t thisLen, int64_t embedDim, int64_t thisWeightOffset)
    {
        int64_t rowBytes = GetInputRowBytes(embedDim, rowAlignment, SparseType::INT2);
        int64_t alignedInputRowBytes = AlignTo32(rowBytes);
        int64_t decodedFloatRowElems = GetDecodedFloatRowElems(embedDim, rowAlignment, SparseType::INT2);

        LocalTensor<IndexType> indicesLt = queIndices.AllocTensor<IndexType>();
        CpGm2Local(indicesLt, indicesGT[startIndices], thisLen);
        queIndices.EnQue(indicesLt);
        indicesLt = queIndices.DeQue<IndexType>();

        if (isWeighted) {
            LocalTensor<float> weightsLt = queWeights.AllocTensor<float>();
            CpGm2Local(weightsLt, indiceWeightsGT[startIndices], thisLen);
            queWeights.EnQue(weightsLt);
        }

        LocalTensor<uint8_t> rawLt = tmpQue.AllocTensor<uint8_t>();
        Duplicate<uint8_t>(rawLt, 0, thisLen * alignedInputRowBytes);
        for (int64_t i = 0; i < thisLen; ++i) {
            int64_t thisInd = static_cast<int64_t>(indicesLt.GetValue(i));
            if (thisInd < 0) {
                continue;
            }
            int64_t indWeightOffset = thisInd * rowBytes + thisWeightOffset;
            CpGm2Local(rawLt[i * alignedInputRowBytes], devWeightsGT[indWeightOffset], rowBytes);
        }
        tmpQue.EnQue(rawLt);
        rawLt = tmpQue.DeQue<uint8_t>();

        LocalTensor<float> floatLt = queIn.AllocTensor<float>();
        __local_mem__ uint8_t* rawPtr = reinterpret_cast<__local_mem__ uint8_t*>(rawLt.GetPhyAddr());
        __local_mem__ float* floatPtr = reinterpret_cast<__local_mem__ float*>(floatLt.GetPhyAddr());
        int32_t wordsPerRow = static_cast<int32_t>(alignedInputRowBytes / static_cast<int64_t>(sizeof(uint32_t)));
        int32_t threadNum = GetIntDequantThreadNum(thisLen * static_cast<int64_t>(wordsPerRow));
        asc_vf_call<IntNBitSplitEmbeddingSimt::Int2DequantRowsSimt>(
            dim3{static_cast<uint32_t>(threadNum), 1U, 1U}, rawPtr, floatPtr, static_cast<int32_t>(thisLen),
            static_cast<int32_t>(embedDim), static_cast<int32_t>(alignedInputRowBytes),
            static_cast<int32_t>(decodedFloatRowElems));
        for (int64_t i = 0; i < thisLen; ++i) {
            if (static_cast<int64_t>(indicesLt.GetValue(i)) < 0) {
                Duplicate<float>(floatLt[i * decodedFloatRowElems], 0.0f, decodedFloatRowElems);
            }
        }

        queIn.EnQue(floatLt);
        tmpQue.FreeTensor(rawLt);
        queIndices.FreeTensor(indicesLt);
    }

    // ========== Compute：Pooling计算（只做累加，不做mean处理） ==========

    __aicore__ inline void Pooling(LocalTensor<float>& outLt, int64_t thisLen, int64_t embedDim, SparseType weightType,
                                   bool isWeighted)
    {
        LocalTensor<float> inputLt = queIn.DeQue<float>();

        int64_t decodedFloatRowElems = GetDecodedFloatRowElems(embedDim, rowAlignment, weightType);

        if (isWeighted) {
            LocalTensor<float> weightsLt = queWeights.DeQue<float>();
            LocalTensor<float> weightVecLt = fp8SignUint32Buf.Get<float>();

            for (int64_t i = 0; i < thisLen; i++) {
                float weight = weightsLt.GetValue(i);
                Duplicate<float>(weightVecLt, weight, embedDim);
                MulAddDst(outLt, inputLt[i * decodedFloatRowElems], weightVecLt, embedDim);
            }
            queWeights.FreeTensor(weightsLt);
        } else {
            // Unweighted模式：直接累加
            for (int64_t i = 0; i < thisLen; i++) {
                Add(outLt, outLt, inputLt[i * decodedFloatRowElems], embedDim);
            }
        }

        queIn.FreeTensor(inputLt);
    }

    // ========== CopyOut：输出结果 ==========

    __aicore__ inline void CopyOut(LocalTensor<float>& outLt, int64_t outOffset, int64_t embedDim)
    {
        // 根据输出类型进行转换
        if constexpr (std::is_same_v<OutputType, float>) {
            // FP32：直接拷贝
            CpLocal2Gm(outGT[outOffset], outLt, embedDim);
        } else if constexpr (std::is_same_v<OutputType, half> || std::is_same_v<OutputType, bfloat16_t>) {
            // FP16&BF16：转换后拷贝
            LocalTensor<OutputType> outLt16 = newqueOut.AllocTensor<OutputType>();
            Cast(outLt16, outLt, RoundMode::CAST_RINT, embedDim);
            newqueOut.EnQue(outLt16);
            outLt16 = newqueOut.DeQue<OutputType>();

            CpLocal2Gm(outGT[outOffset], outLt16, embedDim);
            newqueOut.FreeTensor(outLt16);
        } else if constexpr (std::is_same_v<OutputType, uint8_t>) {
            LocalTensor<float> reduceDstLt = fp8MaskBuf.Get<float>();
            LocalTensor<float> reduceTmpLt = fp8TmpByteBuf.Get<float>();
            ReduceMin(reduceDstLt, outLt, reduceTmpLt, embedDim, false);
            float minVal = reduceDstLt.GetValue(0);
            ReduceMax(reduceDstLt, outLt, reduceTmpLt, embedDim, false);
            float maxVal = reduceDstLt.GetValue(0);

            float scale = (maxVal - minVal) / INT8_QPARAM_SCALE_DIVISOR;
            float bias = minVal;
            float invScale = INT8_QPARAM_SCALE_DIVISOR / (scale * INT8_QPARAM_SCALE_DIVISOR + INT8_QPARAM_EPS);

            Adds(outLt, outLt, -bias, embedDim);
            Muls(outLt, outLt, invScale, embedDim);
            LocalTensor<float> clampLt = fp8TmpByteBuf.Get<float>();
            Clamp(clampLt, outLt, 0.0f, INT8_QPARAM_SCALE_DIVISOR, embedDim);

            LocalTensor<uint8_t> roundTmpLt = fp8MaskBuf.Get<uint8_t>();
            Round<float>(clampLt, clampLt, roundTmpLt, embedDim);

            LocalTensor<int16_t> int16Lt = fp8SignUint32Buf.Get<int16_t>();
            Cast(int16Lt, clampLt, RoundMode::CAST_RINT, embedDim);

            LocalTensor<uint8_t> outLt8 = newqueOut.AllocTensor<uint8_t>();
            Cast(outLt8, int16Lt, RoundMode::CAST_NONE, embedDim);
            LocalTensor<float> qparamsLt = outLt8[embedDim].template ReinterpretCast<float>();
            qparamsLt.SetValue(0, scale);
            qparamsLt.SetValue(1, bias);
            newqueOut.EnQue(outLt8);
            outLt8 = newqueOut.DeQue<uint8_t>();

            CpLocal2Gm(outGT[outOffset], outLt8, embedDim + INT8_QPARAMS_BYTES);
            newqueOut.FreeTensor(outLt8);
        }
    }

    // ========== Process：处理一个bag（可能分批） ==========

    __aicore__ inline void Process(int64_t remain, int64_t startIndices, int64_t embedDim, int64_t thisWeightOffset,
                                   int64_t outOffset, int64_t tableIndex, bool isWeighted)
    {
        float meanLen = static_cast<float>(1) / static_cast<float>(remain);

        // 初始化累加缓冲区（处理整个bag）
        LocalTensor<float> outLt = queOut.AllocTensor<float>();
        Duplicate<float>(outLt, 0, embedDim);

        // 根据实际的embedDim动态调整批次大小，计算当前表的对齐长度
        uint8_t weightType = weightsTysGT.GetValue(tableIndex);
        SparseType weightTypeEnum = static_cast<SparseType>(weightType);
        int64_t alignedInputRowBytes = GetAlignedInputRowBytes(embedDim, rowAlignment, weightTypeEnum);
        int64_t decodedFloatRowBytes = GetDecodedFloatRowBytes(embedDim, rowAlignment, weightTypeEnum);

        // 计算当前表能处理的indices数量
        int64_t maxIndicesByQueIn = queInBytes / decodedFloatRowBytes;
        int64_t maxIndicesByTmpQue = indicesNumOneBlock;
        if (weightTypeEnum != SparseType::FP32) {
            maxIndicesByTmpQue = tmpQueBytes / alignedInputRowBytes;
        }
        int64_t maxIndicesForThisTable = Std::min(Std::min(maxIndicesByQueIn, maxIndicesByTmpQue), indicesNumOneBlock);

        int64_t thisLen = remain;
        while (remain > 0) {
            if (thisLen > maxIndicesForThisTable) {
                thisLen = maxIndicesForThisTable;
            }
            remain -= thisLen;

            // copyIn（包含反量化）
            CopyInWithDequantize(startIndices, thisLen, embedDim, thisWeightOffset, tableIndex);

            // compute（pooling，累加到outLt）
            Pooling(outLt, thisLen, embedDim, weightTypeEnum, isWeighted);

            startIndices = startIndices + thisLen;
            thisLen = remain;
        }

        if (poolMode == static_cast<int64_t>(PoolingMode::MEAN)) {
            Muls<float>(outLt, outLt, meanLen, embedDim);
        }
        queOut.EnQue(outLt);
        LocalTensor<float> finalOutLt = queOut.DeQue<float>();
        CopyOut(finalOutLt, outOffset, embedDim);
        queOut.FreeTensor(finalOutLt);
    }

    // ========== 成员变量 ==========

    // Shape参数
    int64_t devWeightsDim0;
    int64_t weightsOffsetsDim0;
    int64_t dOffsetsDim0;
    int64_t indicesDim0;
    int64_t offsetsDim0;
    int64_t outDim0;
    int64_t outDim1;
    int64_t maxD;
    int64_t batchs;

    // Pooling模式参数
    int64_t poolMode;

    // Weighted模式标志
    bool isWeighted;

    // 输出数据类型
    int64_t outputDtype;

    // FP8参数
    int64_t fp8ExponentBits;
    int64_t fp8ExponentBias;
    int64_t rowAlignment;
    uint32_t fp8SignShift;
    uint32_t fp8BodyShift;
    float fp8Multiplier;

    // Tiling参数
    int64_t splitBaseLen;
    int64_t tailSplitIndex;
    int64_t lenOfThisCore;
    int64_t offsetOfThisCore;

    // UB内存参数
    int64_t ubCanUsed;
    int64_t queInBytes;
    int64_t tmpQueBytes;
    int64_t fp8SignUint32Bytes;
    int64_t indicesNumOneBlock;

    // TPipe和TQue
    TPipe pipe;
    TQue<TPosition::VECIN, 1> queIndices;
    TQue<TPosition::VECIN, 1> queIn;
    TQue<TPosition::VECIN, 1> tmpQue;
    TQue<TPosition::VECIN, 1> queWeights;
    TQue<TPosition::VECOUT, 1> queOut;
    TQue<TPosition::VECOUT, 1> newqueOut;
    TBuf<TPosition::VECCALC> fp8MaskBuf;
    TBuf<TPosition::VECCALC> fp8TmpByteBuf;
    TBuf<TPosition::VECCALC> fp8SignUint32Buf;

    // GlobalTensor
    GlobalTensor<uint8_t> devWeightsGT;
    GlobalTensor<IndexType> indicesGT;
    GlobalTensor<IndexType> offsetGT;
    GlobalTensor<int32_t> dOffsetGT;
    GlobalTensor<int64_t> weightOffsetGT;
    GlobalTensor<uint8_t> weightsTysGT;
    GlobalTensor<float> indiceWeightsGT;
    GlobalTensor<OutputType> outGT;
};

}  // namespace IntNBitSplitEmbeddingPooling

#endif  // INT_NBIT_SPLIT_EMBEDDING_POOLING_KERNEL_H
