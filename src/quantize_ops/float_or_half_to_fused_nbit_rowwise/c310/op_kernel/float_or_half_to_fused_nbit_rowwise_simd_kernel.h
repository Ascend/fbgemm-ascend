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

#ifndef FLOAT_OR_HALF_TO_FUSED_NBIT_ROWWISE_SIMD_KERNEL_H
#define FLOAT_OR_HALF_TO_FUSED_NBIT_ROWWISE_SIMD_KERNEL_H

#include "common.h"

namespace FloatOrHalfToFusedNbitRowwiseSimd {

using namespace AscendC;
using FloatOrHalfToFusedNbitRowwiseCommon::Args;
using FloatOrHalfToFusedNbitRowwiseCommon::SCALE_BIAS_BYTES;

constexpr int32_t DATA_ALIGN_BYTES = 32;
constexpr int32_t REDUCE_ELEM_ALIGN = 64;

template <typename T>
class KernelSimd {
public:
    __aicore__ inline KernelSimd(Args& args)
    {
        GET_TILING_DATA(tilingData, args.tiling);

        nrows = tilingData.nrows;
        ncols = tilingData.ncols;
        bitRate = tilingData.bitRate;
        blockLen = tilingData.blockLen;
        outputColumns = tilingData.outputColumns;
        numElemPerByte = tilingData.numElemPerByte;
        maxQuant = (1 << bitRate) - 1;

        int64_t coreIdx = GetBlockIdx();
        int64_t splitBaseLen = tilingData.splitBaseLen;
        int32_t tailSplitIndex = tilingData.tailSplitIndex;

        if (coreIdx < tailSplitIndex) {
            rowCount = splitBaseLen + 1;
            rowStart = coreIdx * (splitBaseLen + 1);
        } else {
            rowCount = splitBaseLen;
            rowStart = tailSplitIndex * (splitBaseLen + 1) + (coreIdx - tailSplitIndex) * splitBaseLen;
        }

        if (rowStart >= nrows) {
            rowCount = 0;
            rowStart = 0;
            return;
        }
        if (rowStart + rowCount > nrows) {
            rowCount = nrows - rowStart;
        }

        auto* inputBase = reinterpret_cast<__gm__ T*>(args.input) + rowStart * ncols;
        inputGT.SetGlobalBuffer(inputBase, rowCount * ncols);

        auto* outputBase = reinterpret_cast<__gm__ uint8_t*>(args.output) + rowStart * outputColumns;
        outputGT.SetGlobalBuffer(outputBase, rowCount * outputColumns);

        int64_t fp32Bytes = blockLen * sizeof(float);
        int64_t fp16Bytes = blockLen * sizeof(half);
        int64_t outBufBytes = ((blockLen + DATA_ALIGN_BYTES - 1) / DATA_ALIGN_BYTES) * DATA_ALIGN_BYTES;
        int64_t idxElemMax = AscendC::CeilDivision(blockLen, numElemPerByte);
        int64_t idxBytes = idxElemMax * sizeof(int32_t);
        int64_t idxBufBytes = ((idxBytes + DATA_ALIGN_BYTES - 1) / DATA_ALIGN_BYTES) * DATA_ALIGN_BYTES;
        int64_t tmpPackedBytes = ((idxElemMax + DATA_ALIGN_BYTES - 1) / DATA_ALIGN_BYTES) * DATA_ALIGN_BYTES;

        pipe.InitBuffer(rawBuf, blockLen * sizeof(T));
        pipe.InitBuffer(workBuf, fp32Bytes);
        pipe.InitBuffer(outBuf, outBufBytes);
        pipe.InitBuffer(quantBuf, outBufBytes);
        pipe.InitBuffer(quantFp16Buf, fp16Bytes);
        pipe.InitBuffer(idxBaseBuf, idxBufBytes);
        pipe.InitBuffer(idxWorkBuf, idxBufBytes);
        pipe.InitBuffer(packTmpBuf, tmpPackedBytes);

        rawLT = rawBuf.Get<T>(blockLen);
        workLT = workBuf.Get<float>(blockLen);
        outLT = outBuf.Get<uint8_t>(outBufBytes);
        quantLT = quantBuf.Get<uint8_t>(outBufBytes);
        quantFp16LT = quantFp16Buf.Get<half>(blockLen);
        idxBaseLT = idxBaseBuf.Get<int32_t>(idxBufBytes / sizeof(int32_t));
        idxWorkLT = idxWorkBuf.Get<int32_t>(idxBufBytes / sizeof(int32_t));
        packTmpLT = packTmpBuf.Get<uint8_t>(tmpPackedBytes);

        if constexpr (std::is_same_v<T, float>) {
            fp32LT = rawLT.template ReinterpretCast<float>();
        } else {
            pipe.InitBuffer(fp32Buf, fp32Bytes);
            fp32LT = fp32Buf.Get<float>(blockLen);
        }
    }

    __aicore__ inline void Process()
    {
        for (int64_t i = 0; i < rowCount; i++) {
            ProcessRow(i);
        }
    }

private:
    __aicore__ inline void LoadBlockToFp32(int64_t gmOffset, int32_t len)
    {
        CpGm2Local<T>(rawLT, inputGT[gmOffset], len);
        pipe_barrier(PIPE_ALL);
        if constexpr (std::is_same_v<T, half>) {
            Cast(fp32LT, rawLT, RoundMode::CAST_NONE, len);
        }
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline int32_t AlignForReduce(int32_t len)
    {
        int32_t alignedLen = ((len + REDUCE_ELEM_ALIGN - 1) / REDUCE_ELEM_ALIGN) * REDUCE_ELEM_ALIGN;
        if (alignedLen > len) {
            float padVal = fp32LT.GetValue(0);
            for (int32_t i = len; i < alignedLen; ++i) {
                fp32LT.SetValue(i, padVal);
            }
        }
        return alignedLen;
    }

    __aicore__ inline void PackQuantizedSimd(int32_t len)
    {
        if (len <= 0) {
            return;
        }

        int32_t paddedLen = AscendC::CeilDivision(len, numElemPerByte) * numElemPerByte;
        int32_t packedBytes = paddedLen / numElemPerByte;
        if (bitRate == 8) {
            CreateVecIndex<int32_t>(idxBaseLT, 0, packedBytes);
            auto idxBaseU32 = idxBaseLT.template ReinterpretCast<uint32_t>();
            Gather(outLT, quantLT, idxBaseU32, 0, packedBytes);
            return;
        }

        CreateVecIndex<int32_t>(idxBaseLT, 0, packedBytes);
        Muls<int32_t>(idxBaseLT, idxBaseLT, numElemPerByte, packedBytes);

        auto idxBaseU32 = idxBaseLT.template ReinterpretCast<uint32_t>();
        Gather(outLT, quantLT, idxBaseU32, 0, packedBytes);

        int32_t gatherTimes = numElemPerByte;
        for (int32_t i = 1; i < gatherTimes; ++i) {
            Adds<int32_t>(idxWorkLT, idxBaseLT, i, packedBytes);
            auto idxWorkU32 = idxWorkLT.template ReinterpretCast<uint32_t>();

            Gather(packTmpLT, quantLT, idxWorkU32, 0, packedBytes);
            ShiftLeft<uint8_t>(packTmpLT, packTmpLT, i * bitRate, packedBytes);
            Or<uint8_t>(outLT, outLT, packTmpLT, packedBytes);
        }
    }

    __aicore__ inline void ProcessRow(int64_t localRow)
    {
        // ---- SIMD ReduceMin / ReduceMax ----
        float globalMin = FLT_MAX;
        float globalMax = -FLT_MAX;

        LocalTensor<float> reduceOutLT = outLT.ReinterpretCast<float>();

        for (int64_t offset = 0; offset < ncols; offset += blockLen) {
            int32_t len = static_cast<int32_t>(ncols - offset);
            if (ncols - offset > blockLen) {
                len = static_cast<int32_t>(blockLen);
            }

            LoadBlockToFp32(localRow * ncols + offset, len);

            int32_t alignedLen = AlignForReduce(len);

            ReduceMin(reduceOutLT, fp32LT, workLT, alignedLen);
            pipe_barrier(PIPE_ALL);
            float blockMin = reduceOutLT.GetValue(0);

            ReduceMax(reduceOutLT, fp32LT, workLT, alignedLen);
            pipe_barrier(PIPE_ALL);
            float blockMax = reduceOutLT.GetValue(0);

            if (blockMin < globalMin) {
                globalMin = blockMin;
            }
            if (blockMax > globalMax) {
                globalMax = blockMax;
            }
        }

        // ---- Compute scale, bias, inverse_scale ----
        half minHalf = static_cast<half>(globalMin);
        float minVal = static_cast<float>(minHalf);
        float range = globalMax - minVal;

        float scaleRaw = (range == 0.0f) ? 1.0f : range / static_cast<float>(maxQuant);
        half scaleHalf = static_cast<half>(scaleRaw);
        float scale = static_cast<float>(scaleHalf);

        if (scale == 0.0f) {
            scale = 1.0f;
            scaleHalf = static_cast<half>(1.0f);
        }
        float invScale = 1.0f / scale;
        if (invScale > FLT_MAX) {
            scale = 1.0f;
            scaleHalf = static_cast<half>(1.0f);
            invScale = 1.0f;
        }

        // ---- SIMD quantize + scalar pack ----
        int64_t embBytes = (ncols + numElemPerByte - 1) / numElemPerByte;

        for (int64_t offset = 0; offset < ncols; offset += blockLen) {
            int32_t len = static_cast<int32_t>(ncols - offset);
            if (len > static_cast<int32_t>(blockLen)) {
                len = static_cast<int32_t>(blockLen);
            }
            int32_t packedBytes = AscendC::CeilDivision(len, numElemPerByte);

            LoadBlockToFp32(localRow * ncols + offset, len);

            Adds(fp32LT, fp32LT, -minVal, len);
            Muls(fp32LT, fp32LT, invScale, len);
            Maxs(fp32LT, fp32LT, 0.0f, len);
            Mins(fp32LT, fp32LT, static_cast<float>(maxQuant), len);
            Round<float>(fp32LT, fp32LT, len);

            int32_t paddedLen = AscendC::CeilDivision(len, numElemPerByte) * numElemPerByte;
            Duplicate<uint8_t>(quantLT, static_cast<uint8_t>(0), paddedLen);
            Cast(quantFp16LT, fp32LT, RoundMode::CAST_NONE, len);
            Cast(quantLT, quantFp16LT, RoundMode::CAST_NONE, len);
            pipe_barrier(PIPE_ALL);

            PackQuantizedSimd(len);

            pipe_barrier(PIPE_ALL);

            int64_t gmByteOffset = offset / numElemPerByte;
            CpLocal2Gm<uint8_t>(outputGT[localRow * outputColumns + gmByteOffset], outLT, packedBytes);
        }

        // ---- Write scale (half) and bias (half) to end of output row ----
        half biasHalf = minHalf;
        LocalTensor<half> scaleBiasLT = outLT.ReinterpretCast<half>();
        scaleBiasLT.SetValue(0, scaleHalf);
        scaleBiasLT.SetValue(1, biasHalf);

        CpLocal2Gm<uint8_t>(outputGT[localRow * outputColumns + embBytes], outLT, SCALE_BIAS_BYTES);
    }

    int64_t nrows;
    int64_t ncols;
    int32_t bitRate;
    int64_t blockLen;
    int64_t outputColumns;
    int32_t numElemPerByte;
    int32_t maxQuant;
    int64_t rowStart;
    int64_t rowCount;

    GlobalTensor<T> inputGT;
    GlobalTensor<uint8_t> outputGT;

    LocalTensor<T> rawLT;
    LocalTensor<float> fp32LT;
    LocalTensor<float> workLT;
    LocalTensor<uint8_t> outLT;
    LocalTensor<uint8_t> quantLT;
    LocalTensor<half> quantFp16LT;
    LocalTensor<int32_t> idxBaseLT;
    LocalTensor<int32_t> idxWorkLT;
    LocalTensor<uint8_t> packTmpLT;

    TPipe pipe;
    TBuf<TPosition::VECCALC> rawBuf;
    TBuf<TPosition::VECCALC> fp32Buf;
    TBuf<TPosition::VECCALC> workBuf;
    TBuf<TPosition::VECCALC> outBuf;
    TBuf<TPosition::VECCALC> quantBuf;
    TBuf<TPosition::VECCALC> quantFp16Buf;
    TBuf<TPosition::VECCALC> idxBaseBuf;
    TBuf<TPosition::VECCALC> idxWorkBuf;
    TBuf<TPosition::VECCALC> packTmpBuf;
};

}  // namespace FloatOrHalfToFusedNbitRowwiseSimd

#endif
