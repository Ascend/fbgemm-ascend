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

#ifndef FLOAT_OR_HALF_TO_FUSED_NBIT_ROWWISE_SIMT_KERNEL_H
#define FLOAT_OR_HALF_TO_FUSED_NBIT_ROWWISE_SIMT_KERNEL_H

#include "common.h"
#include "simt_api/asc_simt.h"
#include "simt_api/vector_functions.h"
#include <cstdint>
#include <type_traits>

namespace FloatOrHalfToFusedNbitRowwiseSimt {

using namespace AscendC;
using FloatOrHalfToFusedNbitRowwiseCommon::Args;

constexpr int32_t MAX_BUFFER_NUM = 2;
constexpr uint32_t MAX_THREADS_PER_BLOCK = 256;

union HalfBits {
    half value;
    uint16_t bits;
};

union Vec64Bits {
    uint2 u2;
    float f32[2];
    uint16_t h16[4];
    uint8_t u8[8];
};

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void QuantizeRowsSimt(
    __local_mem__ T* input, __local_mem__ uint8_t* output, uint32_t rowsThisCycle, uint32_t ncols, uint32_t bitRate,
    uint32_t numElemPerByte, uint32_t outputColumns)
{
    uint32_t rowStep = blockDim.x;
    int32_t maxQuant = (1 << bitRate) - 1;
    uint32_t embBytes = ncols / numElemPerByte;
    constexpr uint32_t elemsPerVec = sizeof(uint64_t) / sizeof(T);
    uint32_t vecCols = ncols / elemsPerVec;

    for (uint32_t row = threadIdx.x; row < rowsThisCycle; row += rowStep) {
        __local_mem__ T* inputRow = input + row * ncols;
        __local_mem__ uint8_t* outputRow = output + row * outputColumns;

        // ------ Stage 1: min/max + scale ------
        float minVal = FLT_MAX, maxVal = -FLT_MAX;
        __local_mem__ float2* inF2 = reinterpret_cast<__local_mem__ float2*>(inputRow);
        __local_mem__ uint2* inU2 = reinterpret_cast<__local_mem__ uint2*>(inputRow);

        for (uint32_t vecIdx = 0; vecIdx < vecCols; ++vecIdx) {
            if constexpr (std::is_same<T, float>::value) {
                float2 v = inF2[vecIdx];
                minVal = min(minVal, min(v.x, v.y));
                maxVal = max(maxVal, max(v.x, v.y));
            } else {
                Vec64Bits v;
                v.u2 = inU2[vecIdx];
                for (uint32_t i = 0; i < elemsPerVec; ++i) {
                    HalfBits hv;
                    hv.bits = v.h16[i];
                    float vf = static_cast<float>(hv.value);
                    minVal = min(minVal, vf);
                    maxVal = max(maxVal, vf);
                }
            }
        }

        for (uint32_t col = vecCols * elemsPerVec; col < ncols; ++col) {
            float vf = static_cast<float>(inputRow[col]);
            minVal = min(minVal, vf);
            maxVal = max(maxVal, vf);
        }

        half minHalf = static_cast<half>(minVal);
        float minF = static_cast<float>(minHalf);
        float range = maxVal - minF;
        float scale = (range == 0.0f) ? 1.0f : range / static_cast<float>(maxQuant);

        half scaleHalf = static_cast<half>(scale);
        scale = static_cast<float>(scaleHalf);
        if (scale == 0.0f) {
            scale = 1.0f;
            scaleHalf = static_cast<half>(1.0f);
        }

        float inverseScale = 1.0f / scale;
        if (inverseScale > FLT_MAX) {
            inverseScale = 1.0f;
            scaleHalf = static_cast<half>(1.0f);
        }

        // ------ Stage 2: quantize and pack ------
        uint32_t bitBuf = 0, bitsFilled = 0, outByteIdx = 0;
        uint32_t targetBits = ((reinterpret_cast<uintptr_t>(outputRow) & 3) != 0) ? 16 : 32;  // 动态写入位宽

        for (uint32_t vecIdx = 0; vecIdx < vecCols; ++vecIdx) {
            Vec64Bits inVec;
            if constexpr (std::is_same<T, float>::value) {
                float2 f2Val = inF2[vecIdx];
                inVec.f32[0] = f2Val.x;
                inVec.f32[1] = f2Val.y;
            } else {
                inVec.u2 = inU2[vecIdx];
            }

            for (uint32_t i = 0; i < elemsPerVec; ++i) {
                float v;
                if constexpr (std::is_same<T, float>::value) {
                    v = inVec.f32[i];
                } else {
                    HalfBits hv;
                    hv.bits = inVec.h16[i];
                    v = static_cast<float>(hv.value);
                }

                int32_t q = static_cast<int32_t>(lrintf((v - minF) * inverseScale));
                q = (q < 0) ? 0 : ((q > maxQuant) ? maxQuant : q);

                bitBuf |= (static_cast<uint32_t>(q) << bitsFilled);
                bitsFilled += bitRate;

                if (bitsFilled == targetBits) {
                    if (targetBits == 16) {
                        *reinterpret_cast<__local_mem__ uint16_t*>(outputRow + outByteIdx) =
                            static_cast<uint16_t>(bitBuf);
                        outByteIdx += 2;
                        targetBits = 32;
                    } else {
                        *reinterpret_cast<__local_mem__ uint32_t*>(outputRow + outByteIdx) = bitBuf;
                        outByteIdx += 4;
                    }
                    bitBuf = 0;
                    bitsFilled = 0;
                }
            }
        }

        for (uint32_t col = vecCols * elemsPerVec; col < ncols; ++col) {
            float v = static_cast<float>(inputRow[col]);
            int32_t q = static_cast<int32_t>(lrintf((v - minF) * inverseScale));
            q = (q < 0) ? 0 : ((q > maxQuant) ? maxQuant : q);

            bitBuf |= (static_cast<uint32_t>(q) << bitsFilled);
            bitsFilled += bitRate;

            if (bitsFilled == targetBits) {
                if (targetBits == 16) {
                    *reinterpret_cast<__local_mem__ uint16_t*>(outputRow + outByteIdx) = static_cast<uint16_t>(bitBuf);
                    outByteIdx += 2;
                    targetBits = 32;
                } else {
                    *reinterpret_cast<__local_mem__ uint32_t*>(outputRow + outByteIdx) = bitBuf;
                    outByteIdx += 4;
                }
                bitBuf = 0;
                bitsFilled = 0;
            }
        }

        // 根据 ncols % (2 * numElemPerByte) == 0，剩余0或16位
        if (bitsFilled > 0) {
            *reinterpret_cast<__local_mem__ uint16_t*>(outputRow + outByteIdx) = static_cast<uint16_t>(bitBuf);
        }

        // ------ Stage 3: write scale and bias ------
        __local_mem__ uint16_t* scaleBiasPtr = reinterpret_cast<__local_mem__ uint16_t*>(outputRow + embBytes);
        HalfBits scaleBits;
        scaleBits.value = scaleHalf;
        HalfBits biasBits;
        biasBits.value = minHalf;

        scaleBiasPtr[0] = scaleBits.bits;
        scaleBiasPtr[1] = biasBits.bits;
    }
}

template <typename T>
class KernelSimt {
public:
    __aicore__ inline KernelSimt(Args& args)
    {
        GET_TILING_DATA(tilingData, args.tiling);

        nrows = tilingData.nrows;
        ncols = tilingData.ncols;
        bitRate = tilingData.bitRate;
        outputColumns = tilingData.outputColumns;
        numElemPerByte = tilingData.numElemPerByte;
        rowsPerCycle = static_cast<uint32_t>(tilingData.rowsPerCycle);
        bufferNum = static_cast<uint32_t>(tilingData.bufferNum);

        int64_t inputTotal = nrows * ncols;
        int64_t outputTotal = nrows * outputColumns;
        inputGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args.input), inputTotal);
        outputGT.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.output), outputTotal);

        int64_t inputCycleElements = rowsPerCycle * ncols;
        int64_t outputCycleElements = rowsPerCycle * outputColumns;
        pipe.InitBuffer(inputQueue, bufferNum, inputCycleElements * sizeof(T));
        pipe.InitBuffer(outputQueue, bufferNum, outputCycleElements);
    }

    __aicore__ inline void Process()
    {
        int64_t coreIdx = static_cast<int64_t>(GetBlockIdx());
        int64_t coreNum = static_cast<int64_t>(GetBlockNum());

        int64_t rowsBase = nrows / coreNum;
        int64_t rowsTail = nrows % coreNum;
        int64_t rowsThisCore = rowsBase;
        if (coreIdx < rowsTail) {
            rowsThisCore += 1;
        }
        if (rowsThisCore <= 0) {
            return;
        }

        int64_t startRow = coreIdx * rowsBase;
        if (coreIdx < rowsTail) {
            startRow += coreIdx;
        } else {
            startRow += rowsTail;
        }

        int64_t processedRows = 0;
        while (processedRows < rowsThisCore) {
            int64_t rowsRemain = rowsThisCore - processedRows;
            int64_t rowsThisCycle = rowsRemain;
            if (rowsThisCycle > rowsPerCycle) {
                rowsThisCycle = rowsPerCycle;
            }

            int64_t globalRowOffset = startRow + processedRows;
            int64_t inputOffset = globalRowOffset * ncols;
            int64_t outputOffset = globalRowOffset * outputColumns;

            int64_t inputElements = rowsThisCycle * ncols;
            int64_t outputElements = rowsThisCycle * outputColumns;

            LocalTensor<T> inputLt = inputQueue.AllocTensor<T>();
            CpGm2Local(inputLt, inputGT[inputOffset], inputElements);
            inputQueue.EnQue(inputLt);
            inputLt = inputQueue.DeQue<T>();

            LocalTensor<uint8_t> outputLt = outputQueue.AllocTensor<uint8_t>();

            __local_mem__ T* inputLocal = reinterpret_cast<__local_mem__ T*>(inputLt.GetPhyAddr());
            __local_mem__ uint8_t* outputLocal = reinterpret_cast<__local_mem__ uint8_t*>(outputLt.GetPhyAddr());

            uint32_t threadCount = static_cast<uint32_t>(rowsThisCycle);

            AscendC::Simt::VF_CALL<QuantizeRowsSimt<T>>(
                AscendC::Simt::Dim3{threadCount, 1, 1}, inputLocal, outputLocal, static_cast<uint32_t>(rowsThisCycle),
                static_cast<uint32_t>(ncols), bitRate, numElemPerByte, static_cast<uint32_t>(outputColumns));

            outputQueue.EnQue(outputLt);
            outputLt = outputQueue.DeQue<uint8_t>();
            CpLocal2Gm(outputGT[outputOffset], outputLt, outputElements);

            outputQueue.FreeTensor(outputLt);
            inputQueue.FreeTensor(inputLt);
            processedRows += rowsThisCycle;
        }
    }

private:
    int64_t nrows;
    int64_t ncols;
    uint32_t bitRate;
    uint64_t outputColumns;
    uint32_t numElemPerByte;
    uint32_t rowsPerCycle;
    uint32_t bufferNum;

    TPipe pipe;
    TQue<TPosition::VECIN, MAX_BUFFER_NUM> inputQueue;
    TQue<TPosition::VECOUT, MAX_BUFFER_NUM> outputQueue;

    GlobalTensor<T> inputGT;
    GlobalTensor<uint8_t> outputGT;
};

}  // namespace FloatOrHalfToFusedNbitRowwiseSimt

#endif
