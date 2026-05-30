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

#include "kernel_operator.h"

// 输出类型分发宏：0=fp32, 1=fp16, else=bf16
#define OUTPUT_DTYPE_DISPATCH(dtype, OUT_T, BODY) \
    do {                                          \
        if (dtype == 0) {                         \
            using OUT_T = float;                  \
            BODY;                                 \
        } else if (dtype == 1) {                  \
            using OUT_T = half;                   \
            BODY;                                 \
        } else {                                  \
            using OUT_T = bfloat16_t;             \
            BODY;                                 \
        }                                         \
    } while (0)

using namespace AscendC;

namespace Fused8BitRowwiseQuantizedToFloatOrHalfSimt {

__simt_callee__ inline void ReadScaleBias(__gm__ const uint8_t* ptr, bool quantPaddingFloatType, float& scale,
                                          float& bias)
{
    if (quantPaddingFloatType) {
        const __gm__ float* pFloat = reinterpret_cast<const __gm__ float*>(ptr);
        scale = pFloat[0];
        bias = pFloat[1];
    } else {
        const __gm__ half* pHalf = reinterpret_cast<const __gm__ half*>(ptr);
        scale = static_cast<float>(pHalf[0]);
        bias = static_cast<float>(pHalf[1]);
    }
}

template <typename OUT_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void DequantizeRowsParallel(
    __gm__ const uint8_t* input, __gm__ OUT_T* output, int32_t rows, int32_t cols, int32_t outputCols,
    int32_t quantPaddingSize, bool scaleBiasLast, bool quantPaddingFloatType, int32_t threadsPerRow,
    int32_t threadsPerRowLog2, int32_t rowsPerBlock)
{
    const int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    const int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    const int32_t blockNum = AscendC::Simt::GetBlockNum();

    const int32_t laneInRow = (threadsPerRowLog2 >= 0) ? (tid & (threadsPerRow - 1)) : (tid % threadsPerRow);
    const int32_t rowInBlock = (threadsPerRowLog2 >= 0) ? (tid >> threadsPerRowLog2) : (tid / threadsPerRow);

    constexpr int32_t VEC_SIZE = 4;

    for (int32_t rowBlockStart = blockIdx * rowsPerBlock; rowBlockStart < rows;
         rowBlockStart += blockNum * rowsPerBlock) {
        const int32_t globalRow = rowBlockStart + rowInBlock;
        if (globalRow >= rows) {
            return;
        }

        __gm__ const uint8_t* rowInput = input + static_cast<int64_t>(globalRow) * cols;
        __gm__ OUT_T* rowOutput = output + static_cast<int64_t>(globalRow) * outputCols;

        __gm__ const uint8_t* scaleBiasPtr = nullptr;
        __gm__ const uint8_t* dataPtr = nullptr;

        if (scaleBiasLast) {
            dataPtr = rowInput;
            scaleBiasPtr = rowInput + outputCols;
        } else {
            scaleBiasPtr = rowInput;
            dataPtr = rowInput + 2 * quantPaddingSize;
        }

        float scale = 0.0f;
        float bias = 0.0f;
        ReadScaleBias(scaleBiasPtr, quantPaddingFloatType, scale, bias);

        int32_t vecCols = outputCols / VEC_SIZE;

        if constexpr (std::is_same_v<OUT_T, float>) {
            const __gm__ uint32_t* data32 = reinterpret_cast<const __gm__ uint32_t*>(dataPtr);
            __gm__ float4* out4 = reinterpret_cast<__gm__ float4*>(rowOutput);
            for (int32_t c = laneInRow; c < vecCols; c += threadsPerRow) {
                uint32_t p = data32[c];
                float4 v;
                v.x = static_cast<float>(p & 0xFF) * scale + bias;
                v.y = static_cast<float>((p >> 8) & 0xFF) * scale + bias;
                v.z = static_cast<float>((p >> 16) & 0xFF) * scale + bias;
                v.w = static_cast<float>((p >> 24) & 0xFF) * scale + bias;
                out4[c] = v;
            }
        } else if constexpr (std::is_same_v<OUT_T, half>) {
            const __gm__ uint32_t* data32 = reinterpret_cast<const __gm__ uint32_t*>(dataPtr);
            __gm__ uint64_t* out64 = reinterpret_cast<__gm__ uint64_t*>(rowOutput);
            for (int32_t c = laneInRow; c < vecCols; c += threadsPerRow) {
                uint32_t p = data32[c];
                union {
                    half h;
                    uint16_t u;
                } c0, c1, c2, c3;
                c0.h = static_cast<half>(static_cast<float>(p & 0xFF) * scale + bias);
                c1.h = static_cast<half>(static_cast<float>((p >> 8) & 0xFF) * scale + bias);
                c2.h = static_cast<half>(static_cast<float>((p >> 16) & 0xFF) * scale + bias);
                c3.h = static_cast<half>(static_cast<float>((p >> 24) & 0xFF) * scale + bias);
                uint64_t v = (static_cast<uint64_t>(c0.u) << 0) | (static_cast<uint64_t>(c1.u) << 16) |
                             (static_cast<uint64_t>(c2.u) << 32) | (static_cast<uint64_t>(c3.u) << 48);
                out64[c] = v;
            }
        } else {
            const __gm__ uint32_t* data32 = reinterpret_cast<const __gm__ uint32_t*>(dataPtr);
            __gm__ uint64_t* out64 = reinterpret_cast<__gm__ uint64_t*>(rowOutput);
            for (int32_t c = laneInRow; c < vecCols; c += threadsPerRow) {
                uint32_t p = data32[c];
                union {
                    bfloat16_t bf;
                    uint16_t u;
                } c0, c1, c2, c3;
                c0.bf = static_cast<bfloat16_t>(static_cast<float>(p & 0xFF) * scale + bias);
                c1.bf = static_cast<bfloat16_t>(static_cast<float>((p >> 8) & 0xFF) * scale + bias);
                c2.bf = static_cast<bfloat16_t>(static_cast<float>((p >> 16) & 0xFF) * scale + bias);
                c3.bf = static_cast<bfloat16_t>(static_cast<float>((p >> 24) & 0xFF) * scale + bias);
                uint64_t v = (static_cast<uint64_t>(c0.u) << 0) | (static_cast<uint64_t>(c1.u) << 16) |
                             (static_cast<uint64_t>(c2.u) << 32) | (static_cast<uint64_t>(c3.u) << 48);
                out64[c] = v;
            }
        }

        int32_t tailStart = vecCols * VEC_SIZE;
        for (int32_t col = tailStart + laneInRow; col < outputCols; col += threadsPerRow) {
            float val = static_cast<float>(dataPtr[col]);
            rowOutput[col] = static_cast<OUT_T>(val * scale + bias);
        }
    }
}

}  // namespace Fused8BitRowwiseQuantizedToFloatOrHalfSimt

extern "C" __global__ __aicore__ void fused8_bit_rowwise_quantized_to_float_or_half(GM_ADDR inputData, GM_ADDR output,
                                                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    int64_t dtype = tilingData.dtype;
    bool scaleBiasLast = tilingData.scaleBiasLast;
    bool quantPaddingFloatType = tilingData.quantPaddingFloatType;

    OUTPUT_DTYPE_DISPATCH(dtype, OUT_T, {
        __gm__ const uint8_t* gmInput = reinterpret_cast<__gm__ const uint8_t*>(inputData);
        __gm__ OUT_T* gmOutput = reinterpret_cast<__gm__ OUT_T*>(output);

        int32_t totalThreads = tilingData.totalThreads;

        AscendC::Simt::VF_CALL<Fused8BitRowwiseQuantizedToFloatOrHalfSimt::DequantizeRowsParallel<OUT_T>>(
            AscendC::Simt::Dim3{static_cast<uint32_t>(totalThreads), 1, 1}, gmInput, gmOutput,
            static_cast<int32_t>(tilingData.rows), static_cast<int32_t>(tilingData.cols),
            static_cast<int32_t>(tilingData.outputCols), static_cast<int32_t>(tilingData.quantPaddingSize),
            scaleBiasLast, quantPaddingFloatType, tilingData.threadsPerRow, tilingData.threadsPerRowLog2,
            tilingData.rowsPerBlock);
    });
}
