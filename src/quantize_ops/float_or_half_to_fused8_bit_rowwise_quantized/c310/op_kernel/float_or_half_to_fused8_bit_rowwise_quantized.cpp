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

#include <cmath>
#include <limits>
#include <type_traits>
#include "kernel_operator.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/asc_fp16.h"

#define DTYPE_DISPATCH(dtype, DTYPE, BODY) \
    do {                                   \
        if (dtype == 0) {                  \
            using DTYPE = float;           \
            BODY;                          \
        } else if (dtype == 1) {           \
            using DTYPE = half;            \
            BODY;                          \
        } else {                           \
            using DTYPE = bfloat16_t;      \
            BODY;                          \
        }                                  \
    } while (0)

using namespace AscendC;

namespace FloatOrHalfToFused8BitRowwiseQuantizedSimt {

constexpr float EPSILON = 1e-20f;
constexpr float SCALE_FACTOR = 255.0f;
constexpr float MAX_FLOAT = std::numeric_limits<float>::max();
constexpr float MIN_FLOAT = std::numeric_limits<float>::lowest();

__simt_callee__ inline uint8_t QuantizeOne(float v, float bias, float invScale)
{
    float q = (v - bias) * invScale;
    q = (q < 0.0f ? 0.0f : q);
    q = (q > 255.0f ? 255.0f : q);
    q += 0.5f;  // round to nearest, matching FBGEMM CUDA's lrintf behavior
    return static_cast<uint8_t>(q);
}

__simt_callee__ inline void Unpack4Half(uint64_t p, float& v0, float& v1, float& v2, float& v3)
{
    union {
        half h;
        uint16_t u;
    } e0, e1, e2, e3;
    e0.u = static_cast<uint16_t>(p & 0xFFFF);
    e1.u = static_cast<uint16_t>((p >> 16) & 0xFFFF);
    e2.u = static_cast<uint16_t>((p >> 32) & 0xFFFF);
    e3.u = static_cast<uint16_t>((p >> 48) & 0xFFFF);
    v0 = static_cast<float>(e0.h);
    v1 = static_cast<float>(e1.h);
    v2 = static_cast<float>(e2.h);
    v3 = static_cast<float>(e3.h);
}

__simt_callee__ inline void Unpack4Bf16(uint64_t p, float& v0, float& v1, float& v2, float& v3)
{
    union {
        bfloat16_t bf;
        uint16_t u;
    } e0, e1, e2, e3;
    e0.u = static_cast<uint16_t>(p & 0xFFFF);
    e1.u = static_cast<uint16_t>((p >> 16) & 0xFFFF);
    e2.u = static_cast<uint16_t>((p >> 32) & 0xFFFF);
    e3.u = static_cast<uint16_t>((p >> 48) & 0xFFFF);
    v0 = static_cast<float>(e0.bf);
    v1 = static_cast<float>(e1.bf);
    v2 = static_cast<float>(e2.bf);
    v3 = static_cast<float>(e3.bf);
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void QuantizeRowsParallel(__gm__ T* input, __gm__ uint8_t* output,
                                                                           int32_t rows, int32_t cols,
                                                                           int32_t ncolsAligned, int32_t threadsPerRow,
                                                                           int32_t rowsPerBlock)
{
    const int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    const int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    const int32_t blockNum = AscendC::Simt::GetBlockNum();

    const int32_t rowInBlock = tid / threadsPerRow;
    const int32_t laneInRow = tid % threadsPerRow;

    constexpr int32_t VEC_SIZE = 4;
    bool useVectorized = (cols % VEC_SIZE == 0) && (cols >= 128);
    int32_t vecCols = cols / VEC_SIZE;

    for (int32_t rowBlockStart = blockIdx * rowsPerBlock; rowBlockStart < rows;
         rowBlockStart += blockNum * rowsPerBlock) {
        const int32_t globalRow = rowBlockStart + rowInBlock;
        if (globalRow >= rows)
            return;
        __gm__ T* rowInput = input + static_cast<int64_t>(globalRow) * cols;
        __gm__ uint8_t* rowOutput = output + static_cast<int64_t>(globalRow) * (ncolsAligned + 8);

        float threadMin = MAX_FLOAT;
        float threadMax = MIN_FLOAT;

        if constexpr (std::is_same_v<T, float>) {
            if (useVectorized) {
                const __gm__ float4* in4 = reinterpret_cast<const __gm__ float4*>(rowInput);
                for (int32_t c = laneInRow; c < vecCols; c += threadsPerRow) {
                    float4 v = in4[c];
                    threadMin = AscendC::Std::min(threadMin, v.x);
                    threadMax = AscendC::Std::max(threadMax, v.x);
                    threadMin = AscendC::Std::min(threadMin, v.y);
                    threadMax = AscendC::Std::max(threadMax, v.y);
                    threadMin = AscendC::Std::min(threadMin, v.z);
                    threadMax = AscendC::Std::max(threadMax, v.z);
                    threadMin = AscendC::Std::min(threadMin, v.w);
                    threadMax = AscendC::Std::max(threadMax, v.w);
                }
            }
        } else if constexpr (std::is_same_v<T, half>) {
            if (useVectorized) {
                const __gm__ uint64_t* in64 = reinterpret_cast<const __gm__ uint64_t*>(rowInput);
                for (int32_t c = laneInRow; c < vecCols; c += threadsPerRow) {
                    float v0, v1, v2, v3;
                    Unpack4Half(in64[c], v0, v1, v2, v3);
                    threadMin = AscendC::Std::min(threadMin, v0);
                    threadMax = AscendC::Std::max(threadMax, v0);
                    threadMin = AscendC::Std::min(threadMin, v1);
                    threadMax = AscendC::Std::max(threadMax, v1);
                    threadMin = AscendC::Std::min(threadMin, v2);
                    threadMax = AscendC::Std::max(threadMax, v2);
                    threadMin = AscendC::Std::min(threadMin, v3);
                    threadMax = AscendC::Std::max(threadMax, v3);
                }
            }
        } else {
            if (useVectorized) {
                const __gm__ uint64_t* in64 = reinterpret_cast<const __gm__ uint64_t*>(rowInput);
                for (int32_t c = laneInRow; c < vecCols; c += threadsPerRow) {
                    float v0, v1, v2, v3;
                    Unpack4Bf16(in64[c], v0, v1, v2, v3);
                    threadMin = AscendC::Std::min(threadMin, v0);
                    threadMax = AscendC::Std::max(threadMax, v0);
                    threadMin = AscendC::Std::min(threadMin, v1);
                    threadMax = AscendC::Std::max(threadMax, v1);
                    threadMin = AscendC::Std::min(threadMin, v2);
                    threadMax = AscendC::Std::max(threadMax, v2);
                    threadMin = AscendC::Std::min(threadMin, v3);
                    threadMax = AscendC::Std::max(threadMax, v3);
                }
            }
        }

        int32_t scalarStart = useVectorized ? (vecCols * VEC_SIZE) : 0;
        for (int32_t col = scalarStart + laneInRow; col < cols; col += threadsPerRow) {
            float val = static_cast<float>(rowInput[col]);
            threadMin = AscendC::Std::min(threadMin, val);
            threadMax = AscendC::Std::max(threadMax, val);
        }

        threadMin = asc_reduce_min(threadMin);
        threadMax = asc_reduce_max(threadMax);

        float rowMin = threadMin, rowMax = threadMax;
        float range = rowMax - rowMin;
        float bias = rowMin;
        float invScale = SCALE_FACTOR / (range + EPSILON);

        if (useVectorized) {
            __gm__ uint32_t* out32 = reinterpret_cast<__gm__ uint32_t*>(rowOutput);
            if constexpr (std::is_same_v<T, float>) {
                const __gm__ float4* in4 = reinterpret_cast<const __gm__ float4*>(rowInput);
                for (int32_t c = laneInRow; c < vecCols; c += threadsPerRow) {
                    float4 v = in4[c];
                    uint8_t b0 = QuantizeOne(v.x, bias, invScale);
                    uint8_t b1 = QuantizeOne(v.y, bias, invScale);
                    uint8_t b2 = QuantizeOne(v.z, bias, invScale);
                    uint8_t b3 = QuantizeOne(v.w, bias, invScale);
                    out32[c] = (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
                }
            } else if constexpr (std::is_same_v<T, half>) {
                const __gm__ uint64_t* in64 = reinterpret_cast<const __gm__ uint64_t*>(rowInput);
                for (int32_t c = laneInRow; c < vecCols; c += threadsPerRow) {
                    float v0, v1, v2, v3;
                    Unpack4Half(in64[c], v0, v1, v2, v3);
                    uint8_t b0 = QuantizeOne(v0, bias, invScale);
                    uint8_t b1 = QuantizeOne(v1, bias, invScale);
                    uint8_t b2 = QuantizeOne(v2, bias, invScale);
                    uint8_t b3 = QuantizeOne(v3, bias, invScale);
                    out32[c] = (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
                }
            } else {
                const __gm__ uint64_t* in64 = reinterpret_cast<const __gm__ uint64_t*>(rowInput);
                for (int32_t c = laneInRow; c < vecCols; c += threadsPerRow) {
                    float v0, v1, v2, v3;
                    Unpack4Bf16(in64[c], v0, v1, v2, v3);
                    uint8_t b0 = QuantizeOne(v0, bias, invScale);
                    uint8_t b1 = QuantizeOne(v1, bias, invScale);
                    uint8_t b2 = QuantizeOne(v2, bias, invScale);
                    uint8_t b3 = QuantizeOne(v3, bias, invScale);
                    out32[c] = (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
                }
            }
        }
        for (int32_t col = scalarStart + laneInRow; col < cols; col += threadsPerRow) {
            float val = static_cast<float>(rowInput[col]);
            rowOutput[col] = QuantizeOne(val, bias, invScale);
        }
        if (laneInRow == 0) {
            __gm__ float* scaleBiasPtr = reinterpret_cast<__gm__ float*>(rowOutput + ncolsAligned);
            scaleBiasPtr[0] = range / SCALE_FACTOR;
            scaleBiasPtr[1] = bias;
        }
    }
}

}  // namespace FloatOrHalfToFused8BitRowwiseQuantizedSimt

extern "C" __global__ __aicore__ void float_or_half_to_fused8_bit_rowwise_quantized(GM_ADDR inputData, GM_ADDR output,
                                                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    int64_t dtype = tilingData.dtype;

    DTYPE_DISPATCH(dtype, DTYPE, {
        __gm__ DTYPE* gmInput = reinterpret_cast<__gm__ DTYPE*>(inputData);
        __gm__ uint8_t* gmOutput = reinterpret_cast<__gm__ uint8_t*>(output);

        int32_t totalThreads = tilingData.totalThreads;

        AscendC::Simt::VF_CALL<FloatOrHalfToFused8BitRowwiseQuantizedSimt::QuantizeRowsParallel<DTYPE>>(
            AscendC::Simt::Dim3{static_cast<uint32_t>(totalThreads), 1, 1}, gmInput, gmOutput,
            static_cast<int32_t>(tilingData.rows), static_cast<int32_t>(tilingData.cols),
            static_cast<int32_t>(tilingData.ncolsAligned), tilingData.threadsPerRow, tilingData.rowsPerBlock);
    });
}
