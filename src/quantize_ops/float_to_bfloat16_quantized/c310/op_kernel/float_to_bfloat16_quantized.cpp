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

namespace FloatToBfloat16QuantizedSimt {

union fint32 {
    float F;
    uint32_t I;
};

__simt_callee__ inline uint16_t float_to_bfloat16(float val_fp)
{
    fint32 temp;
    temp.F = val_fp;
    // Round-to-nearest: add 0.5 ULP of the lower 16 bits, then truncate
    return static_cast<uint16_t>((temp.I + (1U << 15)) >> 16);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void FloatToBfloat16Kernel(__gm__ float* input, __gm__ uint8_t* output,
                                                                            int64_t totalElems)
{
    const int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    const int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    const int32_t blockNum = AscendC::Simt::GetBlockNum();
    const int32_t thdPerBlock = blockDim.x;
    const int32_t gridSize = blockNum * thdPerBlock;

    int64_t idx = static_cast<int64_t>(blockIdx) * thdPerBlock + tid;
    bool useVectorized = (totalElems % 4 == 0);
    int64_t vecElems = totalElems / 4;

    if (useVectorized) {
        const __gm__ float4* in4 = reinterpret_cast<const __gm__ float4*>(input);
        __gm__ uint64_t* out64 = reinterpret_cast<__gm__ uint64_t*>(output);
        for (int64_t i = idx; i < vecElems; i += gridSize) {
            float4 v = in4[i];
            uint16_t b0 = float_to_bfloat16(v.x);
            uint16_t b1 = float_to_bfloat16(v.y);
            uint16_t b2 = float_to_bfloat16(v.z);
            uint16_t b3 = float_to_bfloat16(v.w);
            out64[i] = static_cast<uint64_t>(b0) | (static_cast<uint64_t>(b1) << 16) |
                       (static_cast<uint64_t>(b2) << 32) | (static_cast<uint64_t>(b3) << 48);
        }
    }

    int64_t scalarStart = useVectorized ? (vecElems * 4) : 0;
    __gm__ uint16_t* out16 = reinterpret_cast<__gm__ uint16_t*>(output);
    for (int64_t i = scalarStart + idx; i < totalElems; i += gridSize) {
        out16[i] = float_to_bfloat16(input[i]);
    }
}

}  // namespace FloatToBfloat16QuantizedSimt

extern "C" __global__ __aicore__ void float_to_bfloat16_quantized(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                                                  GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    __gm__ float* gmInput = reinterpret_cast<__gm__ float*>(input);
    __gm__ uint8_t* gmOutput = reinterpret_cast<__gm__ uint8_t*>(output);

    AscendC::Simt::VF_CALL<FloatToBfloat16QuantizedSimt::FloatToBfloat16Kernel>(
        AscendC::Simt::Dim3{tilingData.blockDim, 1, 1}, gmInput, gmOutput, tilingData.totalElems);
}