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

namespace Bfloat16QuantizedToFloatSimt {

union fint32 {
    uint32_t I;
    float F;
};

__simt_callee__ inline float bfloat16_to_float(uint16_t val_bf16)
{
    fint32 temp;
    temp.I = static_cast<uint32_t>(val_bf16) << 16;
    return temp.F;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void Bfloat16QuantizedToFloatKernel(__gm__ uint8_t* input,
                                                                                     __gm__ float* output,
                                                                                     int64_t totalElems)
{
    const int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    const int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    const int32_t blockNum = AscendC::Simt::GetBlockNum();
    const int32_t thdPerBlock = blockDim.x;
    const int32_t gridSize = blockNum * thdPerBlock;

    int64_t idx = static_cast<int64_t>(blockIdx) * thdPerBlock + tid;
    constexpr int32_t ELEMS_PER_VEC = sizeof(uint64_t) / sizeof(uint16_t);
    int64_t vecElems = totalElems / ELEMS_PER_VEC;

    const __gm__ uint64_t* in64 = reinterpret_cast<const __gm__ uint64_t*>(input);
    __gm__ float4* out4 = reinterpret_cast<__gm__ float4*>(output);
    for (int64_t i = idx; i < vecElems; i += gridSize) {
        uint64_t packed = in64[i];
        uint16_t b0 = static_cast<uint16_t>(packed);
        uint16_t b1 = static_cast<uint16_t>(packed >> 16);
        uint16_t b2 = static_cast<uint16_t>(packed >> 32);
        uint16_t b3 = static_cast<uint16_t>(packed >> 48);
        float4 v;
        v.x = bfloat16_to_float(b0);
        v.y = bfloat16_to_float(b1);
        v.z = bfloat16_to_float(b2);
        v.w = bfloat16_to_float(b3);
        out4[i] = v;
    }

    int64_t scalarStart = vecElems * 4;
    if (scalarStart < totalElems) {
        const __gm__ uint16_t* in16 = reinterpret_cast<const __gm__ uint16_t*>(input);
        for (int64_t i = scalarStart + idx; i < totalElems; i += gridSize) {
            output[i] = bfloat16_to_float(in16[i]);
        }
    }
}

}  // namespace Bfloat16QuantizedToFloatSimt

extern "C" __global__ __aicore__ void bfloat16_quantized_to_float(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                                                  GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    __gm__ uint8_t* gmInput = reinterpret_cast<__gm__ uint8_t*>(input);
    __gm__ float* gmOutput = reinterpret_cast<__gm__ float*>(output);

    AscendC::Simt::VF_CALL<Bfloat16QuantizedToFloatSimt::Bfloat16QuantizedToFloatKernel>(
        AscendC::Simt::Dim3{tilingData.blockDim, 1, 1}, gmInput, gmOutput, tilingData.totalElems);
}
