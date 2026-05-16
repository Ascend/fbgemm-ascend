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

namespace Hfp8QuantizedToFloatSimt
{

union fint32
{
    float f;
    uint32_t u;
};

__simt_callee__ inline float hfp8_to_float(uint8_t hfp8, int32_t ebits, uint32_t multiplier_u)
{
    fint32 val_out, sign;

    sign.u = (static_cast<uint32_t>(hfp8) & 0x80U) << 24;
    val_out.u = (static_cast<uint32_t>(hfp8) & 0x7FU) << (24 - (8 - ebits));

    fint32 mult;
    mult.u = multiplier_u;
    val_out.f *= mult.f;
    val_out.u |= sign.u;
    return val_out.f;
}

__simt_vf__ __aicore__ inline void Hfp8ToFloatKernel(__gm__ uint8_t* input, __gm__ float* output, int64_t totalElems,
                                                     int32_t ebits, int32_t exponent_bias)
{
    const int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    const int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    const int32_t blockNum = AscendC::Simt::GetBlockNum();
    const int32_t thdPerBlock = blockDim.x;
    const int32_t gridSize = blockNum * thdPerBlock;

    uint32_t multiplier_u = static_cast<uint32_t>(127 + (127 - exponent_bias)) << 23;

    int64_t idx = static_cast<int64_t>(blockIdx) * thdPerBlock + tid;
    bool useVectorized = (totalElems % 4 == 0);
    int64_t vecElems = totalElems / 4;

    if (useVectorized)
    {
        const __gm__ uint32_t* in32 = reinterpret_cast<const __gm__ uint32_t*>(input);
        __gm__ float4* out4 = reinterpret_cast<__gm__ float4*>(output);
        for (int64_t i = idx; i < vecElems; i += gridSize)
        {
            uint32_t v = in32[i];
            uint8_t b0 = static_cast<uint8_t>(v & 0xFFU);
            uint8_t b1 = static_cast<uint8_t>((v >> 8) & 0xFFU);
            uint8_t b2 = static_cast<uint8_t>((v >> 16) & 0xFFU);
            uint8_t b3 = static_cast<uint8_t>((v >> 24) & 0xFFU);
            out4[i] = float4{hfp8_to_float(b0, ebits, multiplier_u), hfp8_to_float(b1, ebits, multiplier_u),
                             hfp8_to_float(b2, ebits, multiplier_u), hfp8_to_float(b3, ebits, multiplier_u)};
        }
    }

    int64_t scalarStart = useVectorized ? (vecElems * 4) : 0;
    for (int64_t i = scalarStart + idx; i < totalElems; i += gridSize)
    {
        output[i] = hfp8_to_float(input[i], ebits, multiplier_u);
    }
}

}  // namespace Hfp8QuantizedToFloatSimt

extern "C" __global__ __aicore__ void hfp8_quantized_to_float(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                                              GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    __gm__ uint8_t* gmInput = reinterpret_cast<__gm__ uint8_t*>(input);
    __gm__ float* gmOutput = reinterpret_cast<__gm__ float*>(output);

    AscendC::Simt::VF_CALL<Hfp8QuantizedToFloatSimt::Hfp8ToFloatKernel>(
        AscendC::Simt::Dim3{tilingData.blockDim, 1, 1}, gmInput, gmOutput, tilingData.totalElems,
        static_cast<int32_t>(tilingData.ebits), static_cast<int32_t>(tilingData.exponent_bias));
}
