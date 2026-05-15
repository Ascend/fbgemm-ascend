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

namespace FloatToHfp8QuantizedSimt
{

union fint32
{
    float F;
    uint32_t I;
};

__simt_callee__ inline uint8_t float_to_hfp8(float val_fp, int32_t ebits, int32_t exponent_bias, float max_pos)
{
    int32_t mbits = 7 - ebits;
    fint32 val_out, bouncer, smallest_normal;

    val_out.F = val_fp;
    uint32_t sign_bit = val_out.I & 0x80000000U;
    val_out.I = val_out.I & 0x7FFFFFFFU;
    val_out.F = (val_out.F < max_pos) ? val_out.F : max_pos;

    smallest_normal.I = static_cast<uint32_t>(127 - exponent_bias + 1) << 23;

    if (val_out.F >= smallest_normal.F)
    {
        bouncer.I = (val_out.I & 0xFF800000U) + (static_cast<uint32_t>(23 - mbits) << 23);
        val_out.F = (bouncer.F + val_out.F) - bouncer.F;
        val_out.I = (val_out.I - (static_cast<uint32_t>(127 - exponent_bias) << 23)) << (8 - ebits);
        val_out.I = ((val_out.I | sign_bit) >> 24);
    }
    else
    {
        bouncer.I = (static_cast<uint32_t>(127 + (23 + (1 - exponent_bias - mbits))) << 23);
        val_out.F = bouncer.F + val_out.F;
        val_out.I = val_out.I | (sign_bit >> 24);
    }

    return static_cast<uint8_t>(val_out.I);
}

__simt_vf__ __aicore__ inline void FloatToHFP8Kernel(__gm__ float* input, __gm__ uint8_t* output, int64_t totalElems,
                                                     int32_t ebits, int32_t exponent_bias, float max_pos)
{
    const int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    const int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    const int32_t blockNum = AscendC::Simt::GetBlockNum();
    const int32_t thdPerBlock = blockDim.x;
    const int32_t gridSize = blockNum * thdPerBlock;

    int64_t idx = static_cast<int64_t>(blockIdx) * thdPerBlock + tid;
    bool useVectorized = (totalElems % 4 == 0);
    int64_t vecElems = totalElems / 4;

    if (useVectorized)
    {
        const __gm__ float4* in4 = reinterpret_cast<const __gm__ float4*>(input);
        __gm__ uint32_t* out32 = reinterpret_cast<__gm__ uint32_t*>(output);
        for (int64_t i = idx; i < vecElems; i += gridSize)
        {
            float4 v = in4[i];
            uint8_t b0 = float_to_hfp8(v.x, ebits, exponent_bias, max_pos);
            uint8_t b1 = float_to_hfp8(v.y, ebits, exponent_bias, max_pos);
            uint8_t b2 = float_to_hfp8(v.z, ebits, exponent_bias, max_pos);
            uint8_t b3 = float_to_hfp8(v.w, ebits, exponent_bias, max_pos);
            out32[i] = static_cast<uint32_t>(b0) | (static_cast<uint32_t>(b1) << 8) |
                       (static_cast<uint32_t>(b2) << 16) | (static_cast<uint32_t>(b3) << 24);
        }
    }

    int64_t scalarStart = useVectorized ? (vecElems * 4) : 0;
    for (int64_t i = scalarStart + idx; i < totalElems; i += gridSize)
    {
        output[i] = float_to_hfp8(input[i], ebits, exponent_bias, max_pos);
    }
}

}  // namespace FloatToHfp8QuantizedSimt

extern "C" __global__ __aicore__ void float_to_hfp8_quantized(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                                              GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    __gm__ float* gmInput = reinterpret_cast<__gm__ float*>(input);
    __gm__ uint8_t* gmOutput = reinterpret_cast<__gm__ uint8_t*>(output);

    AscendC::Simt::VF_CALL<FloatToHfp8QuantizedSimt::FloatToHFP8Kernel>(
        AscendC::Simt::Dim3{tilingData.blockDim, 1, 1}, gmInput, gmOutput, tilingData.totalElems,
        static_cast<int32_t>(tilingData.ebits), static_cast<int32_t>(tilingData.exponent_bias), tilingData.max_pos);
}
