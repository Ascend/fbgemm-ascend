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

#ifndef LRU_CACHE_INSERT_BYTE_FBGEMM_PADDED_ROW_H
#define LRU_CACHE_INSERT_BYTE_FBGEMM_PADDED_ROW_H

#include <cstdint>

#include "cache_constants.h"

#ifndef __aicore__
#define __aicore__
#endif

namespace fbgemm_compat {

// SparseType (embedding_common.h)
constexpr uint8_t SPARSE_FP32 = 0;
constexpr uint8_t SPARSE_FP16 = 1;
constexpr uint8_t SPARSE_INT8 = 2;
constexpr uint8_t SPARSE_INT4 = 3;
constexpr uint8_t SPARSE_INT2 = 4;
constexpr uint8_t SPARSE_BF16 = 5;
constexpr uint8_t SPARSE_FP8 = 6;

__simt_callee__ inline uint32_t RoundUpU32(uint32_t a, uint32_t b)
{
    b == 0 ? b = 1 : b;
    return ((a + b - 1U) / b) * b;
}

__simt_callee__ inline int32_t UnpaddedRowSizeBytes(int32_t dim, uint8_t weightTy, int32_t scaleBiasBytes)
{
    if (weightTy == SPARSE_FP32) {
        return dim * 4;
    }
    if (weightTy == SPARSE_FP16 || weightTy == SPARSE_BF16) {
        return dim * 2;
    }
    if (weightTy == SPARSE_FP8) {
        return dim;
    }
    if (weightTy == SPARSE_INT8) {
        return dim + scaleBiasBytes;
    }
    if (weightTy == SPARSE_INT4) {
        return dim / 2 + scaleBiasBytes;
    }
    if (weightTy == SPARSE_INT2) {
        return dim / 4 + scaleBiasBytes;
    }
    return 0;
}

__simt_callee__ inline int32_t PaddedRowSizeBytes(
    int32_t dim, uint8_t weightTy, int32_t rowAlignment, int32_t scaleBiasBytes = kINT8QparamsBytes)
{
    int32_t r = UnpaddedRowSizeBytes(dim, weightTy, scaleBiasBytes);
    return static_cast<int32_t>(RoundUpU32(static_cast<uint32_t>(r), static_cast<uint32_t>(rowAlignment)));
}

}  // namespace fbgemm_compat

#endif
