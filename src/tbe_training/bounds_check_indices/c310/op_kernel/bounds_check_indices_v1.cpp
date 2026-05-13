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

#include "bounds_check_indices_common.h"

namespace {
constexpr int32_t kWarpSize = 32;
constexpr int32_t kNumThreads = 256;
}

#define DISPATCH_BOUNDS_CHECK(indiceType, vbe, mode)                         \
    AscendC::Simt::VF_CALL<BoundsCheckIndicesV1Impl<indiceType, vbe, mode>>( \
        AscendC::Simt::Dim3{kWarpSize, kNumThreads / kWarpSize, 1},          \
        reinterpret_cast<__gm__ int64_t*>(rowsPerTable),                     \
        reinterpret_cast<__gm__ indiceType*>(indices),                       \
        reinterpret_cast<__gm__ indiceType*>(offsets),                       \
        reinterpret_cast<__gm__ int64_t*>(warning),                          \
        reinterpret_cast<__gm__ int32_t*>(bOffsets),                         \
        static_cast<indiceType>(tilingData.numIndices),                      \
        tilingData.numTables,                                                \
        tilingData.batchSize,                                                \
        tilingData.totalB,                                                   \
        tilingData.batchSizeDivMagic,                                        \
        tilingData.batchSizeDivShift)

template <typename indiceType, bool vbe, BoundsCheckMode mode>
__simt_vf__ __aicore__ LAUNCH_BOUND(kNumThreads) inline void BoundsCheckIndicesV1Impl(
    __gm__ int64_t* rowsPerTable,
    __gm__ indiceType* indices,
    __gm__ indiceType* offsets,
    __gm__ int64_t* warning,
    __gm__ int32_t* bOffsets,
    indiceType numIndices,
    int32_t numTables,
    int32_t batchSize,
    int32_t totalB,
    uint32_t batchSizeDivMagic,
    uint32_t batchSizeDivShift)
{
    int32_t bTIdx = blockIdx.x * blockDim.y + threadIdx.y;
    if (!vbe && bTIdx >= totalB) {
        return;
    }

    const FastDivmod<uint32_t> fd(batchSizeDivMagic, batchSizeDivShift, static_cast<uint32_t>(batchSize));
    int32_t tIdx = fd.Div(bTIdx);
    int32_t bIdx = fd.Mod(bTIdx);
    if (vbe) {
        if (tIdx >= numTables) {
            return;
        }
        int32_t bStart = bOffsets[tIdx];
        int32_t bEnd = bOffsets[tIdx + 1];
        batchSize = bEnd - bStart;
        if (bIdx >= batchSize) {
            return;
        }
        bTIdx = bStart + bIdx;
    }

    int64_t numRows = rowsPerTable[tIdx];
    indiceType indiceStart = offsets[bTIdx];
    indiceType indiceEnd = offsets[bTIdx + 1];

    if (mode == BoundsCheckMode::FATAL) {
        BOUNDS_ASSERT(indiceStart >= 0);
        BOUNDS_ASSERT(indiceStart <= indiceEnd);
        BOUNDS_ASSERT(indiceEnd <= numIndices);
    } else if (mode == BoundsCheckMode::WARNING) {
        if (indiceStart < 0 || indiceStart > indiceEnd || indiceEnd > numIndices) {
            if (asc_atomic_add(&warning[0], 1) == 0) {
                AscendC::Simt::printf("EmbeddingBoundsCheck (VBE %u): (at least one) Out of bounds access for "
                                      "batch: %d, table: %d, indices_start: %lld, indices_end: %lld,"
                                      " num_indices: %lld. Setting indices_start and indices_end within the range.\n",
                                      vbe, bIdx, tIdx, static_cast<int64_t>(indiceStart), static_cast<int64_t>(indiceEnd), static_cast<int64_t>(numIndices));
            }
            AdjustOffset(indiceStart, indiceEnd, numIndices, &offsets[bTIdx], &offsets[bTIdx + 1]);
        }
    } else if (mode == BoundsCheckMode::IGNORE) {
        AdjustOffset(indiceStart, indiceEnd, numIndices, &offsets[bTIdx], &offsets[bTIdx + 1]);
    }

    int32_t bagSize = indiceEnd - indiceStart;
    for (int32_t i = threadIdx.x; i < bagSize; i += kWarpSize) {
        indiceType idx = indices[indiceStart + i];

        if (idx == -1) {
            continue;
        }

        if (mode == BoundsCheckMode::FATAL) {
            BOUNDS_ASSERT(idx >= 0);
            BOUNDS_ASSERT(idx < numRows);
        } else if (mode == BoundsCheckMode::WARNING) {
            if (idx < 0 || idx >= numRows) {
                if (asc_atomic_add(&warning[0], 1) == 0) {
                    AscendC::Simt::printf("EmbeddingBoundsCheck (VBE %u): (at least one) Out of bounds access for batch: %d, table: %d, "
                                          "bag element: %lld, idx: %lld, num_rows: %lld, indices_start: %lld, indices_end: %lld, T: %d, B: %d, b_t: %d. Setting idx to zero.\n",
                                          vbe, bIdx, tIdx, static_cast<int64_t>(i), static_cast<int64_t>(idx), numRows, static_cast<int64_t>(indiceStart),
                                          static_cast<int64_t>(indiceEnd), numTables, batchSize, bTIdx);
                }
                indices[indiceStart + i] = 0;
            }
        } else if (mode == BoundsCheckMode::IGNORE) {
            if (idx < 0 || idx >= numRows) {
                indices[indiceStart + i] = 0;
            }
        }
    }

    if (mode == BoundsCheckMode::FATAL) {
        BOUNDS_ASSERT(offsets[totalB] == numIndices);
    } else if (mode == BoundsCheckMode::WARNING) {
        if (offsets[totalB] != numIndices) {
            if (asc_atomic_add(&warning[0], 1) == 0) {
                AscendC::Simt::printf("EmbeddingBoundsCheck (VBE %u): the last element in offsets is incorrect for "
                                        "total batch size total_B: %d, total table num T: %d, last element in offsets: %lld, indices size: %lld. "
                                        " Setting the last element in offsets to be indices size.\n",
                                        vbe, totalB, numTables, static_cast<int64_t>(offsets[totalB]), static_cast<int64_t>(numIndices));
            }
            offsets[totalB] = numIndices;
        }
    } else if (mode == BoundsCheckMode::IGNORE) {
        if (offsets[totalB] != numIndices) {
            offsets[totalB] = numIndices;
        }
    }
}

extern "C" __global__ __aicore__ void bounds_check_indices_v1(
    GM_ADDR rowsPerTable,
    GM_ADDR indices,
    GM_ADDR offsets,
    GM_ADDR warning,
    GM_ADDR bOffsets,
    GM_ADDR indicesOut,
    GM_ADDR offsetsOut,
    GM_ADDR warningOut,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    bool vbe = (tilingData.vbe != 0);
    BoundsCheckMode mode = static_cast<BoundsCheckMode>(tilingData.boundsCheckMode);

    INVOKE_BOUNDS_CHECK(DISPATCH_BOUNDS_CHECK, DTYPE_INDICES, vbe, mode)
}

#undef DISPATCH_BOUNDS_CHECK