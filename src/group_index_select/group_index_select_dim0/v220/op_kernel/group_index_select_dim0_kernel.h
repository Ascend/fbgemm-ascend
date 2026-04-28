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

#ifndef GROUP_INDEX_SELECT_DIM0_FUN_H
#define GROUP_INDEX_SELECT_DIM0_FUN_H

#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

namespace GroupIndexSelectDim0 {

constexpr uint32_t BASIC_BLOCK = 32;
constexpr int DATA_ALIGN_BYTES = 32;
constexpr uint32_t KBYTES = 1024;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t GATHER_DATA_UB_BYTE_SIZE = (156 * KBYTES) / BUFFER_NUM;
constexpr uint32_t INDICES_UB_BYTE_SIZE = (30 * KBYTES) / BUFFER_NUM;
constexpr int DATA_COPY_PAD_ALIGN_BYTE2 = 16;
constexpr int DATA_COPY_PAD_ALIGN_BYTE4 = 8;
constexpr uint32_t MAX_NUM_GROUPS = 32;
constexpr int ONEBLOCK_ELEM = 4096;

struct Args {
    GM_ADDR inputGroups;
    GM_ADDR indicesGroups;
    GM_ADDR outputGroups;

    GM_ADDR workspace;
    GM_ADDR tiling;
};

struct RunArgs {
    uint32_t curGroupIdx;
    uint32_t batchCnt;
    uint32_t batchOffset;
    uint32_t batchSize;
    uint32_t batchTailSize;
};

template <typename T>
__aicore__ constexpr T AlignDown(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 / num2) * num2;
}

template <typename T>
__aicore__ inline void CpGm2Local(const LocalTensor<T>& lt, const GlobalTensor<T>& gt, int64_t len) {
    constexpr int DATA_ALIGN_BYTES = 32;
    uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
    uint32_t unAlignLen = len * sizeof(T) - alignLen;

    DataCopy(lt, gt, alignLen / sizeof(T));
    if (unAlignLen != 0) {
        const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
        const DataCopyPadExtParams<T> dataCopyPadExtParams{false, 0, 0, 0};
        DataCopyPad(lt[alignLen / sizeof(T)], gt[alignLen / sizeof(T)], dataCopyExtParams, dataCopyPadExtParams);
    }
}

template <typename T>
__aicore__ inline void CpLocal2Gm(const GlobalTensor<T>& gt, const LocalTensor<T>& lt, int64_t len) {
    constexpr int DATA_ALIGN_BYTES = 32;
    uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
    uint32_t unAlignLen = len * sizeof(T) - alignLen;
    DataCopy(gt, lt, alignLen / sizeof(T));
    if (unAlignLen != 0) {
        const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
        DataCopyPad(gt[alignLen / sizeof(T)], lt[alignLen / sizeof(T)], dataCopyExtParams);
    }
}

template <typename T>
__aicore__ constexpr T MinValue(T t)
{
    return t;
}

template <typename T, typename ...Args>
__aicore__ constexpr T MinValue(T t, Args... args)
{
    T minValue = MinValue(args...);
    return t < minValue ? t : minValue;
}

template <typename inputType>
class GroupIndexSelectDim0Kernel {
public:
    __aicore__ inline GroupIndexSelectDim0Kernel(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);

        inputPtr = args.inputGroups;
        indicesPtr = args.indicesGroups;
        outputPtr = args.outputGroups;

        groupNum =  tilingData.groupNum;
        for (int32_t i = 0; i < groupNum; i++) {
            groupIndicesLen[i] = tilingData.groupIndicesLen[i];
            groupInnerDim[i] = tilingData.groupInnerDim[i];
            groupInputRows[i] = tilingData.groupInputRows[i];
        }

        pipe.InitBuffer(gatherDataQue, BUFFER_NUM, GATHER_DATA_UB_BYTE_SIZE);
        pipe.InitBuffer(indicesQue, BUFFER_NUM, INDICES_UB_BYTE_SIZE);
    }

    __aicore__ inline void CoreSplitPolicy(uint32_t coreTaskNum, uint32_t& coreProcSize, uint32_t& coreProcOffset)
    {
        auto coreIdx = GetBlockIdx();
        auto coreNum = GetBlockNum();
        uint32_t actualCoreNum = (coreTaskNum >= coreNum) ? coreNum : coreTaskNum;
        uint32_t splitNextCoreProcTask = coreTaskNum / actualCoreNum;
        uint32_t splitPrevCoreProcTask = splitNextCoreProcTask + 1;
        uint32_t splitCoreIdx = coreTaskNum % actualCoreNum;

        if (coreIdx < splitCoreIdx) {
            coreProcSize = splitPrevCoreProcTask;
            coreProcOffset = coreIdx * splitPrevCoreProcTask;
        } else if (coreIdx < actualCoreNum) {
            coreProcSize = splitNextCoreProcTask;
            coreProcOffset = splitCoreIdx * splitPrevCoreProcTask + (coreIdx - splitCoreIdx) * splitNextCoreProcTask;
        } else {
            coreProcSize = 0;
            coreProcOffset = 0;
        }
    }

    __aicore__ inline __gm__ inputType *GetTensorAddr(GM_ADDR tensorList, uint32_t index)
    {
        __gm__ uint64_t *dataAddr = reinterpret_cast<__gm__ uint64_t *>(tensorList);
        uint64_t tensorPtrOffset = *dataAddr;

        __gm__ uint64_t *tensorPtr = dataAddr + (tensorPtrOffset >> 3);
        return reinterpret_cast<__gm__ inputType *>(*(tensorPtr + index));
    }

    __aicore__ inline void calBatch(RunArgs& runArgs)
    {
        uint32_t coreProcSize = 0;
        uint32_t coreProcOffset = 0;
        uint32_t coreTaskNum = this->groupIndicesLen[runArgs.curGroupIdx];
        int32_t groupInnerDim = this->groupInnerDim[runArgs.curGroupIdx];
        CoreSplitPolicy(coreTaskNum, coreProcSize, coreProcOffset);

        uint32_t maxIndicesBatchSize = AlignDown(static_cast<uint32_t>(INDICES_UB_BYTE_SIZE / sizeof(uint32_t)), BASIC_BLOCK);
        uint32_t gatherDataBatchSize = AlignDown(static_cast<uint32_t>(GATHER_DATA_UB_BYTE_SIZE / (sizeof(inputType) * groupInnerDim)), BASIC_BLOCK);
        uint32_t batchSize = MinValue(gatherDataBatchSize, maxIndicesBatchSize, coreProcSize);

        runArgs.batchSize = batchSize;
        runArgs.batchOffset = coreProcOffset;
        if (batchSize != 0) {
            runArgs.batchCnt = coreProcSize / batchSize;
            runArgs.batchTailSize = coreProcSize % batchSize;
        } else {
            runArgs.batchCnt = 0;
            runArgs.batchTailSize = 0;
        }
    }

    __aicore__ inline void ProcBatch(const RunArgs& runArgs, bool isTail)
    {
        uint32_t batchSize = (isTail ? runArgs.batchTailSize : runArgs.batchSize);
        uint32_t groupIdx = runArgs.curGroupIdx;
        uint32_t batchOffset = runArgs.batchOffset;

        int32_t innerDim = groupInnerDim[groupIdx];

        if (batchSize == 0) {
            return;
        }

        LocalTensor<int64_t> indexLt = indicesQue.AllocTensor<int64_t>();

        CpGm2Local(indexLt, curIndicesGt[batchOffset], batchSize);

        indicesQue.EnQue(indexLt);
        indexLt = indicesQue.DeQue<int64_t>();
        pipe_barrier(PIPE_ALL);

        LocalTensor<inputType> gatherDataLt = gatherDataQue.AllocTensor<inputType>();

        for (uint32_t i = 0; i < batchSize; i++) {
            int32_t rowIdx = indexLt.GetValue(i);
            if (rowIdx >= 0 && rowIdx < groupInputRows[groupIdx]) {
                CpGm2Local(gatherDataLt[i * innerDim], curInputGt[rowIdx * innerDim], innerDim);
            } else {
                Duplicate(gatherDataLt[i * innerDim], static_cast<inputType>(0), innerDim);
            }
        }

        indicesQue.FreeTensor(indexLt);

        gatherDataQue.EnQue(gatherDataLt);
        gatherDataLt = gatherDataQue.DeQue<inputType>();
        pipe_barrier(PIPE_ALL);

        CpLocal2Gm(curOutputGt[batchOffset * innerDim], gatherDataLt, batchSize * innerDim);
        gatherDataQue.FreeTensor(gatherDataLt);
    }

    __aicore__ inline void ProcessGroup(uint32_t groupIdx)
    {
        RunArgs runArgs;
        runArgs.curGroupIdx = groupIdx;

        curInputGt.SetGlobalBuffer((__gm__ inputType *)GetTensorAddr(inputPtr, groupIdx));
        curOutputGt.SetGlobalBuffer((__gm__ inputType *)GetTensorAddr(outputPtr, groupIdx));
        curIndicesGt.SetGlobalBuffer((__gm__ int64_t *)GetTensorAddr(indicesPtr, groupIdx));

        calBatch(runArgs);

        for (auto i = 0; i < runArgs.batchCnt; i++) {
            ProcBatch(runArgs, false);
            runArgs.batchOffset += runArgs.batchSize;
        }
        
        if (runArgs.batchTailSize != 0) {
            ProcBatch(runArgs, true);
        }
    }

    __aicore__ inline void Compute()
    {
        for (int32_t i = 0; i < groupNum; i++) {
            ProcessGroup(i);
        }
    }

private:
    GM_ADDR inputPtr;
    GM_ADDR indicesPtr;
    GM_ADDR outputPtr;

    // Tiling
    int32_t groupNum;
    int32_t groupIndicesLen[MAX_NUM_GROUPS];
    int32_t groupInnerDim[MAX_NUM_GROUPS];
    int32_t groupInputRows[MAX_NUM_GROUPS];

    // Gt
    GlobalTensor<inputType> curInputGt;
    GlobalTensor<int64_t> curIndicesGt;
    GlobalTensor<inputType> curOutputGt;

    // LocalTensors
    LocalTensor<int64_t> indicesLt;
    LocalTensor<inputType> gatherDataLt;

    // TPipe
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> indicesQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> indexInQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> gatherDataQue;
};
}
#endif
