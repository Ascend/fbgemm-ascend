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

#ifndef GROUP_INDEX_SELECT_DIM0_BACKWARD_FUN_H
#define GROUP_INDEX_SELECT_DIM0_BACKWARD_FUN_H

#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

namespace GroupIndexSelectDim0Backward {

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
    GM_ADDR gradOutputs;
    GM_ADDR indicesGroups;
    GM_ADDR inputReturnGroups;

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
class GroupIndexSelectDim0BackwardKernel {
public:
    __aicore__ inline GroupIndexSelectDim0BackwardKernel(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);

        gradPtr = args.gradOutputs;
        indicesPtr = args.indicesGroups;
        inputReturnPtr = args.inputReturnGroups;

        groupNum =  tilingData.groupNum;
        for (int32_t i = 0; i < groupNum; i++) {
            groupIndicesLen[i] = tilingData.groupIndicesLen[i];
            groupInnerDim[i] = tilingData.groupInnerDim[i];
            groupGradRows[i] = tilingData.groupGradRows[i];
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
        uint32_t coreTaskNum = this->groupGradRows[runArgs.curGroupIdx];
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

        for (uint32_t i = 0; i < batchSize; i++) {
            LocalTensor<int64_t> indicesLt = indicesQue.AllocTensor<int64_t>();
            CpGm2Local(indicesLt, curIndicesGt[batchOffset], batchSize);
            indicesQue.EnQue(indicesLt);
            indicesLt = indicesQue.DeQue<int64_t>();

            LocalTensor<inputType> gatherDataLt = gatherDataQue.template AllocTensor<inputType>();
            Duplicate(gatherDataLt, static_cast<inputType>(0), gatherDataLt.GetSize());
            int64_t rowIdx = indicesLt.GetValue(i);

            CpGm2Local(gatherDataLt, curGradGt[(i + batchOffset) * innerDim], innerDim);

            gatherDataQue.EnQue(gatherDataLt);
            gatherDataLt = gatherDataQue.DeQue<inputType>();
            pipe_barrier(PIPE_ALL);

            SetAtomicAdd<inputType>();
            CpLocal2Gm(curInputReturnGt[(rowIdx) * innerDim], gatherDataLt, innerDim);
            SetAtomicNone();
            pipe_barrier(PIPE_ALL);

            gatherDataQue.FreeTensor(gatherDataLt);
            indicesQue.FreeTensor<int64_t>(indicesLt);
        }
    }

    __aicore__ inline void ProcessGroup(uint32_t groupIdx)
    {
        RunArgs runArgs;
        runArgs.curGroupIdx = groupIdx;

        curGradGt.SetGlobalBuffer((__gm__ inputType *)GetTensorAddr(gradPtr, groupIdx));
        curInputReturnGt.SetGlobalBuffer((__gm__ inputType *)GetTensorAddr(inputReturnPtr, groupIdx));
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
    GM_ADDR gradPtr;
    GM_ADDR indicesPtr;
    GM_ADDR inputReturnPtr;

    // Tiling
    int32_t groupNum;
    int32_t groupIndicesLen[MAX_NUM_GROUPS];
    int32_t groupInnerDim[MAX_NUM_GROUPS];
    int32_t groupGradRows[MAX_NUM_GROUPS];

    // Gt
    GlobalTensor<inputType> curGradGt;
    GlobalTensor<int64_t> curIndicesGt;
    GlobalTensor<inputType> curInputReturnGt;

    // TPipe
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> indicesQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> indexInQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> gatherDataQue;
};
}
#endif
