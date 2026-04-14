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

#ifndef RUN_LENGTH_ENCODE_MULTI_CORE_KERNEL_H
#define RUN_LENGTH_ENCODE_MULTI_CORE_KERNEL_H

#include "run_length_encode_helper.h"

constexpr int64_t MAGIC_GM_PAGE_SIZE = 128;
constexpr int32_t BUFFER_NUM_MULTI = 1;

template <typename T, bool COUNT_OUT>
class RunLengthEncodeMultiCoreKernel {
public:
    __aicore__ inline RunLengthEncodeMultiCoreKernel(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR count, GM_ADDR length, GM_ADDR shape_out,
                                GM_ADDR workspace, const RunLengthEncodeTilingData* tilingData)
    {
        coreIdx_ = GetBlockIdx();
        isFinalCore_ = (coreIdx_ == GetBlockNum() - 1);

        coreTileLength_ = (isFinalCore_) ? tilingData->tileLengthTailCore : tilingData->tileLengthPerCore;
        totalSize_ = tilingData->totalSize;
        baseCount_ = 1 + coreIdx_ * tilingData->tileLengthPerCore;
        tileLengthPerCore_ = tilingData->tileLengthPerCore;
        adjUbTileLength_ = tilingData->adjUbTileLength;
        ubTileLength_ = adjUbTileLength_ - 1;

        xGm_.SetGlobalBuffer((__gm__ T*)(x) + tilingData->tileLengthPerCore * coreIdx_);
        yGm_.SetGlobalBuffer((__gm__ T*)(y));
        countGm_.SetGlobalBuffer((__gm__ int32_t*)(count));

        if (isFinalCore_) {
            lengthGm_.SetGlobalBuffer((__gm__ int32_t*)(length));
            shapeGm_.SetGlobalBuffer((__gm__ uint64_t*)shape_out);
        }

        pipe_->InitBuffer(xQueue_, BUFFER_NUM_MULTI, tilingData->valueQueueSize);
        pipe_->InitBuffer(yQueue_, BUFFER_NUM_MULTI, tilingData->valueQueueSize);

        pipe_->InitBuffer(countQueue_, BUFFER_NUM_MULTI, tilingData->countQueueSize);
        pipe_->InitBuffer(idxCopyInQueue_, BUFFER_NUM_MULTI, tilingData->idxCopyInQueueSize);

        pipe_->InitBuffer(collectingCntBuf_, tilingData->collectingCntBufSize);
        pipe_->InitBuffer(prevIdxBuf_, tilingData->prevIdxBufSize);
        pipe_->InitBuffer(lengthBuf_, 32);
        pipe_->InitBuffer(shapeBuf_, tilingData->shapeBufSize);

        collectNumGm_.SetGlobalBuffer((__gm__ int64_t*)(workspace));
        valueWorkspaceGm_.SetGlobalBuffer((__gm__ T*)(workspace + MAGIC_GM_PAGE_SIZE * GetBlockNum()) +
                                          coreIdx_ * tilingData->tileLengthPerCore);
        idxWorkspaceGm_.SetGlobalBuffer(
            (__gm__ int32_t*)(workspace + MAGIC_GM_PAGE_SIZE * GetBlockNum() + tilingData->totalSize * sizeof(T)) +
            coreIdx_ * tilingData->tileLengthPerCore);
        idxWorkspaceStartGm_.SetGlobalBuffer(
            (__gm__ int32_t*)(workspace + MAGIC_GM_PAGE_SIZE * GetBlockNum() + tilingData->totalSize * sizeof(T)));
    }

    __aicore__ inline void Process()
    {
        // 仅末核负责最终 shape 初始化与回写
        if (isFinalCore_) {
            LocalTensor<uint64_t> shapeTensor = shapeBuf_.Get<uint64_t>();
            Duplicate(shapeTensor, (uint64_t)1, SHAPE_LEN);
        }

        int64_t coreCollectNums = 0;
        // 阶段一：本核在自身分块内收集 unique 值（及可选索引）到 workspace
        ProcessCollecting(coreCollectNums);

        PipeBarrier<PIPE_ALL>();
        SyncAll();

        if (coreCollectNums == 0 && !isFinalCore_) {
            return;
        }

        int64_t yOffset = 0;
        int64_t prevTailCount = 0;
        // 根据前序核收集数量计算本核写回偏移，并获取上一个非空核的尾索引值。
        FindOffset(yOffset, prevTailCount);

        // 阶段二：将 workspace 中的收集结果按偏移归并到最终输出。
        ProcessMerging(yOffset, coreCollectNums, prevTailCount);

        if (isFinalCore_) {
            coreCollectNums += yOffset;
            CopyOutShape(coreCollectNums, coreCollectNums);
            CopyOutLength(coreCollectNums);
        }
    }

    __aicore__ inline void ProcessMerging(int64_t yOffset, int64_t collectNums, int64_t prevTailCount)
    {
        // 归并阶段按 ubTileLength 切块处理，先跑主块再处理尾块。
        int64_t mergingLoops = collectNums / ubTileLength_;
        int64_t mergingTails = collectNums % ubTileLength_;

        int64_t copyInOffset = 0;
        int64_t copyOutOffset = yOffset;

        for (int64_t i = 0; i < mergingLoops; ++i) {
            CopyInAndComputeCount(copyInOffset, copyOutOffset, ubTileLength_, prevTailCount);
            copyInOffset += ubTileLength_;
            copyOutOffset += ubTileLength_;
        }

        if (mergingTails > 0) {
            CopyInAndComputeCount(copyInOffset, copyOutOffset, mergingTails, prevTailCount);
        }
    }

    __aicore__ inline void ProcessCollecting(int64_t& coreCollectNums)
    {
        // 采集阶段：每次读取 ubTileLength_+1 个元素，利用“相邻比较”得到当前 tile 的 unique 边界。
        int64_t ubLoops = static_cast<int64_t>(
                              AscendC::CeilDivision(static_cast<int32_t>(coreTileLength_),
                                                    static_cast<int32_t>(ubTileLength_))) -
                          1;
        int64_t ubMainLength = ubTileLength_ + 1;
        int64_t ubTailLength = coreTileLength_ - ubTileLength_ * ubLoops;
        ubTailLength = (isFinalCore_) ? ubTailLength : ubTailLength + 1;

        int64_t offsetXGm = 0;
        int64_t gatherCnt = 0;
        int64_t innerBaseCount = baseCount_;

        for (int64_t i = 0; i < ubLoops; ++i) {
            CopyInX(offsetXGm, ubMainLength);
            CollectUniques<false>(innerBaseCount, ubMainLength, gatherCnt);
            CopyOutCollecteds2Worksapce(coreCollectNums, gatherCnt);
            offsetXGm += ubTileLength_;
            coreCollectNums += gatherCnt;
            innerBaseCount += ubTileLength_;
        }

        CopyInX(offsetXGm, ubTailLength);
        // 尾块在末核需要补齐最后一个 unique 元素，非末核保持与主块一致。
        if (isFinalCore_) {
            CollectUniques<true>(innerBaseCount, ubTailLength, gatherCnt);
        } else {
            CollectUniques<false>(innerBaseCount, ubTailLength, gatherCnt);
        }

        CopyOutCollecteds2Worksapce(coreCollectNums, gatherCnt);
        coreCollectNums += gatherCnt;

        CopyOutCnt2Workspace(coreCollectNums);
    }

    template <bool IS_TAIL_LOOP>
    __aicore__ inline void CollectUniques(int64_t innerBaseCount, int64_t nums, int64_t& gatherCnt)
    {
        LocalTensor<T> xLocal = xQueue_.template DeQue<T>();
        LocalTensor<T> outTensor = yQueue_.template AllocTensor<T>();
        uint64_t reduceCntValue = -1;
        CollectPostUniqueValue<T, IS_TAIL_LOOP>(outTensor, xLocal, nums, reduceCntValue);
        yQueue_.EnQue(outTensor);

        if constexpr (COUNT_OUT) {
            LocalTensor<int32_t> outCount = countQueue_.template AllocTensor<int32_t>();
            uint64_t reduceCntIdx = -1;
            CollectPostUniqueIdx<T, IS_TAIL_LOOP>(outCount, xLocal, innerBaseCount, totalSize_, nums, reduceCntIdx,
                                                  START_POSITION);
            countQueue_.EnQue(outCount);
        }

        gatherCnt = reduceCntValue;
        xQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyInX(int64_t offset, int64_t copyLen)
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(copyLen * sizeof(T)), 0, 0, 0};

        LocalTensor<T> xLocal = xQueue_.template AllocTensor<T>();
        DataCopyPad(xLocal, xGm_[offset], dataCopyParams, padParams);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyOutCollecteds2Worksapce(int64_t offset, int64_t copyLen)
    {
        LocalTensor<T> yLocal = yQueue_.template DeQue<T>();

        if constexpr (COUNT_OUT) {
            LocalTensor<int32_t> countLocal = countQueue_.template DeQue<int32_t>();
            CpLocal2Gm<int32_t>(idxWorkspaceGm_[offset], countLocal, copyLen);
            countQueue_.FreeTensor(countLocal);
        }

        CpLocal2Gm<T>(valueWorkspaceGm_[offset], yLocal, copyLen);
        yQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutCnt2Workspace(int64_t coreCollectNums)
    {
        LocalTensor<int64_t> countLocal = collectingCntBuf_.Get<int64_t>();
        countLocal.SetValue(0, coreCollectNums);
        SimpleNativePipeSync<HardEvent::S_MTE3>();
        CpLocal2Gm<int64_t>(collectNumGm_[(MAGIC_GM_PAGE_SIZE / sizeof(int64_t)) * coreIdx_], countLocal, 1);
    }

    __aicore__ inline void CopyInCounts(LocalTensor<int64_t>& countLocal)
    {
        DataCopyPadExtParams<int64_t> padParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = coreIdx_ + 1;
        dataCopyParams.blockLen = sizeof(int64_t);
        dataCopyParams.srcStride = MAGIC_GM_PAGE_SIZE - sizeof(int64_t);
        dataCopyParams.dstStride = 0;

        DataCopyPad<int64_t, PaddingMode::Compact>(countLocal, collectNumGm_, dataCopyParams, padParams);
    }

    __aicore__ inline void FindOffset(int64_t& yOffset, int64_t& prevTail)
    {
        LocalTensor<int64_t> countLocal = collectingCntBuf_.Get<int64_t>();
        CopyInCounts(countLocal);
        SimpleNativePipeSync<HardEvent::MTE2_S>();
        // 使用确定性的标量前缀和，避免 0 长度向量归约的边界问题。
        yOffset = 0;
        for (int64_t i = 0; i < coreIdx_; ++i) {
            yOffset += countLocal.GetValue(i);
        }

        if constexpr (COUNT_OUT) {
            bool first = true;
            for (int i = coreIdx_ - 1; i >= 0; i--) {
                int64_t coreCount = countLocal.GetValue(i);
                if (coreCount != 0) {
                    first = false;
                    CopyInCoreFinal(i, coreCount, prevTail);
                    break;
                }
            }
            if (first) {
                prevTail = 0;
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyInCoreFinal(int64_t prevCoreIdx, int64_t prevCoreCount, int64_t& prevTail)
    {
        int64_t offset = prevCoreIdx * tileLengthPerCore_ + prevCoreCount - 1;
        LocalTensor<int32_t> prevIdxLocal = prevIdxBuf_.Get<int32_t>();
        DataCopyPrevIdx(prevIdxLocal, offset);
        SimpleNativePipeSync<HardEvent::MTE2_S>();
        prevTail = static_cast<int64_t>(prevIdxLocal.GetValue(0));
    }

    __aicore__ inline void DataCopyPrevIdx(LocalTensor<int32_t> prevIdxLocal, int64_t offset)
    {
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = sizeof(int32_t);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(prevIdxLocal, idxWorkspaceStartGm_[offset], dataCopyParams, padParams);
    }

    __aicore__ inline void CopyInAndComputeCount(int64_t wsOffset, int64_t yOffset, int64_t nums,
                                                 int64_t& prevTailCount)
    {
        LocalTensor<T> xLocal = xQueue_.template AllocTensor<T>();
        CopyInUniqueValues(xLocal, wsOffset, nums);

        if constexpr (COUNT_OUT) {
            // 首个 count 需要扣除上一核尾值，保证跨核拼接后计数连续正确。
            CopyInUniqueIdx(wsOffset, nums);
            LocalTensor<int32_t> idxLocal = idxCopyInQueue_.template DeQue<int32_t>();
            LocalTensor<int32_t> outCount = countQueue_.template AllocTensor<int32_t>();
            int32_t firstValue = idxLocal.GetValue(0) - prevTailCount;
            prevTailCount = idxLocal.GetValue(nums - 1);
            PostAdjDiff<int32_t>(outCount, idxLocal, firstValue, nums, START_POSITION);
            idxCopyInQueue_.FreeTensor(idxLocal);
            countQueue_.EnQue(outCount);
            CopyOutCount(yOffset, nums);
        }

        SimpleNativePipeSync<HardEvent::MTE2_MTE3>();
        CpLocal2Gm<T>(yGm_[yOffset], xLocal, nums);
        SimpleNativePipeSync<HardEvent::MTE3_MTE2>();
        xQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutCount(int64_t offset, int64_t copyLen)
    {
        LocalTensor<int32_t> outCount = countQueue_.template DeQue<int32_t>();
        CpLocal2Gm<int32_t>(countGm_[offset], outCount, copyLen);
        countQueue_.FreeTensor(outCount);
    }

    __aicore__ inline void CopyInUniqueValues(LocalTensor<T>& xLocal, int64_t offset, int64_t nums)
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = nums * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(xLocal, valueWorkspaceGm_[offset], dataCopyParams, padParams);
    }

    __aicore__ inline void CopyInUniqueIdx(int64_t offset, int64_t nums)
    {
        LocalTensor<int32_t> idxLocal = idxCopyInQueue_.template AllocTensor<int32_t>();
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = nums * sizeof(int32_t);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(idxLocal, idxWorkspaceGm_[offset], dataCopyParams, padParams);
        idxCopyInQueue_.EnQue(idxLocal);
        SimpleNativePipeSync<HardEvent::MTE2_S>();
    }

    __aicore__ inline void CopyOutLength(uint64_t uniqueNums)
    {
        LocalTensor<int32_t> lengthLocal = lengthBuf_.Get<int32_t>();
        lengthLocal.SetValue(0, static_cast<int32_t>(uniqueNums));
        SimpleNativePipeSync<HardEvent::S_MTE3>();
        CpLocal2Gm<int32_t>(lengthGm_, lengthLocal, 1);
    }

    __aicore__ inline void CopyOutShape(uint64_t dimNumValue, uint64_t dimNumIdx)
    {
        LocalTensor<uint64_t> shapeTensor = shapeBuf_.Get<uint64_t>();

        shapeTensor.SetValue(SHAPE0_SIZE_IDX, UINT64_SHAPE_DIM_ONE);
        shapeTensor.SetValue(SHAPE0_DIM0_IDX, dimNumValue);

        shapeTensor.SetValue(SHAPE1_SIZE_IDX, UINT64_SHAPE_DIM_ONE);
        if constexpr (COUNT_OUT) {
            shapeTensor.SetValue(SHAPE1_DIM0_IDX, dimNumIdx);
        } else {
            shapeTensor.SetValue(SHAPE1_DIM0_IDX, dimNumValue);
        }

        shapeTensor.SetValue(SHAPE2_SIZE_IDX, 1);
        shapeTensor.SetValue(SHAPE2_DIM0_IDX, 1);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = SHAPE_LEN * sizeof(uint64_t);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        SimpleNativePipeSync<HardEvent::S_MTE3>();
        DataCopyPad(shapeGm_, shapeTensor, dataCopyParams);
    }

private:
    TQue<QuePosition::VECIN, BUFFER_NUM_MULTI> xQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM_MULTI> idxCopyInQueue_;

    TQue<QuePosition::VECOUT, BUFFER_NUM_MULTI> yQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_MULTI> countQueue_;

    TBuf<TPosition::VECCALC> collectingCntBuf_;
    TBuf<TPosition::VECCALC> prevIdxBuf_;
    TBuf<TPosition::VECCALC> lengthBuf_;
    TBuf<TPosition::VECCALC> shapeBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> countGm_;
    GlobalTensor<int32_t> lengthGm_;
    GlobalTensor<uint64_t> shapeGm_;

    GlobalTensor<int64_t> collectNumGm_;
    GlobalTensor<T> valueWorkspaceGm_;
    GlobalTensor<int32_t> idxWorkspaceGm_;
    GlobalTensor<int32_t> idxWorkspaceStartGm_;

    int64_t ubTileLength_;
    int64_t adjUbTileLength_;
    int64_t coreIdx_;
    bool isFinalCore_;
    int64_t coreTileLength_;
    int64_t totalSize_;
    int64_t baseCount_;

    int64_t tileLengthPerCore_;
    TPipe* pipe_ = nullptr;
};

#endif  // RUN_LENGTH_ENCODE_MULTI_CORE_KERNEL_H
