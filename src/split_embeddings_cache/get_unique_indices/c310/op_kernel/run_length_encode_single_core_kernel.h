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

#ifndef RUN_LENGTH_ENCODE_SINGLE_CORE_KERNEL_H
#define RUN_LENGTH_ENCODE_SINGLE_CORE_KERNEL_H

#include "run_length_encode_helper.h"

constexpr int32_t BUFFER_NUM = 1;

template <typename T, bool COUNT_OUT>
class RunLengthEncodeSingleCoreKernel {
public:
    __aicore__ inline RunLengthEncodeSingleCoreKernel(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR count, GM_ADDR length, GM_ADDR shape_out,
                                GM_ADDR workspace, const RunLengthEncodeTilingData* tilingData)
    {
        (void)workspace;
        totalLength_ = static_cast<uint32_t>(tilingData->totalSize);
        blockSize_ = GetDataBlockSizeInBytes();

        xGm_.SetGlobalBuffer((__gm__ T*)(x));
        yGm_.SetGlobalBuffer((__gm__ T*)(y));
        countGm_.SetGlobalBuffer((__gm__ int32_t*)(count));
        lengthGm_.SetGlobalBuffer((__gm__ int32_t*)(length));
        shapeGm_.SetGlobalBuffer((__gm__ uint64_t*)shape_out);

        valueQueueSize_ = static_cast<uint32_t>(AscendC::CeilDivision(
                              static_cast<int32_t>(totalLength_ * sizeof(T)), static_cast<int32_t>(blockSize_))) *
                          blockSize_;
        pipe_->InitBuffer(xQueue_, BUFFER_NUM, valueQueueSize_);
        pipe_->InitBuffer(yQueue_, BUFFER_NUM, valueQueueSize_);

        if constexpr (COUNT_OUT) {
            pipe_->InitBuffer(countQueue_, BUFFER_NUM, totalLength_ * sizeof(int32_t));
        }

        pipe_->InitBuffer(idxBuf_,
                          static_cast<uint32_t>(AscendC::CeilDivision(
                              static_cast<int32_t>(totalLength_ * sizeof(int32_t)), static_cast<int32_t>(blockSize_))) *
                              blockSize_);
        pipe_->InitBuffer(lengthBuf_, blockSize_);
        pipe_->InitBuffer(shapeBuf_,
                          static_cast<uint32_t>(AscendC::CeilDivision(
                              static_cast<int32_t>(SHAPE_LEN * sizeof(uint64_t)), static_cast<int32_t>(blockSize_))) *
                              blockSize_);
    }

    __aicore__ inline void Process()
    {
        LocalTensor<uint64_t> shapeTensor = shapeBuf_.Get<uint64_t>();
        Duplicate(shapeTensor, (uint64_t)1, SHAPE_LEN);
        // 单核流程：整段 copy-in -> unique/value/count 计算 -> 结果一次性 copy-out。
        CopyInX();
        ComputeAndCopyOut();
    }

    __aicore__ inline void CopyInX()
    {
        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = totalLength_ * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        LocalTensor<T> xLocal = xQueue_.template AllocTensor<T>();
        DataCopyPad(xLocal, xGm_, dataCopyParams, padParams);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void ComputeAndCopyOut()
    {
        LocalTensor<int32_t> idxTensor = idxBuf_.Get<int32_t>();
        LocalTensor<T> xLocal = xQueue_.template DeQue<T>();
        LocalTensor<T> outTensor = yQueue_.template AllocTensor<T>();

        uint64_t reduceCntValue = 0;
        uint64_t reduceCntIdx = 0;

        // 先收集 unique 值，再按开关决定是否生成边界索引并转为 count。
        CollectPostUniqueValue<T, true>(outTensor, xLocal, totalLength_, reduceCntValue);

        if constexpr (COUNT_OUT) {
            CollectPostUniqueIdx<T, true>(idxTensor, xLocal, 1, totalLength_, totalLength_, reduceCntIdx,
                                          START_POSITION);
        }

        xQueue_.FreeTensor(xLocal);

        if constexpr (COUNT_OUT) {
            LocalTensor<int32_t> outCount = countQueue_.template AllocTensor<int32_t>();
            // 边界索引做相邻差分得到每段 count。
            PostAdjDiff<int32_t>(outCount, idxTensor, idxTensor.GetValue(0), reduceCntIdx, START_POSITION);
            countQueue_.EnQue(outCount);
        }

        yQueue_.EnQue(outTensor);

        CopyOut(reduceCntValue, reduceCntIdx);
        CopyOutLength(reduceCntValue);
        CopyOutShape(reduceCntValue, reduceCntIdx);
    }

    __aicore__ inline void CopyOut(int32_t copyLenValue, int32_t copyLenIdx)
    {
        DataCopyExtParams dataCopyParamsValue;
        dataCopyParamsValue.blockCount = 1;
        dataCopyParamsValue.blockLen = static_cast<uint32_t>(copyLenValue) * sizeof(T);
        dataCopyParamsValue.srcStride = 0;
        dataCopyParamsValue.dstStride = 0;

        LocalTensor<T> yLocal = yQueue_.template DeQue<T>();

        if constexpr (COUNT_OUT) {
            LocalTensor<int32_t> outCount = countQueue_.template DeQue<int32_t>();
            CpLocal2Gm<int32_t>(countGm_, outCount, copyLenIdx);
            countQueue_.FreeTensor(outCount);
        }

        DataCopyPad(yGm_, yLocal, dataCopyParamsValue);
        yQueue_.FreeTensor(yLocal);
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
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue_;

    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> countQueue_;

    TBuf<TPosition::VECCALC> idxBuf_;
    TBuf<TPosition::VECCALC> lengthBuf_;
    TBuf<TPosition::VECCALC> shapeBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> countGm_;
    GlobalTensor<int32_t> lengthGm_;
    GlobalTensor<uint64_t> shapeGm_;

    uint32_t totalLength_;
    uint32_t blockSize_;
    uint32_t valueQueueSize_;

    TPipe* pipe_ = nullptr;
};

#endif  // RUN_LENGTH_ENCODE_SINGLE_CORE_KERNEL_H
