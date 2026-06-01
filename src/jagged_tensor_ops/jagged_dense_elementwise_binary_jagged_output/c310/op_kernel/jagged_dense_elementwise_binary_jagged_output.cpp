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

#include <cstdint>
#include "kernel_common_utils.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

using namespace AscendC;

namespace JaggedDenseEwBinaryJOut_Kernel {
constexpr int32_t ALIGN_32 = 32;
constexpr int32_t MODE_ADD = 0;
constexpr int32_t MODE_MUL = 1;
constexpr int32_t MAX_OFFSETS_CNT = 5;

struct JaggedDenseEwBinaryArgs {
    GM_ADDR xValues;
    GM_ADDR dense;
    GM_ADDR offsets;
    GM_ADDR out;
    int32_t denseDim1;
    int32_t denseDim2;
    int32_t singleLoopSize;
    int64_t denseTotal;
    int64_t jaggedTotal;
    int32_t elementwiseMode;
    int32_t offsetCnt;
    int64_t maxLengths[MAX_OFFSETS_CNT];
    int64_t offsetsLens[MAX_OFFSETS_CNT];
};

template <typename dType, typename tType>
class JaggedDenseElementwiseBinaryJaggedOutput {
public:
    __aicore__ inline JaggedDenseElementwiseBinaryJaggedOutput(){};

    // 初始化GM地址、offsets列表、UB队列，并按row维度划分core任务。
    __aicore__ inline void init(JaggedDenseEwBinaryArgs* args, TPipe* pipe)
    {
        this->args = args;
        this->pipe = pipe;
        thisId = GetBlockIdx();
        align = static_cast<int32_t>(sizeof(dType));
        totalRows = args->jaggedTotal / args->denseDim2;
        int64_t coreRows = totalRows / GetBlockNum();
        int64_t tail = totalRows % GetBlockNum();
        if (thisId < tail) {
            lenOfThisCore = coreRows + 1;
            startOfThisCore = thisId * (coreRows + 1);
        } else {
            lenOfThisCore = coreRows;
            startOfThisCore = tail * (coreRows + 1) + (thisId - tail) * coreRows;
        }
        denseGb.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args->dense), args->denseTotal * align);
        xGb.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args->xValues), args->jaggedTotal * align);
        outGb.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args->out), args->jaggedTotal * align);
        AscendC::ListTensorDesc offsetsDesc;
        offsetsDesc.Init(args->offsets);
        for (int32_t i = 0; i < args->offsetCnt; ++i) {
            GM_ADDR offset = reinterpret_cast<__gm__ uint8_t*>(offsetsDesc.GetDataPtr<__gm__ uint8_t>(i));
            offsetsGb[i].SetGlobalBuffer((__gm__ tType*)offset, args->offsetsLens[i]);
        }
        pipe->InitBuffer(denseQueue, 1, args->singleLoopSize);
        pipe->InitBuffer(xQueue, 1, args->singleLoopSize);
        pipe->InitBuffer(outQueue, 1, args->singleLoopSize);
    }

    __aicore__ inline void Compute()
    {
        int64_t end = startOfThisCore + lenOfThisCore;
        for (int64_t row = startOfThisCore; row < end; ++row) {
            // dense tensor除了D维之外的线性位置，即denseTensor.view(-1,D)
            int64_t denseLinear = 0;
            bool valid = GetDenseLinear(row, denseLinear);
            ComputeRow(row, denseLinear, valid);
        }
    }

private:
    // 二分法在单个offset tensor中查找最后一个小于等于target的位置。
    __aicore__ inline int64_t FindLastLE(int64_t target, GlobalTensor<tType>& offset, int64_t len)
    {
        int64_t left = 0;
        int64_t right = len - 1;
        while (left <= right) {
            int64_t mid = (left + right) / 2;
            if (static_cast<int64_t>(offset.GetValue(mid)) <= target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return right;
    }

    // 将x_values中的row反解为dense除尾维D以外的线性下标。
    __aicore__ inline bool GetDenseLinear(int64_t row, int64_t& denseLinear)
    {
        int64_t target = row;
        int64_t coords[MAX_OFFSETS_CNT] = {0};
        int64_t denseOuter = 0;
        // 从最后一层offset往前找
        for (int32_t i = args->offsetCnt - 1; i >= 0; --i) {
            int64_t pos = FindLastLE(target, offsetsGb[i], args->offsetsLens[i]);
            if (pos < 0 || pos >= args->offsetsLens[i] - 1) {
                return false;
            }
            int64_t begin = static_cast<int64_t>(offsetsGb[i].GetValue(pos));
            int64_t coord = target - begin;
            if (coord < 0 || coord >= args->maxLengths[i]) {
                return false;
            }
            coords[i] = coord;
            target = pos;
            if (i == 0) {
                denseOuter = pos;
            }
        }
        denseLinear = denseOuter;
        for (int32_t i = 0; i < args->offsetCnt; ++i) {
            denseLinear = denseLinear * args->maxLengths[i] + coords[i];
        }
        return true;
    }

    // dense坐标无效时按padding为0处理：add输出x，mul输出0。
    __aicore__ inline void CopyXOrZero(GlobalTensor<uint8_t>& outCopyGb, GlobalTensor<uint8_t>& xCopyGb, int64_t len)
    {
        int64_t remainLen = len;
        while (remainLen > 0) {
            int64_t thisLen = args->singleLoopSize - ALIGN_32;
            if (remainLen < thisLen) {
                thisLen = remainLen;
            }
            LocalTensor<uint8_t> localOut = outQueue.AllocTensor<uint8_t>();
            uint32_t overAlignLen = (thisLen + ALIGN_32 - 1) / ALIGN_32 * ALIGN_32;
            if (args->elementwiseMode == MODE_ADD) {
                LocalTensor<uint8_t> localX = xQueue.AllocTensor<uint8_t>();
                DataCopy(localX, xCopyGb, overAlignLen);
                AscendC::PipeBarrier<PIPE_ALL>();
                CpLocal2Gm(outCopyGb, localX, thisLen);
                xQueue.FreeTensor(localX);
            } else {
                LocalTensor<dType> oVals = localOut.template ReinterpretCast<dType>();
                uint32_t nElem = static_cast<uint32_t>(thisLen / static_cast<int64_t>(sizeof(dType)));
                dType zeroVal = static_cast<dType>(0);
                Duplicate(oVals, zeroVal, nElem);
                AscendC::PipeBarrier<PIPE_ALL>();
                CpLocal2Gm(outCopyGb, localOut, thisLen);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            outQueue.FreeTensor(localOut);
            outCopyGb = outCopyGb[thisLen];
            xCopyGb = xCopyGb[thisLen];
            remainLen -= thisLen;
        }
    }

    __aicore__ inline void ComputeRow(int64_t row, int64_t denseLinear, bool valid)
    {
        int64_t byteLen = static_cast<int64_t>(args->denseDim2) * align;
        int64_t xOffset = row * args->denseDim2 * align;
        GlobalTensor<uint8_t> outCopyGb = outGb[xOffset];
        GlobalTensor<uint8_t> xCopyGb = xGb[xOffset];
        if (!valid) {
            CopyXOrZero(outCopyGb, xCopyGb, byteLen);
            return;
        }
        int64_t denseOffset = denseLinear * args->denseDim2 * align;
        GlobalTensor<uint8_t> denseCopyGb = denseGb[denseOffset];
        int64_t remainLen = byteLen;
        while (remainLen > 0) {
            int64_t thisLen = args->singleLoopSize - ALIGN_32;
            if (remainLen < thisLen) {
                thisLen = remainLen;
            }
            LocalTensor<uint8_t> localDense = denseQueue.AllocTensor<uint8_t>();
            LocalTensor<uint8_t> localX = xQueue.AllocTensor<uint8_t>();
            LocalTensor<uint8_t> localOut = outQueue.AllocTensor<uint8_t>();
            uint32_t overAlignLen = (thisLen + ALIGN_32 - 1) / ALIGN_32 * ALIGN_32;
            DataCopy(localDense, denseCopyGb, overAlignLen);
            DataCopy(localX, xCopyGb, overAlignLen);
            LocalTensor<dType> dVals = localDense.template ReinterpretCast<dType>();
            LocalTensor<dType> xVals = localX.template ReinterpretCast<dType>();
            LocalTensor<dType> oVals = localOut.template ReinterpretCast<dType>();
            AscendC::PipeBarrier<PIPE_ALL>();
            uint32_t nElem = static_cast<uint32_t>(thisLen / static_cast<int64_t>(sizeof(dType)));
            if (args->elementwiseMode == MODE_ADD) {
                Add(oVals, xVals, dVals, nElem);
            } else {
                Mul(oVals, xVals, dVals, nElem);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            CpLocal2Gm(outCopyGb, localOut, thisLen);
            denseQueue.FreeTensor(localDense);
            xQueue.FreeTensor(localX);
            outQueue.FreeTensor(localOut);
            outCopyGb = outCopyGb[thisLen];
            xCopyGb = xCopyGb[thisLen];
            denseCopyGb = denseCopyGb[thisLen];
            remainLen -= thisLen;
        }
    }

    TPipe* pipe;
    int32_t align;
    int32_t thisId;
    int64_t totalRows;
    int64_t startOfThisCore;
    int64_t lenOfThisCore;
    JaggedDenseEwBinaryArgs* args;
    GlobalTensor<uint8_t> denseGb;
    GlobalTensor<uint8_t> xGb;
    GlobalTensor<uint8_t> outGb;
    GlobalTensor<tType> offsetsGb[MAX_OFFSETS_CNT];
    TQue<QuePosition::VECIN, 1> denseQueue;
    TQue<QuePosition::VECIN, 1> xQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
};
}  // namespace JaggedDenseEwBinaryJOut_Kernel

extern "C" __global__ __aicore__ void jagged_dense_elementwise_binary_jagged_output(GM_ADDR xValues, GM_ADDR dense,
                                                                                    GM_ADDR offsets, GM_ADDR out,
                                                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    JaggedDenseEwBinaryJOut_Kernel::JaggedDenseEwBinaryArgs args{
        xValues,
        dense,
        offsets,
        out,
        tiling_data.denseDim1,
        tiling_data.denseDim2,
        tiling_data.singleLoopSize,
        tiling_data.denseTotal,
        tiling_data.jaggedTotal,
        tiling_data.elementwiseMode,
        tiling_data.offsetCnt,
        {tiling_data.maxLengths[0], tiling_data.maxLengths[1], tiling_data.maxLengths[2], tiling_data.maxLengths[3],
         tiling_data.maxLengths[4]},
        {tiling_data.offsetsLens[0], tiling_data.offsetsLens[1], tiling_data.offsetsLens[2], tiling_data.offsetsLens[3],
         tiling_data.offsetsLens[4]},
    };
    TPipe pipe;
    JaggedDenseEwBinaryJOut_Kernel::JaggedDenseElementwiseBinaryJaggedOutput<DTYPE_DENSE, DTYPE_OFFSETS> kernel;
    kernel.init(&args, &pipe);
    kernel.Compute();
}
