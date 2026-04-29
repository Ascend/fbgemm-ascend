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

#ifndef JAGGED_TO_PADDED_DENSE__KERNEL_FUN_H
#define JAGGED_TO_PADDED_DENSE__KERNEL_FUN_H

#include <cstdint>
#include <type_traits>

#include "constant.h"
#include "position.h"
#include "utils.h"
#include "kernel_operator_list_tensor_intf.h"

using namespace AscendC;

namespace JaggedToPaddedDense {

struct Args {
    GM_ADDR values;
    GM_ADDR offsets;
    GM_ADDR out;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

template <typename VALUE_TYPE, typename OFFSET_TYPE>
class JaggedToPaddedDenseV2Kernel {
public:
    __aicore__ inline JaggedToPaddedDenseV2Kernel(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        _SettingShape(tilingData);
        _SettingTiling(tilingData);
        _SettingGlobalTensor(args);
        _SettingLocalTensor(tilingData);
    }

    template <bool ALIGN>
    __aicore__ inline void DataCopyIn(LocalTensor<VALUE_TYPE>& tensor, int cur, int cnt)
    {
        if constexpr (ALIGN) {
            DataCopy(tensor, valuesGT[cur * innerDenseSize], cnt * innerDenseSize);
        } else {
            DataCopyExtParams copyParams{static_cast<uint16_t>(cnt),
                                         static_cast<uint32_t>(innerDenseSize * sizeof(VALUE_TYPE)), 0, 0, 0};
            DataCopyPadExtParams<VALUE_TYPE> invalid{};
            DataCopyPad(tensor, valuesGT[cur * innerDenseSize], copyParams, invalid);
        }
        inQueueX.EnQue(tensor);
        inQueueX.DeQue<VALUE_TYPE>();
    }

    template <bool ALIGN>
    __aicore__ inline void DataCopyOut(LocalTensor<VALUE_TYPE>& tensor,
                                       JaggedPosition<OFFSET_TYPE>& start,
                                       JaggedPosition<OFFSET_TYPE>& end)
    {
        int bound = offsetsGT[dim - 1].GetValue(end[dim - 1]) + end[dim];
        int residual = (end[dim] > 0);
        int tensorsCnt = end[dim - 1] - start[dim - 1] + residual;
        int ubCumsum = 0;
        for (int i = 0; i < tensorsCnt; i++) {
            int pos[4];
            start.GetValidFromTo(bound, pos);
            int rawFrom = pos[0];
            int rawTo = pos[1];
            int validFrom = pos[2];
            int validTo = pos[3];

            int rawRows = rawTo - rawFrom;
            int validRows = validTo - validFrom;

            if (validRows > 0) {
                auto outPtr = start.GetDenseOutPtr();
                if constexpr (ALIGN) {
                    DataCopy(outGT[outPtr], tensor[ubCumsum * innerDenseSize], validRows * innerDenseSize);
                } else {
                    DataCopyExtParams copyParams{static_cast<uint16_t>(validRows),
                                                 static_cast<uint32_t>(innerDenseSize * sizeof(VALUE_TYPE)), 0, 0, 0};
                    DataCopyPad(outGT[outPtr], tensor[ubCumsum * innerDenseSize32], copyParams);
                }
                pipe_barrier(PIPE_ALL);
            }
            ubCumsum += rawRows;
            start.Update(rawRows);
        }
    }

    __aicore__ inline void Compute()
    {
        int remain = lenOfThisCore;
        int cur = startOfThisCore;

        LocalTensor<VALUE_TYPE> tensor = inQueueX.AllocTensor<VALUE_TYPE>();

        while (remain > 0) {
            if (windowSize > remain) {
                windowSize = remain;
            }
            JaggedPosition<OFFSET_TYPE> startPos(cur, offsetsGT, offsetsLens, maxLengths, dim, innerDenseSize);
            JaggedPosition<OFFSET_TYPE> endPos(cur + windowSize, offsetsGT, offsetsLens, maxLengths, dim, innerDenseSize);
            if (innerDenseSize == innerDenseSize32) {
                DataCopyIn<true>(tensor, cur, windowSize);
                DataCopyOut<true>(tensor, startPos, endPos);
            } else {
                DataCopyIn<false>(tensor, cur, windowSize);
                DataCopyOut<false>(tensor, startPos, endPos);
            }

            remain -= windowSize;
            cur += windowSize;
        }
        inQueueX.FreeTensor(tensor);
    }

private:
    __aicore__ inline void _SettingShape(JaggedToPaddedDenseV2TilingData& tilingData)
    {
        total = tilingData.total;
        innerDenseSize = tilingData.innerDenseSize;
        innerDenseSize32 = AlignUp(innerDenseSize * sizeof(VALUE_TYPE), DATA_ALIGN_BYTES) / sizeof(VALUE_TYPE);
        outerDenseSize = tilingData.outerDenseSize;

        dim = tilingData.offsetCnt;
        for (int i = 0; i < dim; i++) {
            maxLengths[i] = tilingData.maxLengths[i];
            offsetsLens[i] = tilingData.offsetsLens[i];
        }
        paddingValue = tilingData.paddingValue;
    }

    __aicore__ inline void _SettingTiling(JaggedToPaddedDenseV2TilingData& tilingData)
    {
        int coreT = total / GetBlockNum();
        int tailSplitIndex = total % GetBlockNum();

        if (GetBlockIdx() >= tailSplitIndex) {
            lenOfThisCore = coreT;
            startOfThisCore = tailSplitIndex * (coreT + 1) + (GetBlockIdx() - tailSplitIndex) * coreT;
        } else {
            lenOfThisCore = coreT + 1;
            startOfThisCore = GetBlockIdx() * (coreT + 1);
        }
    }

    __aicore__ inline void _SettingGlobalTensor(Args& args)
    {
        valuesGT.SetGlobalBuffer((__gm__ VALUE_TYPE*)args.values);
        outGT.SetGlobalBuffer((__gm__ VALUE_TYPE*)args.out);

        AscendC::ListTensorDesc offsets;
        offsets.Init(args.offsets);

        for (int i = 0; i < dim; i++) {
            GM_ADDR offset = (__gm__ uint8_t*)offsets.GetDataPtr<__gm__ uint8_t>(i);
            offsetsGT[i].SetGlobalBuffer((__gm__ OFFSET_TYPE*)offset);
        }
    }

    static __aicore__ inline void _FillGM(const GlobalTensor<VALUE_TYPE>& gt,
                                          const int64_t size,
                                          const float value)
    {
#ifdef INT64_TYPE_USED_COPY_PADDING_UB
        if constexpr (std::is_same<VALUE_TYPE, bfloat16_t>::value) {
            half padValue = static_cast<half>(value);
            GlobalTensor<half> gmWorkspaceAddr;
            gmWorkspaceAddr.SetGlobalBuffer((__gm__ half*)gt.GetPhyAddr());
            Fill<half>(gmWorkspaceAddr, size, padValue);
        } else if constexpr (std::is_same<VALUE_TYPE, int64_t>::value) {
            int64_t paddingValue64 = static_cast<int64_t>(value);
            uint32_t padValue = static_cast<uint32_t>(paddingValue64);
            GlobalTensor<uint32_t> gmWorkspaceAddr;
            gmWorkspaceAddr.SetGlobalBuffer((__gm__ uint32_t*)gt.GetPhyAddr());
            Fill<uint32_t>(gmWorkspaceAddr, size, padValue);
        } else
#endif
        {
            VALUE_TYPE padValue = static_cast<VALUE_TYPE>(value);
            GlobalTensor<VALUE_TYPE> gmWorkspaceAddr;
            gmWorkspaceAddr.SetGlobalBuffer((__gm__ VALUE_TYPE*)gt.GetPhyAddr());
            Fill<VALUE_TYPE>(gmWorkspaceAddr, size, padValue);
        }
    }

    __aicore__ inline void CleanOutGT()
    {
        int64_t outTotalEles = innerDenseSize * outerDenseSize;
        for (auto i = 0; i < dim; i++) {
            outTotalEles *= maxLengths[dim];
        }
        int64_t coreT = outTotalEles / GetBlockNum();
        int64_t tailSplitIndex = outTotalEles % GetBlockNum();

        int64_t cleanLen;
        int64_t cleanStart;
        if (GetBlockIdx() >= tailSplitIndex) {
            cleanLen = coreT;
            cleanStart = tailSplitIndex * (coreT + 1) + (GetBlockIdx() - tailSplitIndex) * coreT;
        } else {
            cleanLen = coreT + 1;
            cleanStart = GetBlockIdx() * (coreT + 1);
        }
        _FillGM(outGT[cleanStart], cleanLen, paddingValue);
        // 清空后同步
        TEventID event = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
        SetFlag<HardEvent::MTE3_MTE2>(event);
        WaitFlag<HardEvent::MTE3_MTE2>(event);
    }

    __aicore__ inline void _SettingLocalTensor(JaggedToPaddedDenseV2TilingData& tilingData)
    {
        ubCanUsed = tilingData.ubCanUsed;

        int rows = ubCanUsed / USE_QUEUE_NUM / innerDenseSize32 / sizeof(VALUE_TYPE);
        windowSize = rows;
        pipe.InitBuffer(inQueueX, USE_QUEUE_NUM, rows * innerDenseSize32 * sizeof(VALUE_TYPE));
    }

private:
    // Shape
    int64_t total;             // values(T, D).size(0)
    int64_t innerDenseSize;    // values(T, D).size(-1)
    int64_t innerDenseSize32;  // 32B对齐的D
    int64_t outerDenseSize;    // len(offsets[0]) - 1
    int64_t dim;               // len(offsets), len(max_lengths)
    int64_t maxLengths[MAX_OFFSETS_CNT];
    int64_t offsetsLens[MAX_OFFSETS_CNT];

    // Ub
    int64_t ubCanUsed;
    int64_t windowSize;  // ub一次最大拷入values的行数
    float paddingValue;

    // ThisCoreLen
    int64_t lenOfThisCore;
    int64_t startOfThisCore;

    // Tpipe
    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, USE_QUEUE_NUM> inQueueX;

    // ThisCoreAddr
    GlobalTensor<VALUE_TYPE> valuesGT;
    GlobalTensor<OFFSET_TYPE> offsetsGT[MAX_OFFSETS_CNT];
    GlobalTensor<VALUE_TYPE> outGT;
};
}  // namespace JaggedToPaddedDense
#endif