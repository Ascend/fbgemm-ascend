/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved. */

#include <cstdint>
#include "kernel_common_utils.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

using namespace AscendC;

namespace JaggedDenseDenseAddJOut_Kernel {
constexpr int32_t ALIGN_32 = 32;
constexpr int32_t MAX_OFFSETS_CNT = 5;

struct Args {
    GM_ADDR xValues;
    GM_ADDR dense0;
    GM_ADDR dense1;
    GM_ADDR offsets;
    GM_ADDR out;
    int32_t denseDim1;
    int32_t denseDim2;
    int32_t singleLoopSize;
    int64_t denseTotal;
    int64_t jaggedTotal;
    int32_t offsetCnt;
    int64_t maxLengths[MAX_OFFSETS_CNT];
    int64_t offsetsLens[MAX_OFFSETS_CNT];
};

template <typename dType, typename tType>
class Kernel {
public:
    __aicore__ inline Kernel(){};

    // 初始化 GM/UB 资源，并按 row 维度将 jagged values 均分到各个 AIV core。
    __aicore__ inline void init(Args* a, TPipe* pipe)
    {
        args = a;
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
        d0Gb.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(a->dense0), a->denseTotal * align);
        d1Gb.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(a->dense1), a->denseTotal * align);
        xGb.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(a->xValues), a->jaggedTotal * align);
        outGb.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(a->out), a->jaggedTotal * align);
        AscendC::ListTensorDesc offsetsDesc;
        offsetsDesc.Init(args->offsets);
        // dynamic offsets 输入以 ListTensor 形式传入，逐个取出后按统一 offsets dtype 绑定 GM。
        for (int32_t i = 0; i < args->offsetCnt; ++i) {
            GM_ADDR offset = (__gm__ uint8_t*)offsetsDesc.GetDataPtr<__gm__ uint8_t>(i);
            offsetsGb[i].SetGlobalBuffer((__gm__ tType*)offset, args->offsetsLens[i]);
        }
        pipe->InitBuffer(q0, 1, a->singleLoopSize);
        pipe->InitBuffer(q1, 1, a->singleLoopSize);
        pipe->InitBuffer(qx, 1, a->singleLoopSize);
        pipe->InitBuffer(qo, 1, a->singleLoopSize);
    }

    __aicore__ inline void Compute()
    {
        int64_t end = startOfThisCore + lenOfThisCore;
        for (int64_t row = startOfThisCore; row < end; ++row) {
            int64_t denseLinear = 0;
            bool valid = GetDenseLinear(row, denseLinear);
            ComputeRow(row, denseLinear, valid);
        }
    }

private:
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

    // 将 jagged values 的 row 下标映射到 dense tensor 的展平 row 下标。
    // 本函数从最内层 offsets[i-1] 开始反推到 offsets[0]，得到 batch 下标 denseOuter 以及每层坐标 coords[i]。
    __aicore__ inline bool GetDenseLinear(int64_t row, int64_t& denseLinear)
    {
        // target 表示当前层需要定位的 row
        int64_t target = row;
        // coords[i] 保存第 i 个 jagged 维度在 padded dense 中的便宜，即 target - offsets[i][parent]。
        int64_t coords[MAX_OFFSETS_CNT] = {0};
        // denseOuter 保存 batch 维下标，对应 offsets[0] 中定位到的 parent 位置。
        int64_t denseOuter = 0;
        for (int32_t i = args->offsetCnt - 1; i >= 0; --i) {
            // offsets[i][pos] <= target < offsets[i][pos + 1] 时，pos 就是当前 child index 对应的 parent index。
            int64_t pos = FindLastLE(target, offsetsGb[i], args->offsetsLens[i]);
            if (pos < 0 || pos >= args->offsetsLens[i] - 1) {
                return false;
            }

            int64_t begin = static_cast<int64_t>(offsetsGb[i].GetValue(pos));
            // coord 是当前 child index 在 parent segment 内的相对位置，即偏移
            int64_t coord = target - begin;
            if (coord < 0 || coord >= args->maxLengths[i]) {
                return false;
            }
            coords[i] = coord;

            // 下一轮向外层反推：当前层 parent index 会成为上一层的 child index。
            target = pos;
            if (i == 0) {
                denseOuter = pos;
            }
        }
        denseLinear = denseOuter;
        // 按行主序将 [batch, coord0, coord1, ..., D] 展平成 dense row 编号。
        for (int32_t i = 0; i < args->offsetCnt; ++i) {
            denseLinear = denseLinear * args->maxLengths[i] + coords[i];
        }
        return true;
    }

    // 当 jagged row 无法映射到 dense 有效范围时，仅将 xValues 原样写回输出。
    __aicore__ inline void CopyX(GlobalTensor<uint8_t>& outCopyGb, GlobalTensor<uint8_t>& xCopyGb, int64_t len)
    {
        int64_t remainLen = len;
        while (remainLen > 0) {
            int64_t thisLen = args->singleLoopSize - ALIGN_32;
            if (remainLen < thisLen) {
                thisLen = remainLen;
            }
            LocalTensor<uint8_t> localX = qx.AllocTensor<uint8_t>();
            uint32_t overAlignLen = (thisLen + ALIGN_32 - 1) / ALIGN_32 * ALIGN_32;
            DataCopy(localX, xCopyGb, overAlignLen);

            AscendC::PipeBarrier<PIPE_ALL>();
            CpLocal2Gm(outCopyGb, localX, thisLen);
            AscendC::PipeBarrier<PIPE_ALL>();

            qx.FreeTensor(localX);
            outCopyGb = outCopyGb[thisLen];
            xCopyGb = xCopyGb[thisLen];
            remainLen -= thisLen;
        }
    }

    __aicore__ inline void ComputeRow(int64_t row, int64_t denseLinear, bool valid)
    {
        int64_t byteLen = static_cast<int64_t>(args->denseDim2) * align;
        int64_t xOffset = row * args->denseDim2 * align;
        GlobalTensor<uint8_t> oGb = outGb[xOffset];
        GlobalTensor<uint8_t> xGb2 = xGb[xOffset];
        // valid=false 表示该 jagged 位置超出 dense max_lengths 或 offsets 反推失败，此时 dense 贡献视为 0。
        if (!valid) {
            CopyX(oGb, xGb2, byteLen);
            return;
        }
        int64_t denseOffset = denseLinear * args->denseDim2 * align;
        GlobalTensor<uint8_t> g0 = d0Gb[denseOffset];
        GlobalTensor<uint8_t> g1 = d1Gb[denseOffset];
        int64_t rem = byteLen;
        while (rem > 0) {
            int64_t tl = args->singleLoopSize - ALIGN_32;
            if (rem < tl) {
                tl = rem;
            }
            LocalTensor<uint8_t> l0 = q0.AllocTensor<uint8_t>();
            LocalTensor<uint8_t> l1 = q1.AllocTensor<uint8_t>();
            LocalTensor<uint8_t> lx = qx.AllocTensor<uint8_t>();
            LocalTensor<uint8_t> lo = qo.AllocTensor<uint8_t>();
            uint32_t over = (tl + ALIGN_32 - 1) / ALIGN_32 * ALIGN_32;
            DataCopy(l0, g0, over);
            DataCopy(l1, g1, over);
            DataCopy(lx, xGb2, over);
            AscendC::PipeBarrier<PIPE_ALL>();
            LocalTensor<dType> v0 = l0.template ReinterpretCast<dType>();
            LocalTensor<dType> v1 = l1.template ReinterpretCast<dType>();
            LocalTensor<dType> vx = lx.template ReinterpretCast<dType>();
            LocalTensor<dType> vo = lo.template ReinterpretCast<dType>();
            uint32_t nElem = static_cast<uint32_t>(tl / static_cast<int64_t>(sizeof(dType)));
            Add(vo, vx, v0, nElem);
            AscendC::PipeBarrier<PIPE_ALL>();
            Add(vo, vo, v1, nElem);
            AscendC::PipeBarrier<PIPE_ALL>();
            CpLocal2Gm(oGb, lo, tl);
            q0.FreeTensor(l0);
            q1.FreeTensor(l1);
            qx.FreeTensor(lx);
            qo.FreeTensor(lo);
            oGb = oGb[tl];
            xGb2 = xGb2[tl];
            g0 = g0[tl];
            g1 = g1[tl];
            rem -= tl;
        }
    }

    TPipe* pipe;
    int32_t align;
    int32_t thisId;
    int64_t totalRows;
    int64_t startOfThisCore;
    int64_t lenOfThisCore;
    Args* args;
    GlobalTensor<uint8_t> d0Gb;
    GlobalTensor<uint8_t> d1Gb;
    GlobalTensor<uint8_t> xGb;
    GlobalTensor<uint8_t> outGb;
    GlobalTensor<tType> offsetsGb[MAX_OFFSETS_CNT];
    TQue<QuePosition::VECIN, 1> q0;
    TQue<QuePosition::VECIN, 1> q1;
    TQue<QuePosition::VECIN, 1> qx;
    TQue<QuePosition::VECOUT, 1> qo;
};
}  // namespace JaggedDenseDenseAddJOut_Kernel

extern "C" __global__ __aicore__ void jagged_dense_dense_elementwise_add_jagged_output(GM_ADDR xValues, GM_ADDR dense_0,
                                                                                       GM_ADDR dense_1, GM_ADDR offsets,
                                                                                       GM_ADDR out, GM_ADDR workspace,
                                                                                       GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    JaggedDenseDenseAddJOut_Kernel::Args args{
        xValues,
        dense_0,
        dense_1,
        offsets,
        out,
        tiling_data.denseDim1,
        tiling_data.denseDim2,
        tiling_data.singleLoopSize,
        tiling_data.denseTotal,
        tiling_data.jaggedTotal,
        tiling_data.offsetCnt,
        {tiling_data.maxLengths[0], tiling_data.maxLengths[1], tiling_data.maxLengths[2], tiling_data.maxLengths[3],
         tiling_data.maxLengths[4]},
        {tiling_data.offsetsLens[0], tiling_data.offsetsLens[1], tiling_data.offsetsLens[2], tiling_data.offsetsLens[3],
         tiling_data.offsetsLens[4]},
    };
    TPipe pipe;
    JaggedDenseDenseAddJOut_Kernel::Kernel<DTYPE_DENSE_0, DTYPE_OFFSETS> k;
    k.init(&args, &pipe);
    k.Compute();
}
