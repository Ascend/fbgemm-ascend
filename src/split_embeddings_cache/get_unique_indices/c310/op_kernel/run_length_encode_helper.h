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

#ifndef RUN_LENGTH_ENCODE_HELPER_H
#define RUN_LENGTH_ENCODE_HELPER_H

#include <type_traits>

#include "kernel_common_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "run_length_encode_constant.h"

using namespace AscendC;

template <typename T>
// 返回一个向量寄存器可容纳的 T 元素个数（vector lane 数）
__aicore__ inline constexpr uint32_t GetVLEleNums()
{
    return GetVecLen() / sizeof(T);
}

template <HardEvent ent>
// 事件同步
__aicore__ inline void SimpleNativePipeSync()
{
    event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(ent));
    SetFlag<ent>(event);
    WaitFlag<ent>(event);
}

template <typename T>
// 将寄存器中按掩码收集的元素写回 UB，封装非对齐写回流程
__aicore__ inline void CollectAndCopy2Ub(__ubuf__ T* dstUbAddr, MicroAPI::RegTensor<T>& srcReg,
                                         MicroAPI::RegTensor<T>& tmpReg, MicroAPI::MaskReg& cmpMask,
                                         MicroAPI::UnalignReg& ureg)
{
    MicroAPI::GatherMask<T, MicroAPI::GatherMaskMode::STORE_REG>(tmpReg, srcReg, cmpMask);
    MicroAPI::DataCopyUnAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstUbAddr, tmpReg, ureg);
}

template <int IDX_INC_NUMS>
// 收集索引并写回 UB，同时把索引寄存器整体递增，供下一次向量迭代复用
__aicore__ inline void CollectIdxWithUpdateAndCopy2Ub(__ubuf__ int32_t* dstUbAddr, MicroAPI::RegTensor<int32_t>& srcReg,
                                                      MicroAPI::RegTensor<int32_t>& tmpReg, MicroAPI::MaskReg& cmpMask,
                                                      MicroAPI::MaskReg& pregAll, MicroAPI::UnalignReg& ureg)
{
    CollectAndCopy2Ub<int32_t>(dstUbAddr, srcReg, tmpReg, cmpMask, ureg);
    MicroAPI::Adds(srcReg, srcReg, (int32_t)IDX_INC_NUMS, pregAll);
}

// 向量化收集“相邻元素不等”的位置索引。
template <typename T, int REP_LENGTH>
static __aicore__ inline void VFCollectPostUniqueIdx(__ubuf__ int32_t* dstIdxAddr, __ubuf__ T* srcValueAddr,
                                                     int32_t startCount, uint32_t repeatTimes, uint32_t totalNums)
{
    if ((repeatTimes == 0) || (totalNums == 0)) {
        return;
    }

    MicroAPI::RegTensor<T> xPrev;
    MicroAPI::RegTensor<T> xNext;

    MicroAPI::RegTensor<int32_t> out;
    MicroAPI::RegTensor<int32_t> idx;

    MicroAPI::UnalignReg uregIn;
    MicroAPI::UnalignReg uregOut;

    MicroAPI::MaskReg cmpRet;
    MicroAPI::MaskReg pregLoop;
    MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    uint32_t sreg0 = totalNums;

    MicroAPI::Arange(idx, startCount);
    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
        pregLoop = MicroAPI::UpdateMask<T>(sreg0);
        auto curtSrcAddr = srcValueAddr + REP_LENGTH * i + 1;

        DataCopy(xPrev, srcValueAddr + REP_LENGTH * i);
        MicroAPI::DataCopyUnAlignPre(uregIn, curtSrcAddr);
        MicroAPI::DataCopyUnAlign(xNext, uregIn, curtSrcAddr);

        MicroAPI::Compare<T, CMPMODE::NE>(cmpRet, xPrev, xNext, pregLoop);

        if constexpr (sizeof(T) == sizeof(int32_t)) {
            constexpr auto idxIncNums = GetVLEleNums<int32_t>();
            CollectIdxWithUpdateAndCopy2Ub<idxIncNums>(dstIdxAddr, idx, out, cmpRet, pregAll, uregOut);
        } else {
            static_assert(sizeof(T) == sizeof(int64_t), "run_length_encode(c310) helper only supports int32/int64.");
            AscendC::MicroAPI::MaskReg maskHalf;
            AscendC::MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(maskHalf, cmpRet);
            constexpr uint32_t idxIncNums = GetVLEleNums<int64_t>();
            CollectIdxWithUpdateAndCopy2Ub<idxIncNums>(dstIdxAddr, idx, out, maskHalf, pregAll, uregOut);
        }
    }
    MicroAPI::DataCopyUnAlignPost(dstIdxAddr, uregOut);
}

// B32（int32）路径实现
template <typename T, int REP_LENGTH>
static __aicore__ inline void VFCollectPostUniqueValue(__ubuf__ T* dstValueAddr, __ubuf__ T* srcValueAddr,
                                                       uint32_t repeatTimes, uint32_t totalNums)
{
    if ((repeatTimes == 0) || (totalNums == 0)) {
        return;
    }

    MicroAPI::RegTensor<T> xPrev;
    MicroAPI::RegTensor<T> xNext;
    MicroAPI::RegTensor<T> out;

    MicroAPI::UnalignReg uregIn;
    MicroAPI::UnalignReg uregOut;

    MicroAPI::MaskReg cmpRet;
    MicroAPI::MaskReg pregLoop;

    uint32_t sreg0 = totalNums;

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
        pregLoop = MicroAPI::UpdateMask<T>(sreg0);
        auto curtSrcAddr = srcValueAddr + REP_LENGTH * i + 1;

        DataCopy(xPrev, srcValueAddr + REP_LENGTH * i);
        MicroAPI::DataCopyUnAlignPre(uregIn, curtSrcAddr);
        MicroAPI::DataCopyUnAlign(xNext, uregIn, curtSrcAddr);

        MicroAPI::Compare<T, CMPMODE::NE>(cmpRet, xPrev, xNext, pregLoop);

        CollectAndCopy2Ub<T>(dstValueAddr, xPrev, out, cmpRet, uregOut);
    }
    MicroAPI::DataCopyUnAlignPost(dstValueAddr, uregOut);
}

template <int REP_LENGTH>
// B64 路径实现（按 int32 lane 比较后重组掩码）
static __aicore__ inline void VFCollectPostUniqueValueB64(__ubuf__ int32_t* dstValueAddr,
                                                          __ubuf__ int32_t* srcValueAddr, uint32_t repeatTimes,
                                                          uint32_t totalNums)
{
    if ((repeatTimes == 0) || (totalNums == 0)) {
        return;
    }

    MicroAPI::RegTensor<int32_t> xPrev;
    MicroAPI::RegTensor<int32_t> xNext;
    MicroAPI::RegTensor<int32_t> out;

    MicroAPI::UnalignReg uregIn;
    MicroAPI::UnalignReg uregOut;

    MicroAPI::MaskReg cmpRet;
    MicroAPI::MaskReg pregLoop;

    MicroAPI::MaskReg maskEven;
    MicroAPI::MaskReg maskOdd;

    uint32_t sreg0 = totalNums;

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
        pregLoop = MicroAPI::UpdateMask<int32_t>(sreg0);
        // B64 路径在这里按 int32 lane 视角处理：地址 +2 个 int32 等价于 +1 个 int64 元素。
        auto curtSrcAddr = srcValueAddr + REP_LENGTH * i + 2;
        DataCopy(xPrev, srcValueAddr + REP_LENGTH * i);
        MicroAPI::DataCopyUnAlignPre(uregIn, curtSrcAddr);
        MicroAPI::DataCopyUnAlign(xNext, uregIn, curtSrcAddr);
        MicroAPI::Compare<int32_t, CMPMODE::NE>(cmpRet, xPrev, xNext, pregLoop);

        MicroAPI::MaskDeInterleave<int32_t>(maskEven, maskOdd, cmpRet, cmpRet);
        MicroAPI::MaskOr(cmpRet, maskEven, maskOdd, pregLoop);
        MicroAPI::MaskInterleave<int32_t>(maskEven, maskOdd, cmpRet, cmpRet);
        CollectAndCopy2Ub<int32_t>(dstValueAddr, xPrev, out, maskEven, uregOut);
    }
    MicroAPI::DataCopyUnAlignPost(dstValueAddr, uregOut);
}

/*
    输入 x：长度为 num 的 tensor。
    规则：mask[i] = (x[i] != x[i+1])，其中 i < num - 1。
    输出：返回满足条件的位置索引（即 mask 的 nonzero）。
*/
template <typename VALUE_TYPE>
// 标量尾处理：补齐“相邻不等位置索引”的尾段结果。
__aicore__ inline void CollectPostUniqueIdxTailScalar(LocalTensor<int32_t>& dstIdx, LocalTensor<VALUE_TYPE>& srcValue,
                                                      int32_t startCount, uint32_t beginPairIdx, uint32_t endPairIdx,
                                                      uint64_t position, uint64_t& rsvdCnt)
{
    for (uint32_t i = beginPairIdx; i < endPairIdx; ++i) {
        if (srcValue.GetValue(i) != srcValue.GetValue(i + 1)) {
            dstIdx.SetValue(position + rsvdCnt, startCount + static_cast<int32_t>(i));
            rsvdCnt += 1;
        }
    }
}

template <typename VALUE_TYPE>
// 标量尾处理：补齐“unique 值收集”的尾段结果。
__aicore__ inline void CollectPostUniqueValueTailScalar(LocalTensor<VALUE_TYPE>& dstValue,
                                                        LocalTensor<VALUE_TYPE>& srcValue, uint32_t beginPairIdx,
                                                        uint32_t endPairIdx, uint64_t& rsvdCnt)
{
    for (uint32_t i = beginPairIdx; i < endPairIdx; ++i) {
        if (srcValue.GetValue(i) != srcValue.GetValue(i + 1)) {
            dstValue.SetValue(rsvdCnt, srcValue.GetValue(i));
            rsvdCnt += 1;
        }
    }
}

template <typename VALUE_TYPE, bool IS_TAIL>
// 收集相邻不等位置索引：向量主循环 + 标量尾处理；IS_TAIL 控制是否追加 endCount。
__aicore__ inline void CollectPostUniqueIdx(LocalTensor<int32_t>& dstIdx, LocalTensor<VALUE_TYPE>& srcValue,
                                            int32_t startCount, int32_t endCount, uint32_t nums, uint64_t& rsvdCnt,
                                            uint64_t position)
{
    if (nums <= 1) {
        rsvdCnt = 0;
        if constexpr (IS_TAIL) {
            if (nums == 1) {
                dstIdx.SetValue(position, endCount);
                rsvdCnt = 1;
            }
        }
        return;
    }

    // 获取 UB 物理地址，供 VF 内核按连续内存访问。
    __local_mem__ VALUE_TYPE* srcValueAddr = (__local_mem__ VALUE_TYPE*)srcValue[0].GetPhyAddr();
    __local_mem__ int32_t* dstIdxAddr = (__local_mem__ int32_t*)dstIdx[position].GetPhyAddr();

    // 计算 VF 主循环参数：向量长度、主块元素数与重复次数。
    constexpr uint32_t repNums = GetVLEleNums<VALUE_TYPE>();
    uint32_t totalNums = nums - 1;
    uint32_t mainNums = (totalNums / repNums) * repNums;
    uint32_t repTimes = mainNums / repNums;

    rsvdCnt = 0;
    if (repTimes > 0) {
        AscendC::VF_CALL<VFCollectPostUniqueIdx<VALUE_TYPE, repNums>>(dstIdxAddr, srcValueAddr, startCount, repTimes,
                                                                      mainNums);
        rsvdCnt = GetSpr<SpecialPurposeReg::AR>() / sizeof(int32_t);
    }

    CollectPostUniqueIdxTailScalar<VALUE_TYPE>(dstIdx, srcValue, startCount, mainNums, totalNums, position, rsvdCnt);
    if constexpr (IS_TAIL) {
        dstIdx.SetValue(position + rsvdCnt, endCount);
        rsvdCnt += 1;
    }
}

/*
    输入 x：长度为 num 的 tensor
    规则：mask[i] = (x[i] != x[i+1])，其中 i < num - 1
    输出：返回 x[mask.nonzero()]，即每段唯一值（尾块可补最后一个元素）
*/
template <typename VALUE_TYPE, bool IS_TAIL>
// 收集 unique 值：向量主循环 + 标量尾处理；IS_TAIL 控制是否补最后一个元素
__aicore__ inline void CollectPostUniqueValue(LocalTensor<VALUE_TYPE>& dstValue, LocalTensor<VALUE_TYPE>& srcValue,
                                              uint32_t nums, uint64_t& rsvdCnt)
{
    if (nums <= 1) {
        rsvdCnt = 0;
        if constexpr (IS_TAIL) {
            if (nums == 1) {
                dstValue.SetValue(0, srcValue.GetValue(0));
                rsvdCnt = 1;
            }
        }
        return;
    }

    rsvdCnt = 0;
    if constexpr (std::is_same<VALUE_TYPE, int64_t>::value) {
        // 获取 UB 物理地址：int64 按 int32 lane 视图读取与写回
        __local_mem__ int32_t* srcValueAddr =
            (__local_mem__ int32_t*)srcValue[0].template ReinterpretCast<int32_t>().GetPhyAddr();
        __local_mem__ int32_t* dstValueAddr =
            (__local_mem__ int32_t*)dstValue[0].template ReinterpretCast<int32_t>().GetPhyAddr();

        // int64 路径将两条 int32 lane 组合为一个逻辑 int64 元素做比较
        // 向量主循环保持 int32 粒度，所有未对齐残余由标量尾处理收敛，避免 lane 步进出错
        constexpr uint32_t repNums = GetVLEleNums<int32_t>();
        static_assert((repNums % 2) == 0, "int64 path requires even int32 vector lanes.");
        uint32_t totalNums = (nums - 1) * 2;  // 单位是 int32 lane，不是 int64 元素个数
        uint32_t mainNums = (totalNums / repNums) * repNums;
        uint32_t repTimes = mainNums / repNums;
        if (repTimes > 0) {
            AscendC::VF_CALL<VFCollectPostUniqueValueB64<repNums>>(dstValueAddr, srcValueAddr, repTimes, mainNums);
            rsvdCnt = GetSpr<SpecialPurposeReg::AR>() / sizeof(VALUE_TYPE);
        }
        CollectPostUniqueValueTailScalar<VALUE_TYPE>(dstValue, srcValue, mainNums / 2, nums - 1, rsvdCnt);
    } else {
        static_assert(sizeof(VALUE_TYPE) == sizeof(int32_t),
                      "run_length_encode(c310) helper only supports int32/int64.");
        // 获取 UB 物理地址（B32 直接按元素访问）
        __local_mem__ VALUE_TYPE* srcValueAddr = (__local_mem__ VALUE_TYPE*)srcValue[0].GetPhyAddr();
        __local_mem__ VALUE_TYPE* dstValueAddr = (__local_mem__ VALUE_TYPE*)dstValue[0].GetPhyAddr();

        // 计算 VF 主循环参数：向量长度、主块元素数与重复次数
        constexpr uint32_t repNums = GetVLEleNums<VALUE_TYPE>();
        uint32_t totalNums = nums - 1;
        uint32_t mainNums = (totalNums / repNums) * repNums;
        uint32_t repTimes = mainNums / repNums;
        if (repTimes > 0) {
            AscendC::VF_CALL<VFCollectPostUniqueValue<VALUE_TYPE, repNums>>(dstValueAddr, srcValueAddr, repTimes,
                                                                            mainNums);
            rsvdCnt = GetSpr<SpecialPurposeReg::AR>() / sizeof(VALUE_TYPE);
        }
        CollectPostUniqueValueTailScalar<VALUE_TYPE>(dstValue, srcValue, mainNums, totalNums, rsvdCnt);
    }
    if constexpr (IS_TAIL) {
        dstValue.SetValue(rsvdCnt, srcValue.GetValue(nums - 1));
        rsvdCnt += 1;
    }
}

template <int REP_LENGTH, typename T>
// 向量化相邻差分：计算 dst[1:] = src[1:] - src[:-1]，分主块与尾块执行
static __aicore__ inline void VFPostAdjDiff(__ubuf__ T* dstIdxAddr, __ubuf__ T* srcIdxAddr, uint32_t repeatTimes,
                                            uint32_t totalNums, uint16_t hasTail)
{
    if ((repeatTimes == 0) && (hasTail == 0)) {
        return;
    }

    MicroAPI::RegTensor<T> idxPrev;
    MicroAPI::RegTensor<T> idxNext;
    MicroAPI::RegTensor<T> out;

    MicroAPI::UnalignReg uregIn;
    MicroAPI::UnalignReg uregOut;

    MicroAPI::MaskReg pregLoop;

    uint32_t sreg0 = totalNums;
    uint32_t tailLen = totalNums % REP_LENGTH;

    auto curtDstAddr = dstIdxAddr + 1;

    // 主块：按完整向量长度处理。
    for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
        pregLoop = MicroAPI::UpdateMask<T>(sreg0);
        auto curtSrcAddr = srcIdxAddr + REP_LENGTH * i + 1;
        DataCopy(idxPrev, srcIdxAddr + REP_LENGTH * i);
        MicroAPI::DataCopyUnAlignPre(uregIn, curtSrcAddr);
        MicroAPI::DataCopyUnAlign(idxNext, uregIn, curtSrcAddr);
        MicroAPI::Sub(out, idxNext, idxPrev, pregLoop);
        MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(curtDstAddr, out, uregOut, REP_LENGTH);
    }

    // 尾块：仅在 hasTail != 0 时处理一次剩余不足一个完整向量的元素。
    if (hasTail != 0) {
        pregLoop = MicroAPI::UpdateMask<T>(sreg0);
        auto curtSrcAddr = srcIdxAddr + REP_LENGTH * repeatTimes + 1;
        DataCopy(idxPrev, srcIdxAddr + REP_LENGTH * repeatTimes);
        MicroAPI::DataCopyUnAlignPre(uregIn, curtSrcAddr);
        MicroAPI::DataCopyUnAlign(idxNext, uregIn, curtSrcAddr);
        MicroAPI::Sub(out, idxNext, idxPrev, pregLoop);
        MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(curtDstAddr, out, uregOut, tailLen);
    }
    MicroAPI::DataCopyUnAlignPost(curtDstAddr, uregOut, 0);
}

/*
    输入 srcIdx：长度为 num 的 idx 序列，起始值通常从 1 开始。
    输出 dstIdx：
      dstIdx[0] = firstValue
      dstIdx[i] = srcIdx[i] - srcIdx[i - 1]（i > 0）
    首元素通过标量写入，其余元素走 VF 差分流程。
*/
template <typename T>
// 计数后处理：将边界索引序列转为 count 序列，首元素由 firstValue 提供。
__aicore__ inline void PostAdjDiff(LocalTensor<T>& dstIdx, LocalTensor<T>& srcIdx, T firstValue, uint32_t nums,
                                   uint64_t position)
{
    if (nums == 0) {
        return;
    }
    if (nums == 1) {
        dstIdx.SetValue(position, firstValue);
        return;
    }

    // 获取 UB 物理地址，供 VF 差分内核访问。
    __local_mem__ T* srcIdxAddr = (__local_mem__ T*)srcIdx[0].GetPhyAddr();
    __local_mem__ T* dstIdxAddr = (__local_mem__ T*)dstIdx[position].GetPhyAddr();

    // 计算 VF 主循环参数：向量长度、主块元素数与重复次数。
    constexpr uint32_t repNums = GetVLEleNums<T>();
    uint32_t totalNums = nums - 1;
    uint32_t mainNums = (totalNums / repNums) * repNums;
    uint32_t repTimes = mainNums / repNums;

    if (repTimes > 0) {
        AscendC::VF_CALL<VFPostAdjDiff<repNums, T>>(dstIdxAddr, srcIdxAddr, repTimes, mainNums, 0);
    }
    dstIdx.SetValue(position, firstValue);
    for (uint32_t i = mainNums + 1; i < nums; ++i) {
        dstIdx.SetValue(position + i, srcIdx.GetValue(i) - srcIdx.GetValue(i - 1));
    }
}

#endif  // RUN_LENGTH_ENCODE_HELPER_H
