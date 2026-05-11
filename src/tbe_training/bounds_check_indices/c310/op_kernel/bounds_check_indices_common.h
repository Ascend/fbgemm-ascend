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

#ifndef BOUNDS_CHECK_INDICES_COMMON_H
#define BOUNDS_CHECK_INDICES_COMMON_H

#include <cstdint>
#include <limits>
#include <type_traits>
#include "kernel_operator.h"
#include "simt_api/device_atomic_functions.h"

#ifdef NDEBUG
#define BOUNDS_ASSERT(expr) ((void)0)
#else
#define BOUNDS_ASSERT(expr) (static_cast<bool>(expr) ? void(0) : __assert_fail(#expr, __FILE__, __LINE__, ""))
#endif

#define INVOKE_BOUNDS_CHECK(dispatchMacro, indiceType, vbe, mode)                                                  \
    if (vbe) {                                                                                                     \
        if (mode == BoundsCheckMode::FATAL) { dispatchMacro(indiceType, true, BoundsCheckMode::FATAL); }           \
        else if (mode == BoundsCheckMode::WARNING) { dispatchMacro(indiceType, true, BoundsCheckMode::WARNING); }  \
        else if (mode == BoundsCheckMode::IGNORE) { dispatchMacro(indiceType, true, BoundsCheckMode::IGNORE); }    \
    } else {                                                                                                       \
        if (mode == BoundsCheckMode::FATAL) { dispatchMacro(indiceType, false, BoundsCheckMode::FATAL); }          \
        else if (mode == BoundsCheckMode::WARNING) { dispatchMacro(indiceType, false, BoundsCheckMode::WARNING); } \
        else if (mode == BoundsCheckMode::IGNORE) { dispatchMacro(indiceType, false, BoundsCheckMode::IGNORE); }   \
    }

enum class BoundsCheckMode : uint8_t {
    FATAL = 0,
    WARNING = 1,
    IGNORE = 2
};

template <typename indiceType>
__simt_callee__ __aicore__ inline void AdjustOffset(
    indiceType& indiceStart,
    indiceType& indiceEnd,
    const indiceType numIndices,
    __gm__ indiceType* offsetStart,
    __gm__ indiceType* offsetEnd)
{
    indiceStart = max(static_cast<indiceType>(0), min(indiceStart, numIndices));
    indiceEnd = max(indiceStart, min(indiceEnd, numIndices));
    *offsetStart = indiceStart;
    *offsetEnd = indiceEnd;
}

template <typename UnsignedT>
class FastDivmod {
public:
    static constexpr UnsignedT UINT_DIV_MAX_DIVIDEND =
        static_cast<UnsignedT>(std::numeric_limits<typename std::make_signed<UnsignedT>::type>::max());

    __simt_callee__ inline FastDivmod(UnsignedT magic, uint32_t shift, UnsignedT divisor)
        : magic_(magic), shift_(shift), divisor_(divisor)
    {
    }

    __simt_callee__ inline UnsignedT Div(UnsignedT n) const
    {
        if (divisor_ <= 1) {
            return (divisor_ == 1) ? n : static_cast<UnsignedT>(0);
        }
        if (n > UINT_DIV_MAX_DIVIDEND) {
            return n / divisor_;
        }
        return AscendC::Simt::UintDiv<UnsignedT>(n, magic_, static_cast<UnsignedT>(shift_));
    }

    __simt_callee__ inline UnsignedT Mod(UnsignedT n) const
    {
        if (divisor_ <= 1) {
            return (divisor_ == 1) ? static_cast<UnsignedT>(0) : n;
        }
        if (n > UINT_DIV_MAX_DIVIDEND) {
            return n % divisor_;
        }
        const UnsignedT q = AscendC::Simt::UintDiv<UnsignedT>(n, magic_, static_cast<UnsignedT>(shift_));
        return n - q * divisor_;
    }

private:
    UnsignedT magic_;
    uint32_t shift_;
    UnsignedT divisor_;
};

#endif // BOUNDS_CHECK_INDICES_COMMON_H
