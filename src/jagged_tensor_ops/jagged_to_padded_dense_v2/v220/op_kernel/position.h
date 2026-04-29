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

#ifndef JAGGED_POSITION_H
#define JAGGED_POSITION_H
#include "constant.h"
#include "utils.h"

constexpr int ROUND_DOWN = -1;
constexpr int ROUND_NONE = 0;
constexpr int ROUND_UP = 1;

template <typename OFFSET_TYPE>
__aicore__ static int FindLastLE(int target, GlobalTensor<OFFSET_TYPE>& offset, int len)
{
    int left = 0;
    int right = len - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (offset.GetValue(mid) <= target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return right;
}

__aicore__ inline void DenseShapeRoundUp(int* dense, int* denseShape, int dim)
{
    for (int i = dim + 1; i >= 0; i--) {
        if (dense[i] >= denseShape[i]) {
            dense[i - 1] += 1;
            dense[i] = 0;
        }
    }
}

__aicore__ inline void DenseShapeRoundDown(int* dense, int* denseShape, int dim)
{
    int i = 0;
    while (i <= dim + 1) {
        if (dense[i] >= denseShape[i]) {
            break;
        }
        i++;
    }

    while (i <= dim + 1) {
        dense[i] = denseShape[i] - 1;
        i++;
    }
}

__aicore__ inline void DenseShapeRoundNone(int* dense, int* denseShape, int dim)
{
    for (int i = 0; i < dim + 2; i++) {
        if (dense[i] >= denseShape[i]) {
            dense[0] = -1;
            break;
        }
    }
}

template <typename OFFSET_TYPE>
class JaggedPosition {
public:
    __aicore__ inline JaggedPosition(int rowId,
                                     GlobalTensor<OFFSET_TYPE>* offsets,
                                     int64_t* offsetsLens,
                                     int64_t* maxLens,
                                     int numJaggedDim,
                                     int innerDenseSize)
    {
        rowId_ = rowId;
        offsetsGT_ = offsets;
        offsetsLens_ = offsetsLens;
        dim_ = numJaggedDim;
        maxLengths_ = maxLens;
        innerDenseSize_ = innerDenseSize;
        _InitDenseShape(maxLens);
        Row2JaggedPos(rowId, pos_);
    }

    __aicore__ inline void GetValidFromTo(int bound, int* result)
    {
        if (pos_[dim_ - 1] >= offsetsLens_[dim_ - 1] - 1) {
            result[0] = -1;
            result[1] = -1;
            result[2] = -1;
            result[3] = -1;
            return;
        }
        int from = offsetsGT_[dim_ - 1].GetValue(pos_[dim_ - 1]) + pos_[dim_];
        int to = offsetsGT_[dim_ - 1].GetValue(pos_[dim_ - 1] + 1);
        if (bound < to) {
            to = bound;
        }

        int densePos[MAX_OFFSETS_CNT + 2];
        JaggedPos2Dense<ROUND_NONE>(pos_, densePos);
        int fromV;
        int toV;
        int rawSize = to - from;
        if (densePos[0] < 0) {
            fromV = -1;
            toV = -1;
        } else {
            fromV = from;
            int validSize =
                (densePos[dim_] + rawSize >= denseShape_[dim_]) ? denseShape_[dim_] - densePos[dim_] : rawSize;
            toV = from + validSize;
        }
        result[0] = from;
        result[1] = to;
        result[2] = fromV;
        result[3] = toV;
    }

    __aicore__ inline int64_t GetDenseOutPtr()
    {
        // 定位dense位置
        int densePos[MAX_OFFSETS_CNT + 2];
        JaggedPos2Dense<ROUND_NONE>(pos_, densePos);

        // 计算拷出偏移位置
        int64_t ptr = 0;
        int64_t stride = 1;
        for (int i = dim_ + 1; i >= 0; i--) {
            int64_t dimPos = densePos[i];
            int64_t dimSize = denseShape_[i];
            ptr += dimPos * stride;
            stride *= dimSize;
        }
        return ptr;
    }

    __aicore__ inline void Update(int rows)
    {
        rowId_ += rows;
        Row2JaggedPos(rowId_, pos_);
    }

    __aicore__ inline int& operator[](size_t index)
    {
        return pos_[index];
    }

protected:
    template <int RoundMode>
    __aicore__ inline void JaggedPos2Dense(int* pos, int* dense)
    {
        dense[0] = pos[0];
        for (int i = 1; i < dim_; i++) {
            int base = offsetsGT_[i - 1].GetValue(pos[i - 1]);
            dense[i] = pos[i] - base;
        }
        dense[dim_] = pos[dim_];
        dense[dim_ + 1] = 0;  // 尾轴inner_dense_size上无偏移

        // 检查dense越界
        if constexpr (RoundMode == ROUND_UP) {
            DenseShapeRoundUp(dense, denseShape_, dim_);
        } else if constexpr (RoundMode == ROUND_DOWN) {
            DenseShapeRoundDown(dense, denseShape_, dim_);
        } else {
            DenseShapeRoundNone(dense, denseShape_, dim_);
        }
    }

    __aicore__ inline void DensePos2Jagged(int* dense, int* pos)
    {
        pos[0] = dense[0];
        for (int i = 1; i < dim_; i++) {
            pos[i] = offsetsGT_[i - 1].GetValue(pos[i - 1]) + dense[i];
        }
        pos[dim_] = dense[dim_];
    }

    __aicore__ inline int64_t JaggedPos2Row(int* pos)
    {
        return offsetsGT_[dim_ - 1].GetValue(pos[dim_ - 1]) + pos[dim_];
    }

    __aicore__ inline void Row2JaggedPos(int target, int* result)
    {
        bool init = false;
        int dim = dim_;
        while (dim > 0) {
            int pos = FindLastLE<OFFSET_TYPE>(target, offsetsGT_[dim - 1], offsetsLens_[dim - 1]);
            if (!init) {
                result[dim] = target - offsetsGT_[dim - 1].GetValue(pos);
                init = true;
            }
            target = pos;
            dim -= 1;
            result[dim] = pos;
        }
    }

private:
    int rowId_;
    int dim_;
    int innerDenseSize_;
    int pos_[MAX_OFFSETS_CNT + 1];
    int denseShape_[MAX_OFFSETS_CNT + 2];
    GlobalTensor<OFFSET_TYPE>* offsetsGT_;
    int64_t* offsetsLens_;
    int64_t* maxLengths_;

    __aicore__ inline void _InitDenseShape(int64_t* maxLengths)
    {
        maxLengths_ = maxLengths;
        for (int i = 0; i < dim_; i++) {
            denseShape_[i + 1] = maxLengths_[i];
        }
        denseShape_[0] = offsetsLens_[0] - 1;
        denseShape_[dim_ + 1] = innerDenseSize_;
    }
};
#endif  // JAGGED_POSITION_H
