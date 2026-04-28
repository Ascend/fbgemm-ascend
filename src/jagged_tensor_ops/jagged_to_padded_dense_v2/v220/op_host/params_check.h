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

#ifndef JAGGED_PARAMS_CHECK_H
#define JAGGED_PARAMS_CHECK_H
#include "constant.h"
#include "ops_log.h"

static bool ValuesCheck(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("valuesShape", context->GetInputShape(INPUT_INDEX::VALUES), return false);
    OPS_LOG_E_IF_NULL("valuesTensor", context->GetInputTensor(INPUT_INDEX::VALUES), return false);

    auto valuesShape = context->GetInputShape(INPUT_INDEX::VALUES)->GetStorageShape();

    if (valuesShape.GetDimNum() != SUPPORT_EMBEDDING_DIM_NUM) {
        OPS_LOG_E("jagged_to_padded_dense_v2",
                  "Only supports values with dim 2, but got %d",
                  valuesShape.GetDimNum());
        return false;
    }

    // values (T, D)
    int64_t T = valuesShape.GetDim(0);
    int64_t D = valuesShape.GetDim(1);

    if (D > MAX_D) {
        OPS_LOG_E("jagged_to_padded_dense_v2",
                  "Only supports values(T, D), where D <= 2048, but got %d",
                  D);
        return false;
    }
    return true;
}

static bool OffsetsCheck(gert::TilingContext* context)
{
    int offsetsCnt = 0;
    while (offsetsCnt < MAX_OFFSETS_CNT) {
        auto offset = context->GetDynamicInputTensor(INPUT_INDEX::OFFSETS, offsetsCnt);
        if (offset == nullptr) {
            break;
        }
        offsetsCnt++;
    }

    if (offsetsCnt < MIN_OFFSETS_CNT || offsetsCnt > MAX_OFFSETS_CNT) {
        OPS_LOG_E("jagged_to_padded_dense_v2",
                  "Only supports %d <= len(offsets) <= %d, but got %d",
                  MIN_OFFSETS_CNT, MAX_OFFSETS_CNT, offsetsCnt);
        return false;
    }

    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return false);
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs()->GetListInt(ATTR_INDEX::MAX_LENGTHS), return false);
    size_t maxLengthsCnt = context->GetAttrs()->GetListInt(ATTR_INDEX::MAX_LENGTHS)->GetSize();
    if (offsetsCnt != maxLengthsCnt) {
        OPS_LOG_E("jagged_to_padded_dense_v2",
                  "len(offsets), %d != len(max_lengths), %d",
                  offsetsCnt, maxLengthsCnt);
        return false;
    }

    return true;
}

#endif  // JAGGED_PARAMS_CHECK_H
