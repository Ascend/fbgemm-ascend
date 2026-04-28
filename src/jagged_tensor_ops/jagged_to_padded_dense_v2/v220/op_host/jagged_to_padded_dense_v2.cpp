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

#include "jagged_to_padded_dense_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "enumerate.h"
#include "constant.h"
#include "params_check.h"
#include "ops_log.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    JaggedToPaddedDenseV2TilingData tiling;
    if (!ValuesCheck(context)) {
        return ge::GRAPH_FAILED;
    }

    auto valuesShape = context->GetInputShape(INPUT_INDEX::VALUES)->GetStorageShape();
    int64_t T = valuesShape.GetDim(0);
    int64_t D = valuesShape.GetDim(1);

    tiling.set_total(T);
    tiling.set_innerDenseSize(D);

    if (!OffsetsCheck(context)) {
        return ge::GRAPH_FAILED;
    }
    // len(offsets[0]) - 1
    auto outerDenseSize = context->GetDynamicInputShape(INPUT_INDEX::OFFSETS, 0)->GetStorageShape().GetDim(0) - 1;
    tiling.set_outerDenseSize(outerDenseSize);

    auto maxLengthsObj = context->GetAttrs()->GetListInt(ATTR_INDEX::MAX_LENGTHS);
    int size = maxLengthsObj->GetSize();
    const int64_t* _maxLengths = maxLengthsObj->GetData();
    int64_t maxLengths[MAX_OFFSETS_CNT];
    int64_t offsetsLens[MAX_OFFSETS_CNT];
    for (int i = 0; i < size; i++) {
        maxLengths[i] = _maxLengths[i];
        offsetsLens[i] = context->GetDynamicInputShape(INPUT_INDEX::OFFSETS, i)->GetStorageShape().GetDim(0);
    }
    tiling.set_offsetCnt(size);
    tiling.set_maxLengths(maxLengths);
    tiling.set_offsetsLens(offsetsLens);

    float paddingValue = *context->GetAttrs()->GetFloat(ATTR_INDEX::PADDING_VALUE);
    tiling.set_paddingValue(paddingValue);

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ubCanUsed;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
    ubCanUsed = ubCanUsed - RESERVER_UB_SIZE;
    if (ubCanUsed < MIN_UB_USED_SIZE) {
        OPS_LOG_E("jagged_to_padded_dense",
                  "ubCanUsed is less than MIN_UB_USED_SIZE, ubCanUsed: %ld, MIN_UB_USED_SIZE: %ld",
                  ubCanUsed, MIN_UB_USED_SIZE);
        return ge::GRAPH_FAILED;
    }
    tiling.set_ubCanUsed(ubCanUsed);
    
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);

    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;

    context->SetBlockDim(coreNum);

    OPS_LOG_E_IF_NULL("context->GetRawTilingData()", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    auto maxLengthsObj = context->GetAttrs()->GetListInt(ATTR_INDEX::MAX_LENGTHS);
    int size = maxLengthsObj->GetSize();
    const int64_t* maxLengths = maxLengthsObj->GetData();
    // len(offsets[0]) - 1
    auto outerDenseSize = context->GetDynamicInputShape(INPUT_INDEX::OFFSETS, 0)->GetDim(0) - 1;
    auto innerDenseSize = context->GetInputShape(INPUT_INDEX::VALUES)->GetDim(1);  // (T, D)

    gert::Shape* outShape = context->GetOutputShape(OUTPUT_INDEX::DENSE);
    outShape->SetDimNum(size + 2);  // (outer, max_lengths[0], ..., inner)
    outShape->SetDim(0, outerDenseSize);
    outShape->SetDim(size + 1, innerDenseSize);
    for (int i = 0; i < size; i++) {
        outShape->SetDim(i+1, maxLengths[i]);
    }

    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class JaggedToPaddedDenseV2 : public OpDef {
public:
    explicit JaggedToPaddedDenseV2(const char* name) : OpDef(name)
    {
        this->Input("values")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT64, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("offsets")
            .ParamType(DYNAMIC)
            .DataTypeList({ge::DT_INT64, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .Follow("values", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Attr("max_length").ListInt();
        this->Attr("padding_value").Float();

        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
#ifdef SUPPORT_950
        this->AICore().AddConfig("ascend950");
#endif
    }
};

OP_ADD(JaggedToPaddedDenseV2);
}  // namespace ops
