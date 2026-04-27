/* Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include "group_index_select_dim0_backward_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "ops_log.h"

namespace optiling {
constexpr int RESERVE_UB_SIZE = 20 * 1024;
constexpr uint32_t MAX_GROUP_NUM = 32;
constexpr int32_t GRAD_INDEX = 0;
constexpr int32_t INDICES_INDEX = 1;

template <typename T>
bool GetValueAttr(const gert::RuntimeAttrs *attrs, uint32_t idx, T &value)
{
    const T *ptr = attrs->GetAttrPointer<T>(idx);
    OPS_LOG_E_IF_NULL("attr ptr is nullptr", ptr, return false);
    value = *ptr;
    return true;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    GroupIndexSelectDim0BackwardTilingData tiling;

    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    int64_t groupNum = 0;

    if (GetValueAttr<int64_t>(attrs, 0, groupNum) == false) {
        return ge::GRAPH_FAILED;
    }

    int32_t numGradRows[MAX_GROUP_NUM];
    int32_t numIndices[MAX_GROUP_NUM];
    int32_t numInnerDim[MAX_GROUP_NUM];

    for (int64_t i = 0; i  < groupNum; i++) {
        OPS_LOG_E_IF_NULL("grad outputs shape", context->GetDynamicInputShape(GRAD_INDEX, i), return ge::GRAPH_FAILED);
        OPS_LOG_E_IF_NULL("indices group shape", context->GetDynamicInputShape(INDICES_INDEX, i), return ge::GRAPH_FAILED);
        const gert::Shape gradOutputsShape = context->GetDynamicInputShape(GRAD_INDEX, i)->GetStorageShape();
        const gert::Shape indicesGroupShape = context->GetDynamicInputShape(INDICES_INDEX, i)->GetStorageShape();

        numGradRows[i] = gradOutputsShape.GetDim(0);
        numIndices[i] = indicesGroupShape.GetDim(0);
        int64_t gradDimNum = gradOutputsShape.GetDimNum();
        int64_t gradDim = 1;
        for (int64_t d = 1; d < gradDimNum; d++) {
            gradDim *= gradOutputsShape.GetDim(d);
        }
        numInnerDim[i] = gradDim;
    }

    tiling.set_groupGradRows(numGradRows);
    tiling.set_groupIndicesLen(numIndices);
    tiling.set_groupInnerDim(numInnerDim);

    ge::DataType inputType = context->GetInputTensor(0)->GetDataType();
    int dataTypeTilingKey = inputType;

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;

    uint64_t ubCanUsed;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
    ubCanUsed = ubCanUsed - RESERVE_UB_SIZE;

    tiling.set_groupNum(groupNum);
    tiling.set_ubCanUsed(ubCanUsed);

    context->SetTilingKey(dataTypeTilingKey);
    context->SetBlockDim(coreNum);

    if (context->GetRawTilingData() == nullptr) {
        OPS_LOG_E("[ERROR]", "GetRawTilingData Failed!");
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    const gert::Shape *inputGroupShape = context->GetInputShape(GRAD_INDEX);
    const gert::Shape *indicesGroupShape = context->GetInputShape(INDICES_INDEX);

    OPS_LOG_E_IF_NULL("inputGroupShape", inputGroupShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indicesGroupShape", indicesGroupShape, return ge::GRAPH_FAILED);

    int64_t inputDimNum = inputGroupShape->GetDimNum();
    int64_t indicesDimNum = indicesGroupShape->GetDimNum();

    if (inputDimNum < 3) { // dim3，每个group内，再list，再tensor
        OPS_LOG_E("", "[ERROR], input shape must have at least 3D, got %lld", inputDimNum);
        return ge::GRAPH_FAILED;
    }

    if (indicesDimNum != 2) { // dim2，每个group内，再tensor
        OPS_LOG_E("", "[ERROR], indices shape must be 2D, got %lld", indicesDimNum);
        return ge::GRAPH_FAILED;
    }

    int64_t numGroups = inputGroupShape->GetDim(0);
    int64_t indicesNumGroups = indicesGroupShape->GetDim(0);

    if (numGroups != indicesNumGroups) {
        OPS_LOG_E("", "[ERROR], numGroups mismatch: input %lld vs indices %lld", numGroups, indicesNumGroups);
        return ge::GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("outputShape", outputShape, return ge::GRAPH_FAILED);

    outputShape->SetDimNum(inputDimNum);
    outputShape->SetDim(0, inputGroupShape->GetDim(0));
    outputShape->SetDim(1, indicesGroupShape->GetDim(1));
    for (size_t i = 2; i < inputDimNum; i++) { // 此处维度2开始，是指每个group内，再list内的tensor
        outputShape->SetDim(i, inputGroupShape->GetDim(i));
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(GRAD_INDEX);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class GroupIndexSelectDim0Backward : public OpDef {
public:
    explicit GroupIndexSelectDim0Backward(const char* name) : OpDef(name)
    {
        this->Input("gradOutputs")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indicesGroups")
            .ParamType(DYNAMIC)
            .DataTypeList({ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("inputReturnGroups")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("groupNum").Int();

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
#ifdef SUPPORT_950
        this->AICore().AddConfig("ascend950");
#endif
    }
};

OP_ADD(GroupIndexSelectDim0Backward);
}  // namespace ops
