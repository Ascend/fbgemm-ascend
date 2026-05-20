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

#include <cstdint>
#include <cmath>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "ops_log.h"
#include "direct_mapped_lxu_cache_lookup_tiling.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("linear_cache_indices", context->GetInputShape(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("lxu_cache_state", context->GetInputShape(1), return ge::GRAPH_FAILED);

    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("invalid_index", context->GetAttrs()->GetInt(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("gather_cache_status", context->GetAttrs()->GetBool(1), return ge::GRAPH_FAILED);

    int64_t invalid_index = *context->GetAttrs()->GetInt(0);
    bool gather_cache_status = *context->GetAttrs()->GetBool(1);

    DirectMappedLxuCacheLookupTilingData tiling;
    if (gather_cache_status) {
        OPS_LOG_E_IF_NULL("uvm_cache_stats", context->GetOptionalInputTensor(2), return ge::GRAPH_FAILED);
        tiling.set_uvm_len(context->GetInputShape(2)->GetStorageShape().GetDim(0));
    } else {
        tiling.set_uvm_len(0);
    }

    tiling.set_invalid_index(invalid_index);
    tiling.set_gather_cache_status(gather_cache_status);

    const gert::Shape linear_cache_indices_shape = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape lxu_cache_state_shape = context->GetInputShape(1)->GetStorageShape();

    int32_t indices = linear_cache_indices_shape.GetDim(0);
    int32_t slots = lxu_cache_state_shape.GetDim(0);
    tiling.set_indices(indices);
    tiling.set_slots(slots);

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;

    ge::DataType inputType = context->GetInputTensor(0)->GetDataType();
    context->SetTilingKey(inputType);
    context->SetBlockDim(coreNum);

    OPS_LOG_E_IF_NULL("Raw tiling data", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    const gert::Shape* linear_cache_indices_shape = context->GetInputShape(0);
    const gert::Shape* lxu_cache_state_shape = context->GetInputShape(1);

    OPS_LOG_E_IF_NULL("linear_cache_indices_shape", linear_cache_indices_shape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("lxu_cache_state_shape", lxu_cache_state_shape, return ge::GRAPH_FAILED);

    if (linear_cache_indices_shape->GetDimNum() != 1) {
        OPS_LOG_E("", "[ERROR], input shape must be 1D, got %lld", linear_cache_indices_shape->GetDimNum());
        return ge::GRAPH_FAILED;
    }

    if (lxu_cache_state_shape->GetDimNum() != 2) {  // 2表示原始索引及其对应的槽位信息
        OPS_LOG_E("", "[ERROR], indices shape must be 2D, got %lld", lxu_cache_state_shape->GetDimNum());
        return ge::GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("outputShape", outputShape, return ge::GRAPH_FAILED);

    outputShape->SetDimNum(linear_cache_indices_shape->GetDimNum());
    outputShape->SetDim(0, linear_cache_indices_shape->GetDim(0));

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class DirectMappedLxuCacheLookup : public OpDef {
public:
    explicit DirectMappedLxuCacheLookup(const char* name) : OpDef(name)
    {
        this->Input("linear_cache_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("lxu_cache_state")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("invalid_index").AttrType(REQUIRED).Int(-1);
        this->Attr("gather_cache_status").AttrType(REQUIRED).Bool(false);
        this->Input("uvm_cache_stats")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("lxu_cache_locations")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(DirectMappedLxuCacheLookup);
}  // namespace ops
