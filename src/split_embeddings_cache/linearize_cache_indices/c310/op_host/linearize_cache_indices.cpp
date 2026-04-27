/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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
#include <algorithm>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "ops_log.h"
#include "linearize_cache_indices_tiling.h"

namespace optiling {
constexpr int32_t INDICES_BASE_OFFSET_ATTR_INDEX = 0;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("inputShape", context->GetInputShape(1), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("inputTensor", context->GetInputTensor(1), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("inputTensor0", context->GetInputTensor(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("inputTensor2", context->GetInputTensor(2), return ge::GRAPH_FAILED);

    auto cacheTensor = context->GetInputTensor(0);
    auto indicesTensor = context->GetInputTensor(1);
    auto tableOffsetsTensor = context->GetInputTensor(2);

    ge::DataType cacheDataType = cacheTensor->GetDataType();
    ge::DataType indicesDataType = indicesTensor->GetDataType();
    ge::DataType tableOffsetsDataType = tableOffsetsTensor->GetDataType();

    if (indicesDataType != ge::DT_INT32 && indicesDataType != ge::DT_INT64) {
        OPS_LOG_E(context, "[ERROR]Invalid indices data type, LinearizeCacheIndices only support int64 and int32.");
        return ge::GRAPH_FAILED;
    }

    if (cacheDataType != ge::DT_INT64) {
        OPS_LOG_E(context,
            "[ERROR]Invalid cache_hash_size_cumsum data type, LinearizeCacheIndices only support int64.");
        return ge::GRAPH_FAILED;
    }
    if (tableOffsetsDataType != ge::DT_INT32 && tableOffsetsDataType != ge::DT_INT64) {
        OPS_LOG_E(context,
                  "[ERROR]Invalid table_offsets data type, LinearizeCacheIndices only support int64 and int32.");
        return ge::GRAPH_FAILED;
    }

    int64_t numIndices = context->GetInputShape(1)->GetOriginShape().GetShapeSize();

    int64_t numTables = context->GetInputShape(0)->GetOriginShape().GetShapeSize() - 1;

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t maxCores = ascendPlatform.GetCoreNumAiv();

    size_t coreNum =
        numIndices > 0 ? std::min(maxCores, std::max(static_cast<size_t>(numIndices) / 32, static_cast<size_t>(1))) : 0;
    if (coreNum == 0) {
        OPS_LOG_E(context, "[ERROR] need more than 0 ai core");
        return ge::GRAPH_FAILED;
    }

    // 获取属性
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    int64_t indicesBaseOffset = 0;
    const int64_t* indicesBaseOffsetPtr = attrs->GetInt(INDICES_BASE_OFFSET_ATTR_INDEX);
    if (indicesBaseOffsetPtr != nullptr) {
        indicesBaseOffset = *indicesBaseOffsetPtr;
    }

    int64_t tableOffsetsSize = numTables - 1;

    LinearizeCacheIndicesTilingData tiling;
    tiling.set_numIndices(numIndices);
    tiling.set_numTables(numTables);
    tiling.set_indicesBaseOffset(indicesBaseOffset);
    tiling.set_tableOffsetsSize(tableOffsetsSize);

    context->SetBlockDim(coreNum);
    
    gert::TilingData* rawTilingData = context->GetRawTilingData();
    if (rawTilingData == nullptr) {
        OPS_LOG_E(context, "[ERROR] Raw tiling data is null");
        return ge::GRAPH_FAILED;
    }
    
    size_t tilingDataSize = tiling.GetDataSize();
    if (rawTilingData->GetCapacity() < tilingDataSize) {
        OPS_LOG_E(context, "[ERROR] Insufficient capacity for tiling data");
        return ge::GRAPH_FAILED;
    }
    
    tiling.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingDataSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    const gert::Shape* indicesShape = context->GetInputShape(1);
    OPS_LOG_E_IF_NULL("indicesShape", indicesShape, return ge::GRAPH_FAILED);

    gert::Shape* outputShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("outputShape", outputShape, return ge::GRAPH_FAILED);

    int64_t numIndices = indicesShape->GetDim(0);
    outputShape->SetDimNum(1);
    outputShape->SetDim(0, numIndices);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    auto inputDataType = context->GetInputDataType(1);
    if (ge::GRAPH_SUCCESS != context->SetOutputDataType(0, inputDataType)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
class LinearizeCacheIndices : public OpDef {
public:
    explicit LinearizeCacheIndices(const char* name) : OpDef(name)
    {
        this->Input("cache_hash_size_cumsum")
            .ParamType(REQUIRED)
            .FormatList({ge::FORMAT_ND})
            .DataTypeList({ge::DT_INT64})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("indices")
            .ParamType(REQUIRED)
            .FormatList({ge::FORMAT_ND})
            .DataTypeList({ge::DT_INT64, ge::DT_INT32})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("table_offsets")
            .ParamType(REQUIRED)
            .FormatList({ge::FORMAT_ND})
            .DataTypeList({ge::DT_INT64, ge::DT_INT32})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("linearized_indices")
            .ParamType(REQUIRED)
            .FormatList({ge::FORMAT_ND})
            .DataType({ge::DT_INT64})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("indices_base_offset").Int();

        this->SetInferShape(optiling::InferShape).SetInferDataType(optiling::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(LinearizeCacheIndices);
}  // namespace ops