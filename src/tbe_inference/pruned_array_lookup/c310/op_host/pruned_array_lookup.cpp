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

#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"
#include "pruned_array_lookup_tiling.h"

namespace optiling {

constexpr int RESERVER_UB_SIZE = 20 * 1024;
constexpr int DCACHE_SIZE = 128 * 1024;

// input index
constexpr int INDICES_INDEX = 0;
constexpr int OFFSETS_INDEX = 1;
constexpr int INDEX_REMAPPINGS_INDEX = 2;
constexpr int INDEX_REMAPPINGS_OFFSETS_INDEX = 3;
constexpr int OUTPUT_INDEX = 4;

// input/output dim
constexpr int INDICES_DIM = 1;
constexpr int OFFSETS_DIM = 1;
constexpr int INDEX_REMAPPINGS_DIM = 1;
constexpr int INDEX_REMAPPINGS_OFFSETS_DIM = 1;

static ge::graphStatus ShapeTilingFunc(gert::TilingContext* context,
                                       PrunedArrayLookupTilingData& tilingData)
{
    OPS_LOG_E_IF_NULL("indicesShape", context->GetInputShape(INDICES_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indicesTensor", context->GetInputTensor(INDICES_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetsShape", context->GetInputShape(OFFSETS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetsTensor", context->GetInputTensor(OFFSETS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indexRemappingsShape", context->GetInputShape(INDEX_REMAPPINGS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indexRemappingsTensor", context->GetInputTensor(INDEX_REMAPPINGS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indexRemappingsOffsetsShape", context->GetInputShape(INDEX_REMAPPINGS_OFFSETS_INDEX),
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indexRemappingsOffsetsTensor", context->GetInputTensor(INDEX_REMAPPINGS_OFFSETS_INDEX),
                      return ge::GRAPH_FAILED);

    const gert::StorageShape* indicesShape = context->GetInputShape(INDICES_INDEX);
    const gert::StorageShape* offsetsShape = context->GetInputShape(OFFSETS_INDEX);
    const gert::StorageShape* indexRemappingsShape = context->GetInputShape(INDEX_REMAPPINGS_INDEX);
    const gert::StorageShape* indexRemappingsOffsetsShape = context->GetInputShape(INDEX_REMAPPINGS_OFFSETS_INDEX);

    auto indicesStorageShape = indicesShape->GetStorageShape();
    auto offsetsStorageShape = offsetsShape->GetStorageShape();
    auto indexRemappingsStorageShape = indexRemappingsShape->GetStorageShape();
    auto indexRemappingsOffsetsStorageShape = indexRemappingsOffsetsShape->GetStorageShape();

    int64_t indicesLen = indicesStorageShape.GetDim(0);
    int64_t offsetsLen = offsetsStorageShape.GetDim(0);
    int64_t indexRemappingsOffsetsLen = indexRemappingsOffsetsStorageShape.GetDim(0);
    int64_t indexRemappingsLen = indexRemappingsStorageShape.GetDim(0);

    // 获取数据类型
    OPS_LOG_E_IF_NULL("indicesType", context->GetInputDesc(INDICES_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetsType", context->GetInputDesc(OFFSETS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indexRemappingsType", context->GetInputDesc(INDEX_REMAPPINGS_INDEX), return ge::GRAPH_FAILED);
    auto indicesType = context->GetInputDesc(INDICES_INDEX)->GetDataType();
    auto offsetsType = context->GetInputDesc(OFFSETS_INDEX)->GetDataType();
    auto indexRemappingsType = context->GetInputDesc(INDEX_REMAPPINGS_INDEX)->GetDataType();

    // 检查数据类型
    OPS_CHECK(indicesType != offsetsType,
              OPS_LOG_E("Tiling Debug",
                        "The datatype of indices and offsets must be same."),
              return ge::GRAPH_FAILED);

    OPS_CHECK(indicesType != ge::DT_INT32 && indicesType != ge::DT_INT64,
              OPS_LOG_E("Tiling Debug",
                        "Invalid data type. indices must only support int64 and int32."),
              return ge::GRAPH_FAILED);

    // 检查维度
    OPS_CHECK(indicesStorageShape.GetDimNum() != INDICES_DIM,
              OPS_LOG_E("Tiling Debug",
                        "Expected dim for indices is %d, but the actual dim is %d.",
                        INDICES_DIM, indicesStorageShape.GetDimNum()),
              return ge::GRAPH_FAILED);
    OPS_CHECK(indexRemappingsStorageShape.GetDimNum() != INDEX_REMAPPINGS_DIM,
              OPS_LOG_E("Tiling Debug",
                        "Expected dim for index_remappings is %d, but the actual dim is %d.",
                        INDEX_REMAPPINGS_DIM, indexRemappingsStorageShape.GetDimNum()),
              return ge::GRAPH_FAILED);
    OPS_CHECK(offsetsStorageShape.GetDimNum() != OFFSETS_DIM,
              OPS_LOG_E("Tiling Debug",
                        "Expected dim for offsets is %d, but the actual dim is %d.",
                        OFFSETS_DIM, offsetsStorageShape.GetDimNum()),
              return ge::GRAPH_FAILED);
    OPS_CHECK(indexRemappingsOffsetsStorageShape.GetDimNum() != INDEX_REMAPPINGS_OFFSETS_DIM,
              OPS_LOG_E("Tiling Debug",
                        "Expected dim for index_remappings_offsets is %d, but the actual dim is %d.",
                        INDEX_REMAPPINGS_OFFSETS_DIM, indexRemappingsOffsetsStorageShape.GetDimNum()),
              return ge::GRAPH_FAILED);

    // 检查长度关系
    OPS_CHECK(indicesLen <= 0,
              OPS_LOG_E("Tiling Debug",
                        "indices length must be greater than 0, got %lld.",
                        indicesLen),
              return ge::GRAPH_FAILED);
    int64_t batchNum = offsetsLen - 1;
    OPS_CHECK(batchNum <= 0,
              OPS_LOG_E("Tiling Debug",
                        "offsets length must be greater than 0, got %lld.",
                        batchNum),
              return ge::GRAPH_FAILED);
    int64_t tableNum = indexRemappingsOffsetsLen - 1;
    int64_t batchPerTable = batchNum / tableNum;
    tilingData.set_batchNum(offsetsLen - 1);
    tilingData.set_batchPerTable(batchPerTable);
    tilingData.set_tableNum(indexRemappingsOffsetsLen - 1);
    tilingData.set_indicesLen(indicesLen);
    tilingData.set_offsetsLen(offsetsLen);
    tilingData.set_indexRemappingsLen(indexRemappingsLen);
    tilingData.set_indexRemappingsOffsetsLen(indexRemappingsOffsetsLen);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PrunedArrayLookupTilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("TilingContext", context, return ge::GRAPH_FAILED);

    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t systemWorkspacesSize = platformInfo.GetLibApiWorkSpaceSize();

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);
    currentWorkspace[0] = systemWorkspacesSize;

    PrunedArrayLookupTilingData tiling;
    // Shape and dType
    if (ShapeTilingFunc(context, tiling) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Get UB size for kernel
    uint64_t ubCanUsed;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
    ubCanUsed = ubCanUsed - RESERVER_UB_SIZE;

    // tiling data
    size_t coreNum = platformInfo.GetCoreNumAiv();
    OPS_CHECK(coreNum == 0,
              OPS_LOG_E("", "Core num is 0."),
              return ge::GRAPH_FAILED);
    
    // 以batch维度分核 TODO delete if test ok
    auto batchNum = tiling.get_batchNum();
    int64_t actualCoreNum = batchNum < coreNum ? batchNum : coreNum;
    int64_t batchNumPerCore = batchNum / actualCoreNum;
    int64_t bigCore = batchNum % actualCoreNum;
    tiling.set_bigCore(bigCore);
    tiling.set_batchNumPerCore(batchNumPerCore);

    context->SetBlockDim(actualCoreNum);
    context->SetLocalMemorySize(DCACHE_SIZE);

    auto tilingData = context->GetRawTilingData();
    OPS_LOG_E_IF_NULL("tilingData", tilingData, return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(tilingData->GetData(), tilingData->GetCapacity());

    tilingData->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus PrunedArrayLookupInferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("infoshape context", context, return ge::GRAPH_FAILED);
    // Get output shape
    gert::Shape* outputShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("outputShape", outputShape, return ge::GRAPH_FAILED);

    const gert::Shape* indicesShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("indicesShape", indicesShape, return ge::GRAPH_FAILED);
    int64_t outputSize = static_cast<int64_t>(indicesShape->GetDim(0));

    outputShape->SetDimNum(1);
    outputShape->SetDim(0, outputSize);

    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class PrunedArrayLookup : public OpDef {
public:
    explicit PrunedArrayLookup(const char* name) : OpDef(name)
    {
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("offsets")
            .ParamType(REQUIRED)
            .Follow("indices", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("index_remappings")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("index_remappings_offsets")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64}) // 仅支持int64
            .FormatList({ge::FORMAT_ND});

        this->Output("dense_indices")
            .ParamType(REQUIRED)
            .Follow("indices", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});

        this->SetInferShape(ge::PrunedArrayLookupInferShape);

        this->AICore().SetTiling(optiling::PrunedArrayLookupTilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(PrunedArrayLookup);

}  // namespace ops
