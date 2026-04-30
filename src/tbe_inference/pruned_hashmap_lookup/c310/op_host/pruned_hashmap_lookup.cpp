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
#include "pruned_hashmap_lookup_tiling.h"

constexpr int RESERVER_UB_SIZE = 20 * 1024;
constexpr int DCACHE_SIZE = 128 * 1024;

// input index
constexpr int INDICES_INDEX = 0;
constexpr int OFFSETS_INDEX = 1;
constexpr int HASH_TABLE_INDEX = 2;
constexpr int HASH_TABLE_OFFSETS_INDEX = 3;
constexpr int OUTPUT_INDEX = 4;

// input/output dim
constexpr int INDICES_DIM = 1;
constexpr int OFFSETS_DIM = 1;
constexpr int HASH_TABLE_DIM = 2;
constexpr int HASH_TABLE_OFFSETS_DIM = 1;

namespace optiling {

static ge::graphStatus ShapeTilingFunc(gert::TilingContext* context,
                                       PrunedHashmapLookupTilingData& tilingData)
{
    OPS_LOG_E_IF_NULL("indicesShape", context->GetInputShape(INDICES_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indicesTensor", context->GetInputTensor(INDICES_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetsShape", context->GetInputShape(OFFSETS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetsTensor", context->GetInputTensor(OFFSETS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("hashTableShape", context->GetInputShape(HASH_TABLE_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("hashTableTensor", context->GetInputTensor(HASH_TABLE_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("hashTableOffsetsShape", context->GetInputShape(HASH_TABLE_OFFSETS_INDEX),
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("hashTableOffsetsTensor", context->GetInputTensor(HASH_TABLE_OFFSETS_INDEX),
                      return ge::GRAPH_FAILED);

    const gert::StorageShape* indicesShape = context->GetInputShape(INDICES_INDEX);
    const gert::StorageShape* offsetsShape = context->GetInputShape(OFFSETS_INDEX);
    const gert::StorageShape* hashTableShape = context->GetInputShape(HASH_TABLE_INDEX);
    const gert::StorageShape* hashTableOffsetsShape = context->GetInputShape(HASH_TABLE_OFFSETS_INDEX);

    auto indicesStorageShape = indicesShape->GetStorageShape();
    auto offsetsStorageShape = offsetsShape->GetStorageShape();
    auto hashTableStorageShape = hashTableShape->GetStorageShape();
    auto hashTableOffsetsStorageShape = hashTableOffsetsShape->GetStorageShape();

    int64_t indicesLen = indicesStorageShape.GetDim(0);
    int64_t offsetsLen = offsetsStorageShape.GetDim(0);
    int64_t hashTableOffsetsLen = hashTableOffsetsStorageShape.GetDim(0);
    int64_t hashTableLen = hashTableStorageShape.GetDim(0) * 2;

    // 获取数据类型
    OPS_LOG_E_IF_NULL("indicesType", context->GetInputDesc(INDICES_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetsType", context->GetInputDesc(OFFSETS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("hashTableType", context->GetInputDesc(HASH_TABLE_INDEX), return ge::GRAPH_FAILED);
    auto indicesType = context->GetInputDesc(INDICES_INDEX)->GetDataType();
    auto offsetsType = context->GetInputDesc(OFFSETS_INDEX)->GetDataType();
    auto hashTableType = context->GetInputDesc(HASH_TABLE_INDEX)->GetDataType();

    // 检查数据类型
    OPS_CHECK(indicesType != offsetsType,
              OPS_LOG_E("Tiling Debug",
                        "The datatype of indices, offsets and hash_table must be the same."),
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
    OPS_CHECK(offsetsStorageShape.GetDimNum() != OFFSETS_DIM,
              OPS_LOG_E("Tiling Debug",
                        "Expected dim for offsets is %d, but the actual dim is %d.",
                        OFFSETS_DIM, offsetsStorageShape.GetDimNum()),
              return ge::GRAPH_FAILED);
    OPS_CHECK(hashTableOffsetsStorageShape.GetDimNum() != HASH_TABLE_OFFSETS_DIM,
              OPS_LOG_E("Tiling Debug",
                        "Expected dim for hash_table_offsets is %d, but the actual dim is %d.",
                        HASH_TABLE_OFFSETS_DIM, hashTableOffsetsStorageShape.GetDimNum()),
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
    int64_t tableNum = hashTableOffsetsLen - 1;
    int64_t batchPerTable = batchNum / tableNum;
    tilingData.set_batchNum(offsetsLen - 1);
    tilingData.set_batchPerTable(batchPerTable);
    tilingData.set_tableNum(hashTableOffsetsLen - 1);
    tilingData.set_indicesLen(indicesLen);
    tilingData.set_offsetsLen(offsetsLen);
    tilingData.set_hashTableLen(hashTableLen);
    tilingData.set_hashTableOffsetsLen(hashTableOffsetsLen);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PrunedHashmapLookupTilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("TilingContext", context, return ge::GRAPH_FAILED);

    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t systemWorkspacesSize = platformInfo.GetLibApiWorkSpaceSize();

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);
    currentWorkspace[0] = systemWorkspacesSize;

    PrunedHashmapLookupTilingData tiling;
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
static ge::graphStatus PrunedHashmapLookupInferShape(gert::InferShapeContext* context)
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
class PrunedHashmapLookup : public OpDef {
public:
    explicit PrunedHashmapLookup(const char* name) : OpDef(name)
    {
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("offsets")
            .ParamType(REQUIRED)
            .Follow("indices", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("hash_table")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("hash_table_offsets")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64}) // 仅支持int64
            .FormatList({ge::FORMAT_ND});

        this->Output("dense_indices")
            .ParamType(REQUIRED)
            .Follow("indices", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});

        this->SetInferShape(ge::PrunedHashmapLookupInferShape);

        this->AICore().SetTiling(optiling::PrunedHashmapLookupTilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(PrunedHashmapLookup);

}  // namespace ops
