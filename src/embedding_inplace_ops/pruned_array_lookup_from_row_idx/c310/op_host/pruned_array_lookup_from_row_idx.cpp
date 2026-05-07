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

#include "pruned_array_lookup_from_row_idx_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace {
constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t MIN_THREADS_PER_BLOCK = 512;
constexpr int32_t SMALL_DATA_LENGTH = 2048;
constexpr int32_t THREADS_PER_WARP = 32;
constexpr int32_t ADD_CORE_FACTOR = 4;
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("inputShape0", context->GetInputShape(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("inputTensor0", context->GetInputTensor(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("inputTensor2", context->GetInputTensor(2), return ge::GRAPH_FAILED);

    const int64_t numIndices = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    const ge::DataType rowType = context->GetInputTensor(0)->GetDataType();
    const ge::DataType remapType = context->GetInputTensor(2)->GetDataType();

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t coreNum = static_cast<int32_t>(ascendPlatform.GetCoreNumAiv());

    int32_t blockDim = 0;
    uint32_t threadsPerBlock = 0;
    if (numIndices < SMALL_DATA_LENGTH) {
        threadsPerBlock = MIN_THREADS_PER_BLOCK;
        blockDim = 1;
    } else {
        threadsPerBlock = MAX_THREADS_PER_BLOCK;
        blockDim = static_cast<int32_t>(
            std::min((numIndices + static_cast<int64_t>(ADD_CORE_FACTOR) * THREADS_PER_WARP - 1) /
                         (static_cast<int64_t>(THREADS_PER_WARP) * static_cast<int64_t>(ADD_CORE_FACTOR)),
                     static_cast<int64_t>(coreNum)));
    }

    if (blockDim <= 0) {
        OPS_LOG_E("[ERROR] Invalid blockDim for PrunedArrayLookupFromRowIdx", NULL);
        return ge::GRAPH_FAILED;
    }

    int32_t elemsPerBlock = static_cast<int32_t>((numIndices + static_cast<int64_t>(blockDim) - 1) /
                                                 static_cast<int64_t>(blockDim));
    elemsPerBlock = (elemsPerBlock + THREADS_PER_WARP - 1) / THREADS_PER_WARP * THREADS_PER_WARP;

    int32_t tilingKey = -1;
    if (rowType == ge::DT_INT64 && remapType == ge::DT_INT64) {
        tilingKey = 0;
    } else if (rowType == ge::DT_INT64 && remapType == ge::DT_INT32) {
        tilingKey = 1;
    } else if (rowType == ge::DT_INT32 && remapType == ge::DT_INT64) {
        tilingKey = 2;
    } else if (rowType == ge::DT_INT32 && remapType == ge::DT_INT32) {
        tilingKey = 3;
    } else {
        OPS_LOG_E("[ERROR] PrunedArrayLookupFromRowIdx: unsupported dtype combo for row/remap.", NULL);
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(tilingKey);

    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("workspaceSize", workspaceSize, return ge::GRAPH_FAILED);
    workspaceSize[0] = ascendPlatform.GetLibApiWorkSpaceSize();

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    PrunedArrayLookupFromRowIdxTilingData tiling;
    tiling.set_numIndices(numIndices);
    tiling.set_elemsPerBlock(elemsPerBlock);
    tiling.set_threadsPerBlock(threadsPerBlock);

    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const gert::Shape* inShape = context->GetInputShape(0);
    gert::Shape* outShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("inShape", inShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("outShape", outShape, return ge::GRAPH_FAILED);
    *outShape = *inShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const auto rowType = context->GetInputDataType(0);
    if (ge::GRAPH_SUCCESS != context->SetOutputDataType(0, rowType)) {
        return ge::GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class PrunedArrayLookupFromRowIdx : public OpDef {
public:
    explicit PrunedArrayLookupFromRowIdx(const char* name) : OpDef(name)
    {
        this->Input("update_row_indices")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("update_table_indices")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("index_remappings")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("index_remappings_offsets")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dense_indices")
            .ParamType(REQUIRED)
            .Follow("update_row_indices", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(PrunedArrayLookupFromRowIdx);
}  // namespace ops
