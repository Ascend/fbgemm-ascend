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
#include <cmath>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "ops_log.h"
#include "linearize_cache_indices_from_row_idx_tiling.h"

namespace {
    constexpr uint32_t MAX_THREAD_COUNT = 2048;
    constexpr uint32_t MAX_BLOCK_COUNT = 65535;
}

// GetInputShape / GetInputTensor 的编号与 OpDef 中 Input() 声明顺序一致：
// 0 → cache_hash_size_cumsum 各表缓存大小前缀和，长 T+1，最后一元素为哨兵值
// 1 → update_table_indices 每条更新记录对应的表索引，长 N
// 2 → update_row_indices 每条更新记录在表内的行索引，长 N
namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("cache_hash_size_cumsumShape", context->GetInputShape(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("update_table_indicesShape", context->GetInputShape(1), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("update_row_indicesTensor", context->GetInputTensor(2), return ge::GRAPH_FAILED);

    int64_t totalLength = context->GetInputShape(2)->GetOriginShape().GetShapeSize();
    int64_t cumsumLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    uint32_t dimNumRow = context->GetInputShape(2)->GetOriginShape().GetDimNum();
    OPS_LOG_E_IF(dimNumRow != 1, context, return ge::GRAPH_FAILED,
                 "[ERROR]LinearizeCacheIndicesFromRowIdx: update_row_indices must be 1-D");

    ge::DataType inputDataType = context->GetInputTensor(2)->GetDataType();

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t realCoreNum = static_cast<uint32_t>(ascendPlatform.GetCoreNumAiv());
    uint32_t blockDim = (static_cast<uint32_t>(totalLength) + realCoreNum - 1) / realCoreNum;
    if (blockDim > MAX_THREAD_COUNT) {
        blockDim = MAX_THREAD_COUNT;
    }
    if (blockDim == 0) blockDim = 1;
    uint32_t gridDim = (static_cast<uint32_t>(totalLength) + blockDim - 1) / blockDim;
    if (gridDim == 0) gridDim = 1;

    OPS_CHECK(gridDim > MAX_BLOCK_COUNT,
              OPS_LOG_E("[ERROR]totalLength too large",
                        "totalLength exceeds MAX_THREAD_COUNT * MAX_BLOCK_COUNT."),
              return ge::GRAPH_FAILED);

    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("workspaceSize", workspaceSize, return ge::GRAPH_FAILED);
    workspaceSize[0] = ascendPlatform.GetLibApiWorkSpaceSize();

    LinearizeCacheIndicesFromRowIdxTilingData tiling;
    tiling.set_totalLength(totalLength);
    tiling.set_cumsumLength(cumsumLength);
    tiling.set_gridDim(gridDim);
    tiling.set_blockDim(blockDim);

    context->SetBlockDim(realCoreNum);

    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const gert::Shape* rowIdxShape = context->GetInputShape(2);
    OPS_LOG_E_IF_NULL("update_row_indices shape", rowIdxShape, return ge::GRAPH_FAILED);
    gert::Shape* outShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("output shape", outShape, return ge::GRAPH_FAILED);
    *outShape = *rowIdxShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    auto inputDataType = context->GetInputDataType(2);
    if (ge::GRAPH_SUCCESS != context->SetOutputDataType(0, inputDataType)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class LinearizeCacheIndicesFromRowIdx : public OpDef {
public:
    explicit LinearizeCacheIndicesFromRowIdx(const char* name) : OpDef(name)
    {
        this->Input("cache_hash_size_cumsum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("update_table_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("update_row_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("linear_cache_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(LinearizeCacheIndicesFromRowIdx);

} // namespace ops
