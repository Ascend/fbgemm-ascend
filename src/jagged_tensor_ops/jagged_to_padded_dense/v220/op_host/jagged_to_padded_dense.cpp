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

#include "jagged_to_padded_dense_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"
namespace optiling {

constexpr int GM_ALIGN = 64;
constexpr int RESERVER_UB_SIZE = 20 * 1024;
constexpr int MIN_UB_USED_SIZE = 12 * 1024;
constexpr int DATA_TYPE_INT64 = 8;
constexpr int DATA_TYPE_INT32 = 4;
constexpr int DATA_TYPE_FLOAT32 = 4;
constexpr int NUM_QUEUE = 4;
constexpr int UB_ALIGN = 32;
constexpr int SUPPORT_EMBEDDING_DIM_NUM_1D = 1;
constexpr int SUPPORT_EMBEDDING_DIM_NUM_2D = 2;
constexpr size_t MAX_LENGTH_ATTR_IDX = 0;
constexpr size_t PADDING_VALUE_FP32_ATTR_IDX = 1;
constexpr size_t PADDING_VALUE_INT64_ATTR_IDX = 2;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    JaggedToPaddedDenseTilingData tiling;
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    OPS_LOG_E_IF_NULL("valuesShape", context->GetInputShape(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("valuesTensor", context->GetInputTensor(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetsShape", context->GetInputShape(1), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetsTensor", context->GetInputTensor(1), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("max_length", context->GetAttrs()->GetInt(MAX_LENGTH_ATTR_IDX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("padding_value_fp32", context->GetAttrs()->GetInt(PADDING_VALUE_FP32_ATTR_IDX),
        return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("padding_value_int64", context->GetAttrs()->GetInt(PADDING_VALUE_INT64_ATTR_IDX),
        return ge::GRAPH_FAILED);

    auto valuesShape = context->GetInputShape(0)->GetStorageShape();
    auto offsetsShape = context->GetInputShape(1)->GetStorageShape();

    uint64_t ubCanUsed;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
    ubCanUsed = ubCanUsed - RESERVER_UB_SIZE;
    if (ubCanUsed < MIN_UB_USED_SIZE) {
        OPS_LOG_E("jagged_to_padded_dense",
            "ubCanUsed is less than MIN_UB_USED_SIZE, ubCanUsed: %ld, MIN_UB_USED_SIZE: %ld",
            ubCanUsed, MIN_UB_USED_SIZE);
        return ge::GRAPH_FAILED;
    }
    ubCanUsed = ubCanUsed / UB_ALIGN / NUM_QUEUE * UB_ALIGN * NUM_QUEUE;
    tiling.set_ubCanUsed(ubCanUsed);

    // Support values rank-1 (jagged_1d_to_dense) or rank-2 (jagged_to_padded_dense), offsets must be rank-1
    int valuesRank = valuesShape.GetDimNum();
    if ((valuesRank != SUPPORT_EMBEDDING_DIM_NUM_2D && valuesRank != SUPPORT_EMBEDDING_DIM_NUM_1D) ||
        offsetsShape.GetDimNum() != 1) {
        OPS_LOG_E("jagged_to_padded_dense",
            "values must be rank-1 or rank-2, offsets must be rank-1,valuesRank(%d), offsetsRank(%d)",
            valuesRank, offsetsShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);

    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;
    // tiling core
    
    int64_t totalBatch = offsetsShape.GetDim(0) - 1;
    if (totalBatch <= 0) {
        OPS_LOG_E("jagged_to_padded_dense", "invalid offsetsShape: %ld", offsetsShape.GetDim(0));
        return ge::GRAPH_FAILED;
    }
    tiling.set_totalBatch(totalBatch);
    int64_t baseBatchLen = totalBatch / coreNum;
    tiling.set_baseBatchLen(baseBatchLen);
    int64_t tailSplitIndex = totalBatch % coreNum;
    tiling.set_tailSplitIndex(tailSplitIndex);
    int64_t valuesDim0 = valuesShape.GetDim(0);
    tiling.set_valuesDim0(valuesDim0);
    // For 1D values (jagged_1d_to_dense): treat as [total_L, 1]; for 2D: [total_L, D]
    int64_t valuesDim1 = (valuesRank == SUPPORT_EMBEDDING_DIM_NUM_1D) ? 1 : valuesShape.GetDim(1);
    tiling.set_valuesDim1(valuesDim1);
    int64_t offsetDim0 = offsetsShape.GetDim(0);
    tiling.set_offsetDim0(offsetDim0);
    int64_t outDim1 = *context->GetAttrs()->GetInt(MAX_LENGTH_ATTR_IDX);
    tiling.set_outDim1(outDim1);
    float padValFp32 = *context->GetAttrs()->GetFloat(PADDING_VALUE_FP32_ATTR_IDX);
    tiling.set_paddingValueFp32(padValFp32);
    int64_t padValInt64 = *context->GetAttrs()->GetInt(PADDING_VALUE_INT64_ATTR_IDX);
    tiling.set_paddingValueInt64(padValInt64);

    size_t blockDim = (totalBatch < coreNum) ? totalBatch : coreNum;
    context->SetBlockDim(blockDim);

    OPS_LOG_E_IF_NULL("context->GetRawTilingData(0)", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    int64_t maxLen = *context->GetAttrs()->GetInt(0);
    const gert::Shape* valuesShape = context->GetInputShape(0);
    const gert::Shape* offsetsShape = context->GetInputShape(1);

    gert::Shape* outShape = context->GetOutputShape(0);

    OPS_LOG_E_IF_NULL("valuesShape", valuesShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetsShape", offsetsShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("outShape", outShape, return ge::GRAPH_FAILED);

    int64_t batchSize = offsetsShape->GetDim(0) - 1;
 	int valuesRank = valuesShape->GetDimNum();
 	 
    // jagged_1d_to_dense: values 1D [total_L] -> out 2D [B, maxLen]
    // jagged_to_padded_dense: values 2D [total_L, D] -> out 3D [B, maxLen, D]
    if (valuesRank == optiling::SUPPORT_EMBEDDING_DIM_NUM_1D) {
        int dimSize = 2;
        outShape->SetDimNum(dimSize);
        outShape->SetDim(0, batchSize);
        outShape->SetDim(1, maxLen);
    } else {
        int dimSize = 3;
        int dimIndex2 = 2;
        outShape->SetDimNum(dimSize);
        outShape->SetDim(0, batchSize);
        outShape->SetDim(1, maxLen);
        outShape->SetDim(dimIndex2, valuesShape->GetDim(1));
 	}

    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class JaggedToPaddedDense : public OpDef {
public:
    explicit JaggedToPaddedDense(const char* name) : OpDef(name)
    {
        this->Input("values")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .Follow("values", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Attr("max_length").Int();
        this->Attr("padding_value_fp32").Float();
        this->Attr("padding_value_int64").Int();

        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend310p");
#ifdef SUPPORT_950
        this->AICore().AddConfig("ascend950");
#endif
    }
};

OP_ADD(JaggedToPaddedDense);
}  // namespace ops
