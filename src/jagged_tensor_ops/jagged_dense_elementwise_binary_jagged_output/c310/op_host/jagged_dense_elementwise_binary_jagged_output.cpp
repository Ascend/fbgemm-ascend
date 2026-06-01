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
#include <limits>
#include "jagged_dense_elementwise_binary_jagged_output_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "ops_log.h"

namespace optiling {
constexpr int32_t ALIGN_512 = 512;
constexpr int32_t RESERVER_UB_SIZE = (24 * 1024);
constexpr int32_t DIM0 = 0;
constexpr int32_t DIM1 = 1;
constexpr int32_t DIM2 = 2;

constexpr int32_t INPUT_X_INDEX = 0;
constexpr int32_t INPUT_DENSE_INDEX = 1;
constexpr int32_t INPUT_OFFSETS_INDEX = 2;
constexpr int32_t OUTPUT_OUT_INDEX = 0;
constexpr int32_t ATTR_JAGGED_DIM0 = 0;
constexpr int32_t ATTR_ELEMENTWISE_MODE = 1;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("xShape", context->GetInputShape(INPUT_X_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("denseShape", context->GetInputShape(INPUT_DENSE_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offset0Shape", context->GetDynamicInputShape(INPUT_OFFSETS_INDEX, 0), return ge::GRAPH_FAILED);

    auto xShape = context->GetInputShape(INPUT_X_INDEX)->GetStorageShape();
    auto denseShape = context->GetInputShape(INPUT_DENSE_INDEX)->GetStorageShape();
    auto denseType = context->GetInputTensor(INPUT_DENSE_INDEX)->GetDataType();

    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);
    const int32_t* outDim0 = context->GetAttrs()->GetAttrPointer<int32_t>(ATTR_JAGGED_DIM0);
    OPS_LOG_E_IF_NULL("outDim0", outDim0, return ge::GRAPH_FAILED);
    const int32_t* ewMode = context->GetAttrs()->GetAttrPointer<int32_t>(ATTR_ELEMENTWISE_MODE);
    OPS_LOG_E_IF_NULL("ewMode", ewMode, return ge::GRAPH_FAILED);

    int32_t offsetCnt = denseShape.GetDimNum() - 2;
    OPS_CHECK(offsetCnt <= 0 || offsetCnt > MAX_OFFSETS_CNT, OPS_LOG_E("[ERROR]", "offsetCnt invalid"),
              return ge::GRAPH_FAILED);
    OPS_CHECK(denseShape.GetDim(DIM0) !=
                  context->GetDynamicInputShape(INPUT_OFFSETS_INDEX, 0)->GetStorageShape().GetDim(DIM0) - 1,
              OPS_LOG_E("[ERROR]", "dense shape[0] != offsets[0] shape[0] - 1"), return ge::GRAPH_FAILED);

    int64_t maxLengths[MAX_OFFSETS_CNT] = {0};
    int64_t offsetsLens[MAX_OFFSETS_CNT] = {0};
    int64_t denseDim1 = 1;
    for (int32_t i = 0; i < offsetCnt; ++i) {
        OPS_LOG_E_IF_NULL("offsetShape", context->GetDynamicInputShape(INPUT_OFFSETS_INDEX, i),
                          return ge::GRAPH_FAILED);
        maxLengths[i] = denseShape.GetDim(i + 1);
        offsetsLens[i] = context->GetDynamicInputShape(INPUT_OFFSETS_INDEX, i)->GetStorageShape().GetDim(DIM0);
        OPS_CHECK(offsetsLens[i] > std::numeric_limits<int>::max(), OPS_LOG_E("[ERROR]", "offset shape invalid"),
                  return ge::GRAPH_FAILED);
        denseDim1 *= maxLengths[i];
    }

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    int32_t denseDim2 = denseShape.GetDim(offsetCnt + 1);
    int64_t jaggedTotal = static_cast<int64_t>(*outDim0) * denseDim2;
    int64_t denseTotal = denseShape.GetDim(DIM0) * denseDim1 * denseDim2;

    OPS_CHECK(xShape.GetDim(DIM0) != *outDim0 || xShape.GetDim(DIM1) != denseDim2,
              OPS_LOG_E("[ERROR]", "x_values shape mismatch"), return ge::GRAPH_FAILED);
    OPS_CHECK(coreNum == 0, OPS_LOG_E("[ERROR]", "aiv core num == 0"), return ge::GRAPH_FAILED);
    int32_t singleLoopSize = static_cast<int32_t>((ubSize - RESERVER_UB_SIZE) / 3 / ALIGN_512 * ALIGN_512);

    JaggedDenseElementwiseBinaryJaggedOutputTiling tilingData;
    tilingData.set_denseDim1(denseDim1);
    tilingData.set_denseDim2(denseDim2);
    tilingData.set_singleLoopSize(singleLoopSize);
    tilingData.set_denseType(denseType);
    tilingData.set_denseTotal(denseTotal);
    tilingData.set_jaggedTotal(jaggedTotal);
    tilingData.set_elementwiseMode(*ewMode);
    tilingData.set_offsetCnt(offsetCnt);
    tilingData.set_maxLengths(maxLengths);
    tilingData.set_offsetsLens(offsetsLens);

    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    context->SetBlockDim(coreNum);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
using optiling::DIM0;
using optiling::DIM1;
using optiling::INPUT_X_INDEX;
using optiling::OUTPUT_OUT_INDEX;

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const gert::Shape* xShape = context->GetInputShape(INPUT_X_INDEX);
    gert::Shape* outShape = context->GetOutputShape(OUTPUT_OUT_INDEX);
    OPS_LOG_E_IF_NULL("xShape", xShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("outShape", outShape, return ge::GRAPH_FAILED);
    outShape->SetDimNum(2);
    outShape->SetDim(DIM0, xShape->GetDim(DIM0));
    outShape->SetDim(DIM1, xShape->GetDim(DIM1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    context->SetOutputDataType(OUTPUT_OUT_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class JaggedDenseElementwiseBinaryJaggedOutput : public OpDef {
public:
    explicit JaggedDenseElementwiseBinaryJaggedOutput(const char* name) : OpDef(name)
    {
        this->Input("x_values")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("dense").ParamType(REQUIRED).Follow("x_values", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Input("offsets")
            .ParamType(DYNAMIC)
            .DataTypeList({ge::DT_INT64, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Output("out").ParamType(REQUIRED).Follow("x_values", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Attr("jagged_dim0").Int();
        this->Attr("elementwise_mode").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDtype);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(JaggedDenseElementwiseBinaryJaggedOutput);
}  // namespace ops
