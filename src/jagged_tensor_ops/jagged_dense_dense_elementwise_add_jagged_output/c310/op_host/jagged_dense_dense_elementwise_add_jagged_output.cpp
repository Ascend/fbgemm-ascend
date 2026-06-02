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
#include "jagged_dense_dense_elementwise_add_jagged_output_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "ops_log.h"

namespace optiling {
constexpr int32_t ALIGN_512 = 512;
constexpr int32_t RESERVED_UB_SIZE = (24 * 1024);
constexpr int32_t DIM0 = 0;
constexpr int32_t DIM1 = 1;

constexpr int32_t INPUT_X_INDEX = 0;
constexpr int32_t INPUT_DENSE0_INDEX = 1;
constexpr int32_t INPUT_DENSE1_INDEX = 2;
constexpr int32_t INPUT_OFFSETS_INDEX = 3;
constexpr int32_t OUTPUT_OUT_INDEX = 0;
constexpr int32_t ATTR_JAGGED_DIM0 = 0;
constexpr int32_t BUFFER_COUNT = 4;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("valuesTensor", context->GetInputTensor(INPUT_X_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("valuesTensor", context->GetInputTensor(INPUT_DENSE0_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("valuesTensor", context->GetInputTensor(INPUT_DENSE1_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("xShape", context->GetInputShape(INPUT_X_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("dense0Shape", context->GetInputShape(INPUT_DENSE0_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("dense1Shape", context->GetInputShape(INPUT_DENSE1_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offset0Shape", context->GetDynamicInputShape(INPUT_OFFSETS_INDEX, 0), return ge::GRAPH_FAILED);

    auto xShape = context->GetInputShape(INPUT_X_INDEX)->GetStorageShape();
    auto dense0Shape = context->GetInputShape(INPUT_DENSE0_INDEX)->GetStorageShape();
    auto dense1Shape = context->GetInputShape(INPUT_DENSE1_INDEX)->GetStorageShape();
    auto denseType = context->GetInputTensor(INPUT_DENSE0_INDEX)->GetDataType();

    OPS_CHECK(xShape.GetDimNum() != 2, "x_values must be at least 2D", return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);
    const int32_t* outDim0 = context->GetAttrs()->GetAttrPointer<int32_t>(ATTR_JAGGED_DIM0);
    OPS_LOG_E_IF_NULL("outDim0", outDim0, return ge::GRAPH_FAILED);

    int32_t offsetCnt = dense0Shape.GetDimNum() - 2;
    // dense 统一按 [B, max_L0, ..., max_Ln, D] 解释，因此 offsets 个数等于 dense 维度数减去 B 与 D。
    OPS_CHECK(offsetCnt <= 0 || offsetCnt > MAX_OFFSETS_CNT, OPS_LOG_E("[ERROR]", "offsetCnt invalid"),
              return ge::GRAPH_FAILED);
    OPS_CHECK(dense0Shape.GetDimNum() != offsetCnt + 2 || dense1Shape.GetDimNum() != offsetCnt + 2,
              OPS_LOG_E("[ERROR]", "dense dim count mismatch"), return ge::GRAPH_FAILED);
    for (int32_t i = 0; i < dense0Shape.GetDimNum(); ++i) {
        OPS_CHECK(dense0Shape.GetDim(i) != dense1Shape.GetDim(i),
                  OPS_LOG_E("[ERROR]", "dense_0 and dense_1 shape mismatch"), return ge::GRAPH_FAILED);
    }
    OPS_CHECK(dense0Shape.GetDim(DIM0) !=
                  context->GetDynamicInputShape(INPUT_OFFSETS_INDEX, 0)->GetStorageShape().GetDim(DIM0) - 1,
              OPS_LOG_E("[ERROR]", "dense shape[0] != offsets[0] shape[0] - 1"), return ge::GRAPH_FAILED);

    int64_t maxLengths[MAX_OFFSETS_CNT] = {0};
    int64_t offsetsLens[MAX_OFFSETS_CNT] = {0};
    int64_t denseDim1 = 1;
    for (int32_t i = 0; i < offsetCnt; ++i) {
        OPS_LOG_E_IF_NULL("offsetShape", context->GetDynamicInputShape(INPUT_OFFSETS_INDEX, i),
                          return ge::GRAPH_FAILED);
        maxLengths[i] = dense0Shape.GetDim(i + 1);
        offsetsLens[i] = context->GetDynamicInputShape(INPUT_OFFSETS_INDEX, i)->GetStorageShape().GetDim(DIM0);
        OPS_CHECK(offsetsLens[i] > std::numeric_limits<int>::max(),
                  OPS_LOG_E("[ERROR]", "offsets[%d] length invalid, got %d, max allowed %d", i, offsetsLens[i],
                            std::numeric_limits<int>::max()),
                  return ge::GRAPH_FAILED);
        denseDim1 *= maxLengths[i];
    }

    size_t usrSize = 0;
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = usrSize + systemWorkspacesSize;
    size_t coreNum = ascendPlatform.GetCoreNumAiv();

    uint64_t ubSize = 0;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    int32_t denseDim2 = dense0Shape.GetDim(offsetCnt + 1);
    int64_t jaggedTotal = static_cast<int64_t>(*outDim0) * denseDim2;
    int64_t denseTotal = dense0Shape.GetDim(DIM0) * denseDim1 * denseDim2;

    OPS_CHECK(xShape.GetDim(DIM0) != *outDim0 || xShape.GetDim(DIM1) != denseDim2,
              OPS_LOG_E("[ERROR]", "x_values shape mismatch"), return ge::GRAPH_FAILED);
    OPS_CHECK(coreNum == 0, OPS_LOG_E("[ERROR]", "aiv core num == 0"), return ge::GRAPH_FAILED);
    // 预留部分 UB 给系统和队列管理，剩余空间按 512B 对齐后作为单次搬运/计算块大小。
    int32_t singleLoopSize = static_cast<int32_t>((ubSize - RESERVED_UB_SIZE) / BUFFER_COUNT / ALIGN_512 * ALIGN_512);

    JaggedDenseDenseElementwiseAddJaggedOutputTiling tilingData;
    tilingData.set_denseDim1(denseDim1);
    tilingData.set_denseDim2(denseDim2);
    tilingData.set_singleLoopSize(singleLoopSize);
    tilingData.set_denseType(denseType);
    tilingData.set_denseTotal(denseTotal);
    tilingData.set_jaggedTotal(jaggedTotal);
    tilingData.set_offsetCnt(offsetCnt);
    // maxLengths 和 offsetsLens 让 kernel 在不访问 shape 元信息的情况下完成 jagged row 到 dense row 的映射。
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

// 输出 dtype 跟随 x_values，dense_0/dense_1 在 OpDef 中也声明为跟随 x_values。
static ge::graphStatus InferDtype(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    context->SetOutputDataType(OUTPUT_OUT_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class JaggedDenseDenseElementwiseAddJaggedOutput : public OpDef {
public:
    explicit JaggedDenseDenseElementwiseAddJaggedOutput(const char* name) : OpDef(name)
    {
        this->Input("x_values")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("dense_0").ParamType(REQUIRED).Follow("x_values", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Input("dense_1").ParamType(REQUIRED).Follow("x_values", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Input("offsets")
            .ParamType(DYNAMIC)
            .DataTypeList({ge::DT_INT64, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Output("out").ParamType(REQUIRED).Follow("x_values", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Attr("jagged_dim0").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDtype);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(JaggedDenseDenseElementwiseAddJaggedOutput);
}  // namespace ops
