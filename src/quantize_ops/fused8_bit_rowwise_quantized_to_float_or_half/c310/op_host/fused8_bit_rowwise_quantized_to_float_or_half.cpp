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

#include "fused8_bit_rowwise_quantized_to_float_or_half_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("input_data", context->GetInputTensor(0), return ge::GRAPH_FAILED);

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        OPS_LOG_E("[ERROR]", "ai core num is zero.");
        return ge::GRAPH_FAILED;
    }

    const auto inputShape = context->GetInputShape(0)->GetStorageShape();
    uint32_t dimNum = inputShape.GetDimNum();
    int64_t cols = inputShape.GetDim(dimNum - 1);
    int64_t rows = 1;
    for (uint32_t i = 0; i < dimNum - 1; ++i) {
        rows *= inputShape.GetDim(i);
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);
    const int64_t* attr0 = attrs->GetInt(0);
    auto dtype = static_cast<int64_t>(*attr0);
    const bool* attr1 = attrs->GetBool(1);
    auto scaleBiasLast = static_cast<bool>(*attr1);
    const bool* attr2 = attrs->GetBool(2);
    auto quantPaddingFloatType = static_cast<bool>(*attr2);
    int64_t quantPaddingSize = quantPaddingFloatType ? 4 : 2;

    int64_t ncolsAligned = (cols + quantPaddingSize - 1) / quantPaddingSize * quantPaddingSize;
    int64_t outputCols = ncolsAligned - 2 * quantPaddingSize;

    Fused8BitRowwiseQuantizedToFloatOrHalfTilingData tiling;
    tiling.set_rows(rows);
    tiling.set_cols(cols);
    tiling.set_outputCols(outputCols);
    tiling.set_dtype(dtype);
    tiling.set_scaleBiasLast(scaleBiasLast);
    tiling.set_quantPaddingFloatType(quantPaddingFloatType);
    tiling.set_quantPaddingSize(quantPaddingSize);
    tiling.set_coreNum(coreNum);

    int32_t vecCols = static_cast<int32_t>(outputCols / 4);  // VEC_SIZE = 4
    int32_t threadsPerRow = 1;
    if (outputCols > 16) {
        threadsPerRow = 32;
    } else {
        while (threadsPerRow < outputCols) {
            threadsPerRow <<= 1;
        }
    }
    if (vecCols > 0 && vecCols < threadsPerRow) {
        threadsPerRow = vecCols;
    }
    if (threadsPerRow < 1)
        threadsPerRow = 1;

    int32_t rowsPerBlock =
        static_cast<int32_t>((rows + static_cast<int64_t>(coreNum) - 1) / static_cast<int64_t>(coreNum));
    if (rowsPerBlock > 32)
        rowsPerBlock = 32;
    if (rowsPerBlock < 1)
        rowsPerBlock = 1;
    while (threadsPerRow * rowsPerBlock > 1024 && rowsPerBlock > 1) {
        rowsPerBlock /= 2;
    }

    int32_t totalThreads = threadsPerRow * rowsPerBlock;
    if (totalThreads < 128) {
        int32_t targetRowsPerBlock = (128 + threadsPerRow - 1) / threadsPerRow;
        if (targetRowsPerBlock > rowsPerBlock) {
            rowsPerBlock = targetRowsPerBlock;
            if (rowsPerBlock > 32)
                rowsPerBlock = 32;
            totalThreads = threadsPerRow * rowsPerBlock;
        }
    }

    int32_t usedBlocks = static_cast<int32_t>((rows + rowsPerBlock - 1) / rowsPerBlock);
    if (usedBlocks > static_cast<int32_t>(coreNum))
        usedBlocks = static_cast<int32_t>(coreNum);
    if (usedBlocks < 1)
        usedBlocks = 1;

    int32_t threadsPerRowLog2 = -1;
    if ((threadsPerRow & (threadsPerRow - 1)) == 0) {
        threadsPerRowLog2 = 0;
        int32_t temp = threadsPerRow;
        while (temp > 1) {
            temp >>= 1;
            ++threadsPerRowLog2;
        }
    }

    tiling.set_threadsPerRow(threadsPerRow);
    tiling.set_threadsPerRowLog2(threadsPerRowLog2);
    tiling.set_rowsPerBlock(rowsPerBlock);
    tiling.set_totalThreads(totalThreads);
    context->SetBlockDim(usedBlocks);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const gert::Shape* inputShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("inputShape", inputShape, return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);
    const bool* attr1 = attrs->GetBool(1);
    auto scaleBiasLast = static_cast<bool>(*attr1);
    const bool* attr2 = attrs->GetBool(2);
    auto quantPaddingFloatType = static_cast<bool>(*attr2);

    int64_t quantPaddingSize = quantPaddingFloatType ? 4 : 2;
    int64_t cols = inputShape->GetDim(inputShape->GetDimNum() - 1);
    int64_t ncolsAligned = (cols + quantPaddingSize - 1) / quantPaddingSize * quantPaddingSize;
    int64_t outputCols = ncolsAligned - 2 * quantPaddingSize;

    const auto yShape = context->GetOutputShape(0);
    uint32_t dimNum = inputShape->GetDimNum();
    yShape->SetDimNum(dimNum);
    for (uint32_t i = 0; i < dimNum - 1; ++i) {
        yShape->SetDim(i, inputShape->GetDim(i));
    }
    yShape->SetDim(dimNum - 1, outputCols);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Fused8BitRowwiseQuantizedToFloatOrHalf : public OpDef {
public:
    explicit Fused8BitRowwiseQuantizedToFloatOrHalf(const char* name) : OpDef(name)
    {
        this->Input("inputData")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("outputDtype").AttrType(OPTIONAL).Int(0);
        this->Attr("scaleBiasLast").AttrType(OPTIONAL).Bool(true);
        this->Attr("quantPaddingFloatType").AttrType(OPTIONAL).Bool(true);
        this->SetInferShape(ge::InferShape);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque");

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(Fused8BitRowwiseQuantizedToFloatOrHalf);
}  // namespace ops
