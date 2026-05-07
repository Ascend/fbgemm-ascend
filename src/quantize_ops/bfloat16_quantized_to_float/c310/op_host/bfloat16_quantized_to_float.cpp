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

#include "bfloat16_quantized_to_float_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("input", context->GetInputTensor(0), return ge::GRAPH_FAILED);

    auto inputShape = context->GetInputShape(0)->GetStorageShape();
    int64_t totalElems = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(inputShape.GetDimNum()); ++i) {
        totalElems *= inputShape.GetDim(i);
    }

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        OPS_LOG_E("[ERROR]", "ai core num is zero.");
        return ge::GRAPH_FAILED;
    }

    uint32_t blockDim = 0U;
    uint32_t neededCores = 0U;
    if (totalElems < 4096) {
        neededCores = 1;
        blockDim = 256;
    } else if (totalElems < 65536) {
        neededCores = std::min<uint32_t>(16, static_cast<uint32_t>(coreNum));
        blockDim = 256;
    } else if (totalElems < 262144) {
        neededCores = std::min<uint32_t>(32, static_cast<uint32_t>(coreNum));
        blockDim = 512;
    } else if (totalElems < 1048576) {
        neededCores = static_cast<uint32_t>(coreNum);
        blockDim = 512;
    } else {
        neededCores = static_cast<uint32_t>(coreNum);
        blockDim = 1024;
    }

    Bfloat16QuantizedToFloatTilingData tiling;
    tiling.set_totalElems(totalElems);
    tiling.set_blockDim(blockDim);

    context->SetBlockDim(neededCores);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    auto inputShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("inputShape", inputShape, return ge::GRAPH_FAILED);

    auto outShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("outShape", outShape, return ge::GRAPH_FAILED);
    outShape->SetDimNum(inputShape->GetDimNum());
    for (int64_t i = 0; i < static_cast<int64_t>(inputShape->GetDimNum()); ++i) {
        outShape->SetDim(i, inputShape->GetDim(i));
    }
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Bfloat16QuantizedToFloat : public OpDef {
public:
    explicit Bfloat16QuantizedToFloat(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque");
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(Bfloat16QuantizedToFloat);
}  // namespace ops
