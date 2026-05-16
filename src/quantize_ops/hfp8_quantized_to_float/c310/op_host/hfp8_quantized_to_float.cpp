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

#include "hfp8_quantized_to_float_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace optiling
{

// Thresholds for selecting blockDim and core count based on total element count.
// The strategy is to use fewer cores for small tensors to reduce launch overhead,
// and scale up blockDim/core usage as tensor size grows.
constexpr int64_t TOTAL_ELEMS_THRESHOLD_SMALL = 4096;      // < 4K elements: single core, 256 threads
constexpr int64_t TOTAL_ELEMS_THRESHOLD_MEDIUM = 65536;    // < 64K elements: up to 16 cores, 256 threads
constexpr int64_t TOTAL_ELEMS_THRESHOLD_LARGE = 262144;    // < 256K elements: up to 32 cores, 512 threads
constexpr int64_t TOTAL_ELEMS_THRESHOLD_XLARGE = 1048576;  // < 1M elements: all cores, 512 threads
// >= 1M elements: all cores, 1024 threads

constexpr uint32_t BLOCK_DIM_SMALL = 256;
constexpr uint32_t BLOCK_DIM_MEDIUM = 512;
constexpr uint32_t BLOCK_DIM_LARGE = 1024;

constexpr uint32_t MAX_CORES_MEDIUM = 16;
constexpr uint32_t MAX_CORES_LARGE = 32;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("input", context->GetInputTensor(0), return ge::GRAPH_FAILED);

    auto inputShape = context->GetInputShape(0)->GetStorageShape();
    int64_t totalElems = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(inputShape.GetDimNum()); ++i)
    {
        totalElems *= inputShape.GetDim(i);
    }

    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);
    const int64_t* attr0 = attrs->GetInt(0);
    const int64_t* attr1 = attrs->GetInt(1);

    OPS_LOG_E_IF_NULL("ebits", attr0, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("exponent_bias", attr1, return ge::GRAPH_FAILED);

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    if (coreNum == 0)
    {
        OPS_LOG_E("[ERROR]", "ai core num is zero.");
        return ge::GRAPH_FAILED;
    }

    uint32_t blockDim = 0U;
    uint32_t neededCores = 0U;

    if (totalElems < TOTAL_ELEMS_THRESHOLD_SMALL)
    {
        // Small tensor: minimize launch overhead with single core and small blockDim
        blockDim = BLOCK_DIM_SMALL;
        neededCores = 1;
    }
    else if (totalElems < TOTAL_ELEMS_THRESHOLD_MEDIUM)
    {
        // Medium-small tensor: use limited cores to avoid underutilization
        blockDim = BLOCK_DIM_SMALL;
        neededCores = std::min<uint32_t>(MAX_CORES_MEDIUM, static_cast<uint32_t>(coreNum));
    }
    else if (totalElems < TOTAL_ELEMS_THRESHOLD_LARGE)
    {
        // Medium tensor: increase blockDim and core count
        blockDim = BLOCK_DIM_MEDIUM;
        neededCores = std::min<uint32_t>(MAX_CORES_LARGE, static_cast<uint32_t>(coreNum));
    }
    else if (totalElems < TOTAL_ELEMS_THRESHOLD_XLARGE)
    {
        // Large tensor: use all available cores with medium blockDim
        blockDim = BLOCK_DIM_MEDIUM;
        neededCores = static_cast<uint32_t>(coreNum);
    }
    else
    {
        // Extra-large tensor: maximize parallelism with largest blockDim
        blockDim = BLOCK_DIM_LARGE;
        neededCores = static_cast<uint32_t>(coreNum);
    }

    Hfp8QuantizedToFloatTilingData tiling;
    tiling.set_totalElems(totalElems);
    tiling.set_ebits(*attr0);
    tiling.set_exponent_bias(*attr1);
    tiling.set_blockDim(blockDim);

    context->SetBlockDim(neededCores);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge
{
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    auto inputShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("inputShape", inputShape, return ge::GRAPH_FAILED);

    auto outShape = context->GetOutputShape(0);
    outShape->SetDimNum(inputShape->GetDimNum());
    for (int64_t i = 0; i < static_cast<int64_t>(inputShape->GetDimNum()); ++i)
    {
        outShape->SetDim(i, inputShape->GetDim(i));
    }
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops
{
class Hfp8QuantizedToFloat : public OpDef
{
   public:
    explicit Hfp8QuantizedToFloat(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("ebits").AttrType(REQUIRED).Int();
        this->Attr("exponent_bias").AttrType(REQUIRED).Int();
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

OP_ADD(Hfp8QuantizedToFloat);
}  // namespace ops
