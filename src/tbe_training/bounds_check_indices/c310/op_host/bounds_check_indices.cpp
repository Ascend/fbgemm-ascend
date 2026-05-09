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

#include <algorithm>
#include "bounds_check_indices_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace {
constexpr int32_t kWarpSize = 32;
constexpr int32_t kPrefetchMaxBlocks = 8;
constexpr int32_t kV1ThreadsPerBlock = 256;
constexpr int32_t kV2ThreadsPerBlock = 1024;

enum BoundsCheckIndicesV1InputIndex : int32_t {
    V1_INPUT_ROWS_PER_TABLE_INDEX = 0,
    V1_INPUT_INDICES_INDEX = 1,
    V1_INPUT_OFFSETS_INDEX = 2,
    V1_INPUT_WARNING_INDEX = 3,
    V1_INPUT_B_OFFSETS_INDEX = 4,
};

enum BoundsCheckIndicesV1AttrIndex : int32_t {
    V1_ATTR_BOUNDS_CHECK_MODE_INDEX = 0,
    V1_ATTR_MAX_B_INDEX = 1,
    V1_ATTR_T_INDEX = 2,
    V1_ATTR_B_INDEX = 3,
    V1_ATTR_TOTAL_B_INDEX = 4,
    V1_ATTR_VBE_INDEX = 5,
};

enum BoundsCheckIndicesV2InputIndex : int32_t {
    V2_INPUT_ROWS_PER_TABLE_INDEX = 0,
    V2_INPUT_INDICES_INDEX = 1,
    V2_INPUT_OFFSETS_INDEX = 2,
    V2_INPUT_WARNING_INDEX = 3,
    V2_INPUT_B_OFFSETS_INDEX = 4,
    V2_INPUT_B_T_MAP_INDEX = 5,
};

enum BoundsCheckIndicesV2AttrIndex : int32_t {
    V2_ATTR_BOUNDS_CHECK_MODE_INDEX = 0,
    V2_ATTR_INFO_B_NUM_BITS_INDEX = 1,
    V2_ATTR_INFO_B_MASK_INDEX = 2,
    V2_ATTR_T_INDEX = 3,
    V2_ATTR_B_INDEX = 4,
    V2_ATTR_TOTAL_B_INDEX = 5,
    V2_ATTR_VBE_INDEX = 6,
    V2_ATTR_PREFETCH_PIPELINE_INDEX = 7,
};

enum BoundsCheckIndicesOutputIndex : int32_t {
    OUTPUT_INDICES_INDEX = 0,
    OUTPUT_OFFSETS_INDEX = 1,
    OUTPUT_WARNING_INDEX = 2,
};

template <typename UnsignedT>
inline void HostPrecomputeFastDivmod(UnsignedT divisor, UnsignedT& outMagic, uint32_t& outShift)
{
    if (divisor <= 1) {
        outMagic = 0;
        outShift = 0;
        return;
    }
    unsigned __int128 one = 1;
    constexpr uint32_t BIT_WIDTH = static_cast<uint32_t>(sizeof(UnsignedT) * 8); /* 单字节 8 BIT */
    uint32_t s = 0;
    for (; s < BIT_WIDTH; ++s) {
        if ((one << s) >= static_cast<unsigned __int128>(divisor)) {
            break;
        }
    }
    outShift = s;
    outMagic = static_cast<UnsignedT>(
        ((one << BIT_WIDTH) * ((one << s) - static_cast<unsigned __int128>(divisor))) /
        static_cast<unsigned __int128>(divisor) + 1);
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingFuncV1(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    OPS_LOG_E_IF_NULL("indicesShape", context->GetInputShape(V1_INPUT_INDICES_INDEX), return ge::GRAPH_FAILED);
    const auto* indicesShape = context->GetInputShape(V1_INPUT_INDICES_INDEX);
    const int64_t numIndices = indicesShape->GetStorageShape().GetDim(0);

    auto* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    const int32_t* TPtr = attrs->GetAttrPointer<int32_t>(V1_ATTR_T_INDEX);
    OPS_LOG_E_IF_NULL("T attr", TPtr, return ge::GRAPH_FAILED);
    const int32_t numTables = *TPtr;

    const int32_t* BPtr = attrs->GetAttrPointer<int32_t>(V1_ATTR_B_INDEX);
    OPS_LOG_E_IF_NULL("B attr", BPtr, return ge::GRAPH_FAILED);
    const int32_t batchSize = *BPtr;

    const int32_t* totalBPtr = attrs->GetAttrPointer<int32_t>(V1_ATTR_TOTAL_B_INDEX);
    OPS_LOG_E_IF_NULL("total_B attr", totalBPtr, return ge::GRAPH_FAILED);
    const int32_t totalB = *totalBPtr;

    const int32_t* maxBPtr = attrs->GetAttrPointer<int32_t>(V1_ATTR_MAX_B_INDEX);
    const int32_t maxB = (maxBPtr != nullptr) ? *maxBPtr : 0;

    const int32_t* boundsCheckModePtr = attrs->GetAttrPointer<int32_t>(V1_ATTR_BOUNDS_CHECK_MODE_INDEX);
    OPS_LOG_E_IF_NULL("bounds_check_mode attr", boundsCheckModePtr, return ge::GRAPH_FAILED);

    const bool* vbePtr = attrs->GetAttrPointer<bool>(V1_ATTR_VBE_INDEX);
    OPS_LOG_E_IF_NULL("vbe attr", vbePtr, return ge::GRAPH_FAILED);
    bool vbeAttr = *vbePtr;

    int32_t maxB_ = vbeAttr ? maxB : batchSize;
    int32_t gridSize = (maxB_ * numTables + (kV1ThreadsPerBlock / kWarpSize) - 1) / (kV1ThreadsPerBlock / kWarpSize);
    context->SetBlockDim(gridSize);

    uint32_t batchSizeDivMagic = 0;
    uint32_t batchSizeDivShift = 0;
    HostPrecomputeFastDivmod<uint32_t>(static_cast<uint32_t>(maxB_), batchSizeDivMagic, batchSizeDivShift);

    BoundsCheckIndicesTilingData tiling;
    tiling.set_numIndices(numIndices);
    tiling.set_numTables(numTables);
    tiling.set_batchSize(maxB_);
    tiling.set_totalB(totalB);
    tiling.set_vbe(vbeAttr ? 1 : 0);
    tiling.set_boundsCheckMode(*boundsCheckModePtr);
    tiling.set_batchSizeDivMagic(batchSizeDivMagic);
    tiling.set_batchSizeDivShift(batchSizeDivShift);

    auto tilingData = context->GetRawTilingData();
    OPS_LOG_E_IF_NULL("tilingData", tilingData, return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(tilingData->GetData(), tilingData->GetCapacity());
    tilingData->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFuncV2(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    OPS_LOG_E_IF_NULL("indicesShape", context->GetInputShape(V2_INPUT_INDICES_INDEX), return ge::GRAPH_FAILED);
    const auto* indicesShape = context->GetInputShape(V2_INPUT_INDICES_INDEX);
    const int64_t numIndices = indicesShape->GetStorageShape().GetDim(0);

    auto* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    const int32_t* TPtr = attrs->GetAttrPointer<int32_t>(V2_ATTR_T_INDEX);
    OPS_LOG_E_IF_NULL("T attr", TPtr, return ge::GRAPH_FAILED);
    const int32_t numTables = *TPtr;

    const int32_t* BPtr = attrs->GetAttrPointer<int32_t>(V2_ATTR_B_INDEX);
    OPS_LOG_E_IF_NULL("B attr", BPtr, return ge::GRAPH_FAILED);
    const int32_t batchSize = *BPtr;

    const int32_t* totalBPtr = attrs->GetAttrPointer<int32_t>(V2_ATTR_TOTAL_B_INDEX);
    OPS_LOG_E_IF_NULL("total_B attr", totalBPtr, return ge::GRAPH_FAILED);
    const int32_t totalB = *totalBPtr;

    const int32_t* boundsCheckModePtr = attrs->GetAttrPointer<int32_t>(V2_ATTR_BOUNDS_CHECK_MODE_INDEX);
    OPS_LOG_E_IF_NULL("bounds_check_mode attr", boundsCheckModePtr, return ge::GRAPH_FAILED);

    const bool* vbePtr = attrs->GetAttrPointer<bool>(V2_ATTR_VBE_INDEX);
    OPS_LOG_E_IF_NULL("vbe attr", vbePtr, return ge::GRAPH_FAILED);
    bool vbeAttr = *vbePtr;

    const int64_t* infoBNumBitsPtr = attrs->GetAttrPointer<int64_t>(V2_ATTR_INFO_B_NUM_BITS_INDEX);
    const int32_t infoBNumBits = (infoBNumBitsPtr != nullptr) ? static_cast<int32_t>(*infoBNumBitsPtr) : 0;

    const int64_t* infoBMaskPtr = attrs->GetAttrPointer<int64_t>(V2_ATTR_INFO_B_MASK_INDEX);
    const uint32_t infoBMask = (infoBMaskPtr != nullptr) ? static_cast<uint32_t>(*infoBMaskPtr) : 0;

    uint32_t batchSizeDivMagic = 0;
    uint32_t batchSizeDivShift = 0;
    HostPrecomputeFastDivmod<uint32_t>(static_cast<uint32_t>(batchSize), batchSizeDivMagic, batchSizeDivShift);

    int32_t gridSize = (totalB + (kV2ThreadsPerBlock / kWarpSize) - 1) / (kV2ThreadsPerBlock / kWarpSize);
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        OPS_LOG_E("", "Core num is 0.");
        return ge::GRAPH_FAILED;
    }
    int32_t actualGridSize = std::min(static_cast<int32_t>(coreNum), gridSize);

    context->SetBlockDim(actualGridSize);

    BoundsCheckIndicesTilingData tiling;
    tiling.set_numIndices(numIndices);
    tiling.set_numTables(numTables);
    tiling.set_batchSize(batchSize);
    tiling.set_totalB(totalB);
    tiling.set_vbe(vbeAttr ? 1 : 0);
    tiling.set_boundsCheckMode(*boundsCheckModePtr);
    tiling.set_infoBNumBits(infoBNumBits);
    tiling.set_infoBMask(infoBMask);
    tiling.set_batchSizeDivMagic(batchSizeDivMagic);
    tiling.set_batchSizeDivShift(batchSizeDivShift);

    auto tilingData = context->GetRawTilingData();
    OPS_LOG_E_IF_NULL("tilingData", tilingData, return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(tilingData->GetData(), tilingData->GetCapacity());
    tilingData->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShapeV1(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    // Output shape matches input shape (in-place operation)
    const auto* indicesShape = context->GetInputShape(V1_INPUT_INDICES_INDEX);
    const auto* offsetsShape = context->GetInputShape(V1_INPUT_OFFSETS_INDEX);
    const auto* warningShape = context->GetInputShape(V1_INPUT_WARNING_INDEX);

    auto* indicesOutShape = context->GetOutputShape(OUTPUT_INDICES_INDEX);
    auto* offsetsOutShape = context->GetOutputShape(OUTPUT_OFFSETS_INDEX);
    auto* warningOutShape = context->GetOutputShape(OUTPUT_WARNING_INDEX);

    OPS_LOG_E_IF_NULL("indicesShape", indicesShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetsShape", offsetsShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("warningShape", warningShape, return ge::GRAPH_FAILED);

    *indicesOutShape = *indicesShape;
    *offsetsOutShape = *offsetsShape;
    *warningOutShape = *warningShape;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeV1(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    // Output dtype matches input dtype
    auto indicesDtype = context->GetInputDataType(V1_INPUT_INDICES_INDEX);
    auto offsetsDtype = context->GetInputDataType(V1_INPUT_OFFSETS_INDEX);
    auto warningDtype = context->GetInputDataType(V1_INPUT_WARNING_INDEX);

    context->SetOutputDataType(OUTPUT_INDICES_INDEX, indicesDtype);
    context->SetOutputDataType(OUTPUT_OFFSETS_INDEX, offsetsDtype);
    context->SetOutputDataType(OUTPUT_WARNING_INDEX, warningDtype);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeV2(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    // Output shape matches input shape (in-place operation)
    const auto* indicesShape = context->GetInputShape(V2_INPUT_INDICES_INDEX);
    const auto* offsetsShape = context->GetInputShape(V2_INPUT_OFFSETS_INDEX);
    const auto* warningShape = context->GetInputShape(V2_INPUT_WARNING_INDEX);

    auto* indicesOutShape = context->GetOutputShape(OUTPUT_INDICES_INDEX);
    auto* offsetsOutShape = context->GetOutputShape(OUTPUT_OFFSETS_INDEX);
    auto* warningOutShape = context->GetOutputShape(OUTPUT_WARNING_INDEX);

    OPS_LOG_E_IF_NULL("indicesShape", indicesShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetsShape", offsetsShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("warningShape", warningShape, return ge::GRAPH_FAILED);

    *indicesOutShape = *indicesShape;
    *offsetsOutShape = *offsetsShape;
    *warningOutShape = *warningShape;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeV2(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    // Output dtype matches input dtype
    auto indicesDtype = context->GetInputDataType(V2_INPUT_INDICES_INDEX);
    auto offsetsDtype = context->GetInputDataType(V2_INPUT_OFFSETS_INDEX);
    auto warningDtype = context->GetInputDataType(V2_INPUT_WARNING_INDEX);

    context->SetOutputDataType(OUTPUT_INDICES_INDEX, indicesDtype);
    context->SetOutputDataType(OUTPUT_OFFSETS_INDEX, offsetsDtype);
    context->SetOutputDataType(OUTPUT_WARNING_INDEX, warningDtype);

    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class BoundsCheckIndicesV1 : public OpDef {
public:
    explicit BoundsCheckIndicesV1(const char* name) : OpDef(name)
    {
        this->Input("rows_per_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("warning")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("B_offsets")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("indices_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("offsets_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("warning_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("bounds_check_mode").Int();
        this->Attr("max_B").AttrType(OPTIONAL).Int(0);
        this->Attr("T").Int();
        this->Attr("B").Int();
        this->Attr("total_B").Int();
        this->Attr("vbe").Bool();

        this->SetInferShape(ge::InferShapeV1);
        this->SetInferDataType(ge::InferDataTypeV1);
        this->AICore().SetTiling(optiling::TilingFuncV1);
        this->AICore().AddConfig("ascend950");
    }
};

class BoundsCheckIndicesV2 : public OpDef {
public:
    explicit BoundsCheckIndicesV2(const char* name) : OpDef(name)
    {
        this->Input("rows_per_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("warning")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("B_offsets")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("b_t_map")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("indices_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("offsets_out")
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("warning_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("bounds_check_mode").Int();
        this->Attr("info_B_num_bits").AttrType(OPTIONAL).Int(-1);
        this->Attr("info_B_mask").AttrType(OPTIONAL).Int(-1);
        this->Attr("T").Int();
        this->Attr("B").Int();
        this->Attr("total_B").Int();
        this->Attr("vbe").Bool();
        this->Attr("prefetch_pipeline").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShapeV2);
        this->SetInferDataType(ge::InferDataTypeV2);
        this->AICore().SetTiling(optiling::TilingFuncV2);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(BoundsCheckIndicesV1);
OP_ADD(BoundsCheckIndicesV2);

}  // namespace ops
