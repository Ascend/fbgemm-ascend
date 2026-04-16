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

#include "float_or_half_to_fused_nbit_rowwise_tiling.h"

#include <cstdint>
#include <algorithm>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "ops_log.h"

namespace {
constexpr int32_t INPUT_INDEX = 0;
constexpr int32_t BITRATE_INDEX = 0;
constexpr uint32_t MAX_THREADS_PER_BLOCK = 256;
constexpr uint32_t MIN_SIMT_DCACHE_SIZE = 32 * 1024;
constexpr uint32_t REDUCE_ALIGN = 64;
constexpr uint32_t UB_RESERVE = 512;
constexpr uint32_t SIMD_SWITCH_ROW_BYTES = 4 * 1024;
constexpr uint32_t SIMD_SWITCH_NROWS = 512;
constexpr uint32_t PER_ELEM_COST_SCALE = 8;
constexpr int32_t KERNEL_MODE_SIMT = 0;
constexpr int32_t KERNEL_MODE_SIMD = 1;
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    FloatOrHalfToFusedNbitRowwiseTilingData tiling;

    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("inputShape", context->GetInputShape(INPUT_INDEX), return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);
    auto bitRateValue = attrs->GetInt(BITRATE_INDEX);
    OPS_LOG_E_IF_NULL("bitRate", bitRateValue, return ge::GRAPH_FAILED);

    auto inputShape = context->GetInputShape(INPUT_INDEX)->GetStorageShape();
    OPS_CHECK(inputShape.GetDimNum() != 2,
              OPS_LOG_E(context, "[ERROR] FloatOrHalfToFusedNbitRowwise requires input dim num == 2"),
              return ge::GRAPH_FAILED);
    int64_t nrows = inputShape.GetDim(0);
    int64_t ncols = inputShape.GetDim(1);

    int32_t bitRate = static_cast<int32_t>(*bitRateValue);
    OPS_CHECK(bitRate != 1 && bitRate != 2 && bitRate != 4 && bitRate != 8,
              OPS_LOG_E(context, "[ERROR] bitRate must be one of {1,2,4,8}"), return ge::GRAPH_FAILED);

    int32_t numElemPerByte = 8 / bitRate;
    OPS_CHECK(ncols <= 0, OPS_LOG_E(context, "[ERROR] ncols must be greater than 0"), return ge::GRAPH_FAILED);
    OPS_CHECK(ncols % (2 * numElemPerByte) != 0,
              OPS_LOG_E(context, "[ERROR] ncols must be divisible by 2 * numElemPerByte"), return ge::GRAPH_FAILED);
    int64_t embBytes = (ncols + numElemPerByte - 1) / numElemPerByte;
    int64_t outputColumns = embBytes + 2 * static_cast<int64_t>(sizeof(uint16_t));

    auto inputTensor = context->GetInputTensor(INPUT_INDEX);
    OPS_LOG_E_IF_NULL("inputTensor", inputTensor, return ge::GRAPH_FAILED);
    int64_t inputTypeBytes = 0;
    if (inputTensor->GetDataType() == ge::DT_FLOAT) {
        inputTypeBytes = static_cast<int64_t>(sizeof(float));
    } else if (inputTensor->GetDataType() == ge::DT_FLOAT16) {
        inputTypeBytes = static_cast<int64_t>(sizeof(uint16_t));
    } else {
        OPS_LOG_E(context, "[ERROR] input dtype only supports DT_FLOAT or DT_FLOAT16");
        return ge::GRAPH_FAILED;
    }

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    OPS_CHECK(coreNum == 0, OPS_LOG_E(context, "[ERROR] available ai core num is 0"), return ge::GRAPH_FAILED);

    uint64_t ubTotal = 0;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubTotal);

    int64_t inBytesPerRow = ncols * inputTypeBytes;

    int32_t kernelMode = KERNEL_MODE_SIMT;
    // 根据经验值选择SIMD模式
    if (inBytesPerRow > SIMD_SWITCH_ROW_BYTES || nrows < SIMD_SWITCH_NROWS) {
        kernelMode = KERNEL_MODE_SIMD;
    }

    int64_t blockLen = 0;
    int32_t bufferNum = 1;
    int32_t tailSplitIndex = 0;
    uint32_t rowsPerCycle = 0;
    uint32_t localMemorySize = 0;
    int64_t splitBaseLen = 0;

    if (kernelMode == KERNEL_MODE_SIMD) {
        // 与 SIMD kernel 的 UB buffer 使用一致：每个 blockLen 元素占用
        //   inputTypeBytes  -> rawBuf(T)
        //   sizeof(float)   -> workBuf(float)
        //   2*sizeof(uint8_t) -> outBuf(uint8_t) + quantBuf(uint8_t)
        //   sizeof(uint16_t)  -> quantFp16Buf(half)
        uint32_t perElemCostScaled =
            PER_ELEM_COST_SCALE * (inputTypeBytes + sizeof(float) + 2 * sizeof(uint8_t) + sizeof(uint16_t));

        if (inputTensor->GetDataType() == ge::DT_FLOAT16) {
            // fp16 输入时 rawBuf 与 fp32Buf 分开，额外多一个 float 的 fp32Buf
            perElemCostScaled += PER_ELEM_COST_SCALE * sizeof(float);
        }

        // idxBaseBuf、idxWorkBuf(均为 int32_t) 与 packTmpBuf(uint8_t)
        perElemCostScaled += (2 * sizeof(int32_t) + sizeof(uint8_t)) * bitRate;

        uint64_t ubCanUsed = ubTotal - UB_RESERVE;
        blockLen = static_cast<int64_t>((ubCanUsed * PER_ELEM_COST_SCALE) / perElemCostScaled);
        blockLen -= blockLen % REDUCE_ALIGN;
        OPS_CHECK(blockLen < REDUCE_ALIGN, OPS_LOG_E(context, "[ERROR] blockLen is smaller than REDUCE_ALIGN"),
                  return ge::GRAPH_FAILED);

        int64_t actualCoreNum = static_cast<int64_t>(coreNum);
        if (nrows < actualCoreNum) {
            actualCoreNum = nrows;
        }
        if (actualCoreNum <= 0) {
            actualCoreNum = 1;
        }
        splitBaseLen = nrows / actualCoreNum;
        tailSplitIndex = static_cast<int32_t>(nrows % actualCoreNum);

        context->SetBlockDim(static_cast<uint32_t>(actualCoreNum));
    } else {
        int64_t outBytesPerRow = outputColumns;
        int64_t bytesPerRowAll = bufferNum * (inBytesPerRow + outBytesPerRow);

        uint64_t ubCanUsed = ubTotal - MIN_SIMT_DCACHE_SIZE;
        rowsPerCycle = ubCanUsed / bytesPerRowAll;
        rowsPerCycle = static_cast<uint32_t>(std::min(static_cast<int64_t>(rowsPerCycle), nrows));
        rowsPerCycle = std::min(rowsPerCycle, MAX_THREADS_PER_BLOCK);
        localMemorySize = static_cast<uint32_t>(rowsPerCycle * bytesPerRowAll);

        if (static_cast<uint64_t>(localMemorySize) * 2 < ubCanUsed) {
            bufferNum = 2;
            bytesPerRowAll = bufferNum * (inBytesPerRow + outBytesPerRow);
            rowsPerCycle = ubCanUsed / bytesPerRowAll;
            rowsPerCycle = static_cast<uint32_t>(std::min(static_cast<int64_t>(rowsPerCycle), nrows));
            rowsPerCycle = std::min(rowsPerCycle, MAX_THREADS_PER_BLOCK);
            localMemorySize = static_cast<uint32_t>(rowsPerCycle * bytesPerRowAll);
        }

        OPS_CHECK(rowsPerCycle == 0, OPS_LOG_E(context, "[ERROR] rowsPerCycle must be greater than 0"),
                  return ge::GRAPH_FAILED);
        int64_t blockNumByRows = (nrows + rowsPerCycle - 1) / rowsPerCycle;
        int64_t actualCoreNum = static_cast<int64_t>(coreNum);
        if (blockNumByRows < actualCoreNum) {
            actualCoreNum = blockNumByRows;
        }
        if (actualCoreNum <= 0) {
            actualCoreNum = 1;
        }

        context->SetBlockDim(static_cast<uint32_t>(actualCoreNum));
        context->SetLocalMemorySize(localMemorySize);
    }

    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("workspaceSize", workspaceSize, return ge::GRAPH_FAILED);
    workspaceSize[0] = ascendPlatform.GetLibApiWorkSpaceSize();

    tiling.set_nrows(nrows);
    tiling.set_ncols(ncols);
    tiling.set_bitRate(bitRate);
    tiling.set_kernelMode(kernelMode);
    tiling.set_outputColumns(outputColumns);
    tiling.set_numElemPerByte(numElemPerByte);
    tiling.set_rowsPerCycle(rowsPerCycle);
    tiling.set_bufferNum(bufferNum);
    tiling.set_blockLen(blockLen);
    tiling.set_splitBaseLen(splitBaseLen);
    tiling.set_tailSplitIndex(tailSplitIndex);

    auto tilingData = context->GetRawTilingData();
    OPS_LOG_E_IF_NULL("tilingData", tilingData, return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(tilingData->GetData(), tilingData->GetCapacity());
    tilingData->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    auto inputShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("inputShape", inputShape, return ge::GRAPH_FAILED);
    OPS_CHECK(inputShape->GetDimNum() != 2, OPS_LOG_E(context, "[ERROR] InferShape requires input dim num == 2"),
              return ge::GRAPH_FAILED);
    int64_t nrows = inputShape->GetDim(0);
    int64_t ncols = inputShape->GetDim(1);

    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);
    auto bitRateValue = attrs->GetInt(BITRATE_INDEX);
    OPS_LOG_E_IF_NULL("bitRate", bitRateValue, return ge::GRAPH_FAILED);
    int32_t bitRate = static_cast<int32_t>(*bitRateValue);
    OPS_CHECK(bitRate != 1 && bitRate != 2 && bitRate != 4 && bitRate != 8,
              OPS_LOG_E(context, "[ERROR] bitRate must be one of {1,2,4,8}"), return ge::GRAPH_FAILED);
    int32_t numElemPerByte = 8 / bitRate;
    int64_t outputColumns = (ncols + numElemPerByte - 1) / numElemPerByte + 2 * static_cast<int64_t>(sizeof(uint16_t));

    gert::Shape* outShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("outShape", outShape, return ge::GRAPH_FAILED);
    outShape->SetDimNum(2);
    outShape->SetDim(0, nrows);
    outShape->SetDim(1, outputColumns);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, ge::DT_UINT8);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class FloatOrHalfToFusedNbitRowwise : public OpDef {
public:
    explicit FloatOrHalfToFusedNbitRowwise(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("bitRate").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(FloatOrHalfToFusedNbitRowwise);
}  // namespace ops
