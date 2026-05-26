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
#include <cstdio>
#include <algorithm>

#include "register/op_def_registry.h"
#include "emb_inplace_update_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace {
constexpr uint32_t MAX_THREAD_COUNT = 1024; // SIMT 硬件限制：单 block 最大线程数

// Input slot indices (must align with the OpDef::Input order below).
constexpr int32_t IN_DEV_WEIGHTS = 0;
constexpr int32_t IN_UVM_WEIGHTS = 1;
constexpr int32_t IN_WEIGHTS_PLACEMENTS = 2;
constexpr int32_t IN_WEIGHTS_OFFSETS = 3;
constexpr int32_t IN_WEIGHTS_TYS = 4;
constexpr int32_t IN_D_OFFSETS = 5;
constexpr int32_t IN_UPDATE_WEIGHTS = 6;
constexpr int32_t IN_UPDATE_TABLE_INDICES = 7;
constexpr int32_t IN_UPDATE_ROW_INDICES = 8;
constexpr int32_t IN_UPDATE_OFFSETS = 9;

constexpr int32_t ATTR_ROW_ALIGNMENT = 0;
} // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
  OPS_LOG_E_IF_NULL("update_row_indicesShape",
                    context->GetInputShape(IN_UPDATE_ROW_INDICES),
                    return ge::GRAPH_FAILED);
  OPS_LOG_E_IF_NULL("D_offsetsShape", context->GetInputShape(IN_D_OFFSETS),
                    return ge::GRAPH_FAILED);
  OPS_LOG_E_IF_NULL("update_row_indicesTensor",
                    context->GetInputTensor(IN_UPDATE_ROW_INDICES),
                    return ge::GRAPH_FAILED);
  OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);

  int64_t totalUpdates =
      context->GetInputShape(IN_UPDATE_ROW_INDICES)->GetOriginShape().GetShapeSize();
  int64_t dOffsetsLen =
      context->GetInputShape(IN_D_OFFSETS)->GetOriginShape().GetShapeSize();
  int64_t numTables = dOffsetsLen > 0 ? dOffsetsLen - 1 : 0;

  ge::DataType rowIdxDtype =
      context->GetInputTensor(IN_UPDATE_ROW_INDICES)->GetDataType();
  OPS_CHECK(rowIdxDtype != ge::DT_INT32 && rowIdxDtype != ge::DT_INT64,
            OPS_LOG_E("[ERROR]EmbInplaceUpdate",
                      "update_row_indices must be int32 or int64."),
            return ge::GRAPH_FAILED);
  int64_t rowIdxIsInt64 = (rowIdxDtype == ge::DT_INT64) ? 1 : 0;

  const int64_t *rowAlignmentAttr =
      context->GetAttrs()->GetInt(ATTR_ROW_ALIGNMENT);
  OPS_LOG_E_IF_NULL("row_alignment attr", rowAlignmentAttr,
                    return ge::GRAPH_FAILED);
  int64_t rowAlignment = *rowAlignmentAttr;
  OPS_CHECK(rowAlignment < 16 || rowAlignment % 16 != 0,
            OPS_LOG_E("[ERROR]EmbInplaceUpdate",
                      "row_alignment must be a positive multiple of 16 "
                      "(NPU requires 16-byte aligned rows for float4 vectorized copy)."),
            return ge::GRAPH_FAILED);

  auto ascendPlatform =
      platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t realCoreNum = static_cast<uint32_t>(ascendPlatform.GetCoreNumAiv());
  if (realCoreNum == 0) {
    realCoreNum = 1;
  }

  // SIMT launch (warp 协作模型)：每个 warp（32 lane）协作处理一条更新记录。
  //   gridDim          = AI Core 数
  //   threadNumPerBlock = warpsPerBlock × WARP_SIZE，必须是 WARP_SIZE 的倍数
  //   总并行 warp 数    = gridDim × warpsPerBlock
  // 单条记录由 32 个 lane 协作，warp 间通过 stride 循环覆盖 N 条记录。
  constexpr uint32_t WARP_SIZE = 32;
  uint32_t threadNumPerBlock = MAX_THREAD_COUNT; // 默认 1024 = 32 warps × 32 lanes
  uint32_t warpsPerBlock = threadNumPerBlock / WARP_SIZE;
  uint64_t totalWarps =
      static_cast<uint64_t>(realCoreNum) * warpsPerBlock;
  if (totalUpdates > 0 &&
      static_cast<uint64_t>(totalUpdates) < totalWarps) {
    // N 很小：缩减 warpsPerBlock，避免大量空闲 warp（threadNumPerBlock 仍按 warp 对齐）
    uint32_t neededWarps = static_cast<uint32_t>(
        std::max<int64_t>(totalUpdates, 1));
    if (neededWarps < warpsPerBlock) {
        warpsPerBlock = neededWarps;
        threadNumPerBlock = warpsPerBlock * WARP_SIZE;
    }
  }

  uint32_t gridDim = realCoreNum;
  if (totalUpdates == 0) {
    gridDim = 1;
    threadNumPerBlock = WARP_SIZE; // 至少保留 1 个 warp
  } else {
    uint32_t need = static_cast<uint32_t>(
        (totalUpdates + warpsPerBlock - 1) / warpsPerBlock);
    if (need < gridDim) {
      gridDim = need;
    }
    if (gridDim == 0) {
      gridDim = 1;
    }
  }

  size_t *workspaceSize = context->GetWorkspaceSizes(1);
  OPS_LOG_E_IF_NULL("workspaceSize", workspaceSize, return ge::GRAPH_FAILED);
  workspaceSize[0] = ascendPlatform.GetLibApiWorkSpaceSize();

  EmbInplaceUpdateTilingData tiling;
  tiling.set_totalUpdates(totalUpdates);
  tiling.set_numTables(numTables);
  tiling.set_rowAlignment(rowAlignment);
  tiling.set_rowIdxIsInt64(rowIdxIsInt64);
  tiling.set_threadNumPerBlock(static_cast<int32_t>(threadNumPerBlock));
  tiling.set_gridDim(static_cast<int32_t>(gridDim));

  context->SetBlockDim(gridDim);

  OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(),
                    return ge::GRAPH_FAILED);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext *context) {
  // emb_inplace_update is an in-place op with no graph outputs.
  (void)context;
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class EmbInplaceUpdate : public OpDef {
public:
  explicit EmbInplaceUpdate(const char *name) : OpDef(name) {
    this->Input("dev_weights")
        .ParamType(REQUIRED)
        .DataType({ge::DT_UINT8, ge::DT_UINT8})
        .FormatList({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("uvm_weights")
        .ParamType(REQUIRED)
        .DataType({ge::DT_UINT8, ge::DT_UINT8})
        .FormatList({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("weights_placements")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32})
        .FormatList({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("weights_offsets")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64, ge::DT_INT64})
        .FormatList({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("weights_tys")
        .ParamType(REQUIRED)
        .DataType({ge::DT_UINT8, ge::DT_UINT8})
        .FormatList({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("D_offsets")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32})
        .FormatList({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("update_weights")
        .ParamType(REQUIRED)
        .DataType({ge::DT_UINT8, ge::DT_UINT8})
        .FormatList({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("update_table_indices")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32})
        .FormatList({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("update_row_indices")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT64})
        .FormatList({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("update_offsets")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64, ge::DT_INT64})
        .FormatList({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

    this->Attr("row_alignment").AttrType(OPTIONAL).Int(1);

    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend950");
  }
};

OP_ADD(EmbInplaceUpdate);

} // namespace ops
