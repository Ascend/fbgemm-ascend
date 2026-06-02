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

#include "lxu_cache_lookup_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace optiling {

constexpr int ATTR_INDEX_INVALID_INDEX = 0;
constexpr int ATTR_INDEX_GATHER_CACHE_STATS = 1;
constexpr int ATTR_INDEX_UNIQ_LOOKUP = 2;
constexpr int64_t UVM_STATS_MIN_ELEMENTS = 4;

constexpr int32_t INPUT_INDEX_LINEAR = 0;
constexpr int32_t INPUT_INDEX_STATE = 1;
constexpr int32_t INPUT_INDEX_UVM_STATS = 2;
constexpr int32_t INPUT_INDEX_NUM_UNIQ = 3;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape0", context->GetInputShape(INPUT_INDEX_LINEAR), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape1", context->GetInputShape(INPUT_INDEX_STATE), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        OPS_LOG_E(context, "[LxuCacheLookup] coreNum is 0");
        return ge::GRAPH_FAILED;
    }

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;

    gert::Shape indicesShape = context->GetInputShape(INPUT_INDEX_LINEAR)->GetStorageShape();
    gert::Shape stateShape = context->GetInputShape(INPUT_INDEX_STATE)->GetStorageShape();

    OPS_LOG_E_IF(indicesShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[LxuCacheLookup] linear_cache_indices must be 1D");
    OPS_LOG_E_IF(stateShape.GetDimNum() != 2, context, return ge::GRAPH_FAILED,
                 "[LxuCacheLookup] lxu_cache_state must be 2D [C,W]");

    int64_t N = indicesShape.GetDim(0);
    int64_t C = stateShape.GetDim(0);
    int64_t W = stateShape.GetDim(1);

    OPS_CHECK(N <= 0 || C <= 0 || W <= 0,
              OPS_LOG_E(context, "[LxuCacheLookup] invalid shape N(%lld),C(%lld),W(%lld)", N, C, W),
              return ge::GRAPH_FAILED);

    const bool gatherStats = *context->GetAttrs()->GetBool(ATTR_INDEX_GATHER_CACHE_STATS);
    const bool uniqLookup = *context->GetAttrs()->GetBool(ATTR_INDEX_UNIQ_LOOKUP);
    const int64_t invalidIndex = *context->GetAttrs()->GetInt(ATTR_INDEX_INVALID_INDEX);

    LxuCacheLookupTilingData tiling;
    tiling.set_totalLength(N);
    tiling.set_numCacheSets(C);
    tiling.set_numWays(W);
    tiling.set_gatherCacheStats(gatherStats ? 1 : 0);
    tiling.set_invalidIndex(invalidIndex);
    tiling.set_uniqLookup(uniqLookup ? 1 : 0);

    context->SetBlockDim(static_cast<uint32_t>(coreNum));
    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const gert::Shape* indices = context->GetInputShape(0);
    gert::Shape* locations = context->GetOutputShape(0);

    OPS_LOG_E_IF_NULL("indices", indices, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("locations", locations, return ge::GRAPH_FAILED);

    int64_t N = indices->GetDim(0);
    locations->SetDimNum(1);
    locations->SetDim(0, N);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class LxuCacheLookup : public OpDef {
public:
    explicit LxuCacheLookup(const char* name) : OpDef(name)
    {
        this->Input("linear_cache_indices")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("lxu_cache_state").ParamType(REQUIRED).DataTypeList({ge::DT_INT64}).FormatList({ge::FORMAT_ND});
        this->Input("uvm_cache_stats").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("num_uniq_cache_indices")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        this->Output("lxu_cache_locations")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        this->Attr("invalid_index").AttrType(REQUIRED).Int();
        this->Attr("gather_cache_stats").AttrType(OPTIONAL).Bool(false);
        this->Attr("uniq_lookup").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(LxuCacheLookup);

}  // namespace ops
