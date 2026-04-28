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

#include "lru_cache_find_uncached_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace optiling {

// 与 OpDef 中 Attr 注册顺序一致
constexpr int ATTR_INDEX_GATHER_CACHE_STATS = 0;
constexpr int ATTR_INDEX_MAX_INDICES = 1;
// 使用 lru_timestamp 而非 time_stamp：CANN aclnn 生成代码会用局部变量 timeStamp 做 profiling，与 attr 映射的形参名冲突
constexpr int ATTR_INDEX_LRU_TIMESTAMP = 2;
constexpr int ATTR_INDEX_LOCK_CACHE_LINE = 3;
constexpr int64_t UVM_STATS_MIN_ELEMENTS = 4;

constexpr int32_t INPUT_INDEX = 0;
constexpr int32_t INPUT_INDEX_LENGTH = 1;
constexpr int32_t INPUT_INDEX_STATE = 2;
constexpr int32_t INPUT_INDEX_LRU_STATE = 3;
constexpr int32_t INPUT_INDEX_UVM_STATS = 4;
constexpr int32_t INPUT_INDEX_LOCK_COUNTER = 5;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape0", context->GetInputShape(INPUT_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape2", context->GetInputShape(INPUT_INDEX_STATE), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape4", context->GetInputShape(INPUT_INDEX_UVM_STATS), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape5", context->GetInputShape(INPUT_INDEX_LOCK_COUNTER), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("gather_cache_stats_attr", context->GetAttrs()->GetBool(ATTR_INDEX_GATHER_CACHE_STATS),
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("max_indices_attr", context->GetAttrs()->GetInt(ATTR_INDEX_MAX_INDICES), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("lru_timestamp_attr", context->GetAttrs()->GetInt(ATTR_INDEX_LRU_TIMESTAMP), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("lock_cache_line_attr", context->GetAttrs()->GetBool(ATTR_INDEX_LOCK_CACHE_LINE),
                      return ge::GRAPH_FAILED);

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        OPS_LOG_E(context, "[LruCacheFindUncached] coreNum is 0");
        return ge::GRAPH_FAILED;
    }

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;

    gert::Shape uniqueShape = context->GetInputShape(INPUT_INDEX)->GetStorageShape();
    gert::Shape stateShape = context->GetInputShape(INPUT_INDEX_STATE)->GetStorageShape();
    gert::Shape uvmShape = context->GetInputShape(INPUT_INDEX_UVM_STATS)->GetStorageShape();
    gert::Shape lockShape = context->GetInputShape(INPUT_INDEX_LOCK_COUNTER)->GetStorageShape();

    OPS_LOG_E_IF(uniqueShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[LruCacheFindUncached] unique_indices must be 1D");
    OPS_LOG_E_IF(stateShape.GetDimNum() != 2, context, return ge::GRAPH_FAILED,
                 "[LruCacheFindUncached] lxu_cache_state must be 2D [C,W]");
    OPS_LOG_E_IF(uvmShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[LruCacheFindUncached] uvm_cache_stats must be 1D");
    OPS_LOG_E_IF(lockShape.GetDimNum() != 2, context, return ge::GRAPH_FAILED,
                 "[LruCacheFindUncached] lxu_cache_locking_counter must be 2D [C,W]");

    int64_t N = uniqueShape.GetDim(0);
    int64_t C = stateShape.GetDim(0);
    int64_t W = stateShape.GetDim(1);
    int64_t uvmLen = uvmShape.GetDim(0);
    int64_t lockC = lockShape.GetDim(0);
    int64_t lockW = lockShape.GetDim(1);
    int64_t lockLen = lockC * lockW;

    OPS_CHECK(N <= 0 || C <= 0 || W <= 0,
              OPS_LOG_E(context, "[LruCacheFindUncached] invalid shape N,C,W"),
              return ge::GRAPH_FAILED);

    const bool gatherStats = *context->GetAttrs()->GetBool(ATTR_INDEX_GATHER_CACHE_STATS);
    const bool lockLine = *context->GetAttrs()->GetBool(ATTR_INDEX_LOCK_CACHE_LINE);

    if (gatherStats) {
        OPS_CHECK(uvmLen < UVM_STATS_MIN_ELEMENTS,
                  OPS_LOG_E(context,
                            "[LruCacheFindUncached] gather_cache_stats requires uvm_cache_stats "
                            "length >= %lld, got %lld",
                            UVM_STATS_MIN_ELEMENTS, uvmLen),
                  return ge::GRAPH_FAILED);
    }
    if (lockLine) {
        OPS_CHECK(lockLen != C * W,
                  OPS_LOG_E(context,
                            "[LruCacheFindUncached] lock_cache_line requires "
                            "lxu_cache_locking_counter [C,W] same as lxu_cache_state, got [%lld,%lld] vs C,W=%lld,%lld",
                            lockC, lockW, C, W),
                  return ge::GRAPH_FAILED);
    }

    const int64_t maxIndices = *context->GetAttrs()->GetInt(ATTR_INDEX_MAX_INDICES);
    const int64_t timeStamp = *context->GetAttrs()->GetInt(ATTR_INDEX_LRU_TIMESTAMP);

    LruCacheFindUncachedTilingData tiling;
    tiling.set_totalLength(N);
    tiling.set_numCacheSets(C);
    tiling.set_numWays(W);
    tiling.set_uvmStatsLength(uvmLen);
    tiling.set_lockCounterLength(lockLen);
    tiling.set_gatherCacheStats(gatherStats ? 1 : 0);
    tiling.set_maxIndices(maxIndices);
    tiling.set_timeStamp(timeStamp);
    tiling.set_lockCacheLine(lockLine ? 1 : 0);

    // 与 device 侧 GetBlockIdx/GetBlockNum 分片一致：每颗 AIV 跑一份 SIMT VF
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
    const gert::Shape* u = context->GetInputShape(0);
    gert::Shape* cacheSets = context->GetOutputShape(0);

    OPS_LOG_E_IF_NULL("u", u, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("cacheSets", cacheSets, return ge::GRAPH_FAILED);

    int64_t N = u->GetDim(0);
    cacheSets->SetDimNum(1);
    cacheSets->SetDim(0, N);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class LruCacheFindUncached : public OpDef {
public:
    explicit LruCacheFindUncached(const char* name) : OpDef(name)
    {
        this->Input("unique_indices")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("unique_indices_length")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("lxu_cache_state")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("lru_state")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("uvm_cache_stats")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("lxu_cache_locking_counter")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        this->Output("cache_sets")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        this->Attr("gather_cache_stats").AttrType(OPTIONAL).Bool(false);
        this->Attr("max_indices").AttrType(REQUIRED).Int();
        this->Attr("lru_timestamp").AttrType(REQUIRED).Int();
        this->Attr("lock_cache_line").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(LruCacheFindUncached);

}  // namespace ops
