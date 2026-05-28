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
#include "direct_mapped_lru_cache_find_uncached_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace optiling {

// 属性索引常量
constexpr int ATTR_INDEX_MAX_INDICES = 0;

// 使用 lru_timestamp 而非 time_stamp：CANN aclnn 生成代码会用局部变量 timeStamp 做 profiling，与 attr 映射的形参名冲突
constexpr int ATTR_INDEX_LRU_TIMESTAMP = 1;

constexpr int ATTR_INDEX_GATHER_CACHE_STATS = 2;

// 输入张量索引常量
constexpr int INPUT_INDEX_LINEAR_CACHE_INDICES = 0;
constexpr int INPUT_INDEX_LXU_CACHE_STATE = 1;
constexpr int INPUT_INDEX_LRU_STATE = 2;
constexpr int INPUT_INDEX_LXU_CACHE_MISS_TIMESTAMP = 3;
constexpr int INPUT_INDEX_UVM_CACHE_STATS = 4;

// uvm_cache_stats 最小元素个数
constexpr int64_t UVM_STATS_MIN_ELEMENTS = 3;

// Tiling 函数：Host 侧校验输入、计算切分参数，下发到 Kernel 侧
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 空指针校验
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("linear_cache_indices", context->GetInputShape(INPUT_INDEX_LINEAR_CACHE_INDICES),
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("lxu_cache_state", context->GetInputShape(INPUT_INDEX_LXU_CACHE_STATE), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("lxu_cache_miss_timestamp", context->GetInputShape(INPUT_INDEX_LXU_CACHE_MISS_TIMESTAMP),
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("uvm_cache_stats", context->GetInputShape(INPUT_INDEX_UVM_CACHE_STATS), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("max_indices_attr", context->GetAttrs()->GetInt(ATTR_INDEX_MAX_INDICES), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("lru_timestamp_attr", context->GetAttrs()->GetInt(ATTR_INDEX_LRU_TIMESTAMP),
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("gather_cache_stats_attr", context->GetAttrs()->GetBool(ATTR_INDEX_GATHER_CACHE_STATS),
                      return ge::GRAPH_FAILED);

    // 平台信息获取与 AI Core 核数校验
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        OPS_LOG_E(context, "[DirectMappedLruCacheFindUncached] coreNum is 0");
        return ge::GRAPH_FAILED;
    }

    // 设置 workspace 大小
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;

    // 获取各输入张量的存储形状
    gert::Shape indiceShape = context->GetInputShape(INPUT_INDEX_LINEAR_CACHE_INDICES)->GetStorageShape();
    gert::Shape stateShape = context->GetInputShape(INPUT_INDEX_LXU_CACHE_STATE)->GetStorageShape();
    gert::Shape timestampShape = context->GetInputShape(INPUT_INDEX_LXU_CACHE_MISS_TIMESTAMP)->GetStorageShape();
    gert::Shape uvmShape = context->GetInputShape(INPUT_INDEX_UVM_CACHE_STATS)->GetStorageShape();

    // 形状合法性校验（direct-mapped 要求 cache_state 为 [C, 1]）
    OPS_LOG_E_IF(indiceShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[DirectMappedLruFindUncached] linear_cache_indices must be 1D");
    OPS_LOG_E_IF(stateShape.GetDimNum() != 2, context, return ge::GRAPH_FAILED,
                 "[DirectMappedLruFindUncached] lxu_cache_state must be 2D [C,1]");
    OPS_LOG_E_IF(timestampShape.GetDimNum() != 2, context, return ge::GRAPH_FAILED,
                 "[DirectMappedLruFindUncached] lxu_cache_miss_timestamp must be 2D [C,1]");
    OPS_LOG_E_IF(uvmShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[DirectMappedLruFindUncached] uvm_cache_stats must be 1D");

    // 提取维度信息：N = 索引数，C = cache set 数
    int64_t N = indiceShape.GetDim(0);
    int64_t C = stateShape.GetDim(0);
    int64_t timestampC = timestampShape.GetDim(0);
    int64_t uvmLen = uvmShape.GetDim(0);

    OPS_CHECK(N <= 0 || C <= 0, OPS_LOG_E(context, "[DirectMappedLruFindUncached] invalid shape N=%lld, C=%lld", N, C),
              return ge::GRAPH_FAILED);

    const bool gatherStats = *context->GetAttrs()->GetBool(ATTR_INDEX_GATHER_CACHE_STATS);

    // 统计收集模式下的额外校验
    if (gatherStats) {
        OPS_CHECK(uvmLen < UVM_STATS_MIN_ELEMENTS,
                  OPS_LOG_E(context,
                            "[DirectMappedLruFindUncached] gather_cache_stats requires uvm_cache_stats "
                            "length >= %lld, got %lld",
                            UVM_STATS_MIN_ELEMENTS, uvmLen),
                  return ge::GRAPH_FAILED);
    }

    // 提取属性标量值
    const int64_t maxIndices = *context->GetAttrs()->GetInt(ATTR_INDEX_MAX_INDICES);
    const int64_t timeStamp = *context->GetAttrs()->GetInt(ATTR_INDEX_LRU_TIMESTAMP);

    // 填充 Tiling 数据，下发给每个 AI Core
    DirectMappedLruCacheFindUncachedTilingData tiling;
    tiling.set_totalLength(N);
    tiling.set_numCacheSets(C);
    tiling.set_uvmStatsLength(uvmLen);
    tiling.set_gatherCacheStats(gatherStats ? 1 : 0);
    tiling.set_maxIndices(maxIndices);
    tiling.set_timeStamp(timeStamp);

    // 设置 Block 维度并按 Core 切分任务
    context->SetBlockDim(static_cast<uint32_t>(coreNum));
    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

// 输出形状推导：cache_sets 与输入 linear_cache_indices 形状一致 [N]
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

class DirectMappedLruCacheFindUncached : public OpDef {
public:
    // 算子原型注册：定义输入/输出/属性及回调
    explicit DirectMappedLruCacheFindUncached(const char* name) : OpDef(name)
    {
        // 输入定义
        this->Input("linear_cache_indices")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("lxu_cache_state").ParamType(REQUIRED).DataTypeList({ge::DT_INT64}).FormatList({ge::FORMAT_ND});
        this->Input("lru_state").ParamType(REQUIRED).DataTypeList({ge::DT_INT64}).FormatList({ge::FORMAT_ND});
        this->Input("lxu_cache_miss_timestamp")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("uvm_cache_stats").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

        // 输出定义
        this->Output("cache_sets").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

        // 属性定义
        this->Attr("max_indices").AttrType(REQUIRED).Int();
        this->Attr("lru_timestamp").AttrType(REQUIRED).Int();
        this->Attr("gather_cache_stats").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape);

        // 注册 Tiling 回调及支持的硬件平台
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(DirectMappedLruCacheFindUncached);

}  // namespace ops
