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
#include "direct_mapped_lru_cache_insert_byte_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace optiling {

// 属性索引常量
constexpr int ATTR_INDEX_GATHER_CACHE_STATS = 0;
constexpr int ATTR_INDEX_LRU_TIMESTAMP = 1;
constexpr int ATTR_INDEX_ROW_ALIGNMENT = 2;

// UVM 冲突统计最小元素数
constexpr int64_t UVM_STATS_MIN_FOR_CONFLICT = 5;

// Tiling 函数：Host 侧校验输入、计算切分参数，下发到 Kernel 侧
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 空指针校验
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape0", context->GetInputShape(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape1", context->GetInputShape(1), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape2", context->GetInputShape(2), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape3", context->GetInputShape(3), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape4", context->GetInputShape(4), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape5", context->GetInputShape(5), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape6", context->GetInputShape(6), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape7", context->GetInputShape(7), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape8", context->GetInputShape(8), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape9", context->GetInputShape(9), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape10", context->GetInputShape(10), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape11", context->GetInputShape(11), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape12", context->GetInputShape(12), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("gather_attr", context->GetAttrs()->GetBool(ATTR_INDEX_GATHER_CACHE_STATS),
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("lru_ts_attr", context->GetAttrs()->GetInt(ATTR_INDEX_LRU_TIMESTAMP), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("row_align_attr", context->GetAttrs()->GetInt(ATTR_INDEX_ROW_ALIGNMENT), return ge::GRAPH_FAILED);

    // 平台信息获取与 AI Core 核数校验
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        OPS_LOG_E(context, "[DirectMappedLruCacheInsertByte] coreNum is 0");
        return ge::GRAPH_FAILED;
    }

    // 设置 workspace 大小
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);
    currentWorkspace[0] = ascendPlatform.GetLibApiWorkSpaceSize();

    // 获取各输入张量的存储形状
    gert::Shape weightsShape = context->GetInputShape(0)->GetStorageShape();
    gert::Shape hashCumShape = context->GetInputShape(1)->GetStorageShape();
    gert::Shape mapShape = context->GetInputShape(2)->GetStorageShape();
    gert::Shape wOffShape = context->GetInputShape(3)->GetStorageShape();
    gert::Shape wTyShape = context->GetInputShape(4)->GetStorageShape();
    gert::Shape dOffShape = context->GetInputShape(5)->GetStorageShape();
    gert::Shape stateShape = context->GetInputShape(6)->GetStorageShape();
    gert::Shape cacheWShape = context->GetInputShape(7)->GetStorageShape();
    gert::Shape lruShape = context->GetInputShape(8)->GetStorageShape();
    gert::Shape linearIdxShape = context->GetInputShape(9)->GetStorageShape();
    gert::Shape timestampShape = context->GetInputShape(10)->GetStorageShape();
    gert::Shape cacheSetsShape = context->GetInputShape(11)->GetStorageShape();
    gert::Shape uvmShape = context->GetInputShape(12)->GetStorageShape();

    // 形状合法性校验
    OPS_LOG_E_IF(weightsShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[DirectMappedLruCacheInsertByte] weights must be 1D, got %lld", weightsShape.GetDimNum());
    OPS_LOG_E_IF(hashCumShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[DirectMappedLruCacheInsertByte] cache_hash_size_cumsum must be 1D, got %lld",
                 hashCumShape.GetDimNum());
    OPS_LOG_E_IF(mapShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[DirectMappedLruCacheInsertByte] cache_index_table_map must be 1D, got %lld", mapShape.GetDimNum());
    OPS_LOG_E_IF(wOffShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[DirectMappedLruCacheInsertByte] weights_offsets must be 1D, got %lld", wOffShape.GetDimNum());
    OPS_LOG_E_IF(wTyShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[DirectMappedLruCacheInsertByte] weights_tys must be 1D, got %lld", wTyShape.GetDimNum());
    OPS_LOG_E_IF(dOffShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[DirectMappedLruCacheInsertByte] d_offsets must be 1D, got %lld", dOffShape.GetDimNum());
    OPS_LOG_E_IF(stateShape.GetDimNum() != 2, context, return ge::GRAPH_FAILED,
                 "[DirectMappedLruCacheInsertByte] lxu_cache_state must be 2D [C,1], got %lld", stateShape.GetDimNum());
    OPS_LOG_E_IF(cacheWShape.GetDimNum() != 2, context, return ge::GRAPH_FAILED,
                 "[DirectMappedLruCacheInsertByte] lxu_cache_weights must be 2D, got %lld", cacheWShape.GetDimNum());
    OPS_LOG_E_IF(lruShape.GetDimNum() != 2, context, return ge::GRAPH_FAILED,
                 "[DirectMappedLruCacheInsertByte] lru_state must be 2D [C,1], got %lld", lruShape.GetDimNum());
    OPS_LOG_E_IF(linearIdxShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[DirectMappedLruCacheInsertByte] linear_cache_indices must be 1D, got %lld",
                 linearIdxShape.GetDimNum());
    OPS_LOG_E_IF(timestampShape.GetDimNum() != 2, context, return ge::GRAPH_FAILED,
                 "[DirectMappedLruCacheInsertByte] lxu_cache_miss_timestamp must be 2D [C,1], got %lld",
                 timestampShape.GetDimNum());
    OPS_LOG_E_IF(cacheSetsShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[DirectMappedLruCacheInsertByte] cache_sets must be 1D, got %lld", cacheSetsShape.GetDimNum());
    OPS_LOG_E_IF(uvmShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[DirectMappedLruCacheInsertByte] uvm_cache_stats must be 1D, got %lld", uvmShape.GetDimNum());

    // 提取维度信息
    int64_t weightsLen = weightsShape.GetDim(0);
    int64_t hashCumLen = hashCumShape.GetDim(0);
    int64_t mapLen = mapShape.GetDim(0);
    int64_t wOffLen = wOffShape.GetDim(0);
    int64_t wTyLen = wTyShape.GetDim(0);
    int64_t dOffLen = dOffShape.GetDim(0);
    int64_t C = stateShape.GetDim(0);
    int64_t W = stateShape.GetDim(1);
    int64_t cacheRows = cacheWShape.GetDim(0);
    int64_t rowBytes = cacheWShape.GetDim(1);
    int64_t linearIdxLen = linearIdxShape.GetDim(0);
    int64_t uvmLen = uvmShape.GetDim(0);

    OPS_CHECK(dOffLen < 2, OPS_LOG_E(context, "[DirectMappedLruCacheInsertByte] d_offsets must have length >= 2"),
              return ge::GRAPH_FAILED);
    int64_t numTables = dOffLen - 1;

    OPS_CHECK(hashCumLen < numTables,
              OPS_LOG_E(context, "[DirectMappedLruCacheInsertByte] cache_hash_size_cumsum length %lld < numTables %lld",
                        hashCumLen, numTables),
              return ge::GRAPH_FAILED);

    OPS_CHECK(
        wOffLen != numTables,
        OPS_LOG_E(context, "[DirectMappedLruCacheInsertByte] weights_offsets length %lld must equal numTables %lld",
                  wOffLen, numTables),
        return ge::GRAPH_FAILED);

    OPS_CHECK(wTyLen != numTables,
              OPS_LOG_E(context, "[DirectMappedLruCacheInsertByte] weights_tys length %lld must equal numTables %lld",
                        wTyLen, numTables),
              return ge::GRAPH_FAILED);

    // Direct-mapped 模式校验：lxu_cache_state 形状为 [C, 1]
    OPS_CHECK(W != 1,
              OPS_LOG_E(context,
                        "[DirectMappedLruCacheInsertByte] for direct-mapped, lxu_cache_state W must be 1, got %lld", W),
              return ge::GRAPH_FAILED);

    OPS_CHECK(
        cacheRows != C,
        OPS_LOG_E(context,
                  "[DirectMappedLruCacheInsertByte] lxu_cache_weights rows (%lld) must equal C (%lld) from cache state",
                  cacheRows, C),
        return ge::GRAPH_FAILED);

    // LRU 与 Cache State 形状一致性校验
    gert::Shape lruShapeCheck = context->GetInputShape(8)->GetStorageShape();
    OPS_CHECK(stateShape.GetDim(0) != lruShapeCheck.GetDim(0) || stateShape.GetDim(1) != lruShapeCheck.GetDim(1),
              OPS_LOG_E(context, "[DirectMappedLruCacheInsertByte] lxu_cache_state and lru_state shape mismatch"),
              return ge::GRAPH_FAILED);

    // 统计收集模式下的额外校验
    const bool gatherStats = *context->GetAttrs()->GetBool(ATTR_INDEX_GATHER_CACHE_STATS);
    if (gatherStats) {
        OPS_CHECK(
            uvmLen < UVM_STATS_MIN_FOR_CONFLICT,
            OPS_LOG_E(
                context,
                "[DirectMappedLruCacheInsertByte] gather_cache_stats requires uvm_cache_stats length >= %lld, got %lld",
                UVM_STATS_MIN_FOR_CONFLICT, uvmLen),
            return ge::GRAPH_FAILED);
    }

    // 提取属性标量值
    const int64_t timeStamp = *context->GetAttrs()->GetInt(ATTR_INDEX_LRU_TIMESTAMP);
    const int64_t rowAlignment = *context->GetAttrs()->GetInt(ATTR_INDEX_ROW_ALIGNMENT);

    // 填充 Tiling 数据，下发给每个 AI Core
    DirectMappedLruCacheInsertByteTilingData tiling;
    tiling.set_totalLength(linearIdxLen);
    tiling.set_numCacheSets(C);
    tiling.set_cacheWeightsRowBytes(rowBytes);
    tiling.set_weightsTotalLength(weightsLen);
    tiling.set_uvmStatsLength(uvmLen);
    tiling.set_gatherCacheStats(gatherStats ? 1 : 0);
    tiling.set_timeStamp(timeStamp);
    tiling.set_rowAlignment(rowAlignment);
    tiling.set_numTables(numTables);
    tiling.set_hashCumsumLength(hashCumLen);
    tiling.set_cacheIndexMapLength(mapLen);

    // 设置 Block 维度并按 Core 切分任务
    context->SetBlockDim(static_cast<uint32_t>(coreNum));
    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

// 输出形状推导：reserved_out 固定为 [1]
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    gert::Shape* out = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("out", out, return ge::GRAPH_FAILED);
    out->SetDimNum(1);
    out->SetDim(0, 1);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class DirectMappedLruCacheInsertByte : public OpDef {
public:
    // 算子原型注册：定义输入/输出/属性及回调
    explicit DirectMappedLruCacheInsertByte(const char* name) : OpDef(name)
    {
        // 输入定义
        this->Input("weights").ParamType(REQUIRED).DataTypeList({ge::DT_UINT8}).FormatList({ge::FORMAT_ND});
        this->Input("cache_hash_size_cumsum")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("cache_index_table_map")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("weights_offsets").ParamType(REQUIRED).DataTypeList({ge::DT_INT64}).FormatList({ge::FORMAT_ND});
        this->Input("weights_tys").ParamType(REQUIRED).DataTypeList({ge::DT_UINT8}).FormatList({ge::FORMAT_ND});
        this->Input("d_offsets").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("lxu_cache_state").ParamType(REQUIRED).DataTypeList({ge::DT_INT64}).FormatList({ge::FORMAT_ND});
        this->Input("lxu_cache_weights").ParamType(REQUIRED).DataTypeList({ge::DT_UINT8}).FormatList({ge::FORMAT_ND});
        this->Input("lru_state").ParamType(REQUIRED).DataTypeList({ge::DT_INT64}).FormatList({ge::FORMAT_ND});
        this->Input("linear_cache_indices")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("lxu_cache_miss_timestamp")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("cache_sets").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("uvm_cache_stats").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

        // 输出定义
        this->Output("reserved_out").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

        // 属性定义

        this->Attr("gather_cache_stats").AttrType(OPTIONAL).Bool(false);
        this->Attr("lru_timestamp").AttrType(REQUIRED).Int();
        this->Attr("row_alignment").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape);

        // 注册 Tiling 回调及支持的硬件平台
        this->AICore().SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
#ifdef SUPPORT_950
        this->AICore().AddConfig("ascend950");
#endif
    }
};

OP_ADD(DirectMappedLruCacheInsertByte);

}  // namespace ops
