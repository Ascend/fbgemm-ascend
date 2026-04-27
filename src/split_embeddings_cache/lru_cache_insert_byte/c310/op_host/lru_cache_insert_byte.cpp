/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
#include "lru_cache_insert_byte_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace optiling {

constexpr int ATTR_INDEX_GATHER_CACHE_STATS = 0;
constexpr int ATTR_INDEX_LRU_TIMESTAMP = 1;
constexpr int ATTR_INDEX_ROW_ALIGNMENT = 2;

// fbgemm uvm_cache_stats_index::num_conflict_unique_misses == 4
constexpr int64_t UVM_STATS_MIN_FOR_CONFLICT = 5;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape6", context->GetInputShape(6), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape7", context->GetInputShape(7), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape9", context->GetInputShape(9), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape10", context->GetInputShape(10), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape11", context->GetInputShape(11), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape2", context->GetInputShape(2), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape1", context->GetInputShape(1), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape3", context->GetInputShape(3), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape4", context->GetInputShape(4), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape5", context->GetInputShape(5), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("shape12", context->GetInputShape(12), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("gather_attr", context->GetAttrs()->GetBool(ATTR_INDEX_GATHER_CACHE_STATS),
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("lru_ts_attr", context->GetAttrs()->GetInt(ATTR_INDEX_LRU_TIMESTAMP),
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("row_align_attr", context->GetAttrs()->GetInt(ATTR_INDEX_ROW_ALIGNMENT),
                      return ge::GRAPH_FAILED);

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        OPS_LOG_E(context, "[LruCacheInsertByte] coreNum is 0");
        return ge::GRAPH_FAILED;
    }

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);
    currentWorkspace[0] = ascendPlatform.GetLibApiWorkSpaceSize();

    gert::Shape sortedSetsShape = context->GetInputShape(6)->GetStorageShape();
    gert::Shape sortedIdxShape = context->GetInputShape(7)->GetStorageShape();
    gert::Shape stateShape = context->GetInputShape(9)->GetStorageShape();
    gert::Shape cacheWShape = context->GetInputShape(10)->GetStorageShape();
    gert::Shape lruShape = context->GetInputShape(11)->GetStorageShape();
    gert::Shape uvmShape = context->GetInputShape(12)->GetStorageShape();
    gert::Shape weightsShape = context->GetInputShape(0)->GetStorageShape();
    gert::Shape dOffShape = context->GetInputShape(5)->GetStorageShape();
    gert::Shape hashCumShape = context->GetInputShape(1)->GetStorageShape();
    gert::Shape mapShape = context->GetInputShape(2)->GetStorageShape();
    gert::Shape wOffShape = context->GetInputShape(3)->GetStorageShape();
    gert::Shape wTyShape = context->GetInputShape(4)->GetStorageShape();

    OPS_LOG_E_IF(sortedSetsShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[LruCacheInsertByte] sorted_cache_sets must be 1D");
    OPS_LOG_E_IF(sortedIdxShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[LruCacheInsertByte] cache_set_sorted_unique_indices must be 1D");
    OPS_LOG_E_IF(stateShape.GetDimNum() != 2, context, return ge::GRAPH_FAILED,
                 "[LruCacheInsertByte] lxu_cache_state must be 2D [C,W]");
    OPS_LOG_E_IF(cacheWShape.GetDimNum() != 2, context, return ge::GRAPH_FAILED,
                 "[LruCacheInsertByte] lxu_cache_weights must be 2D");
    OPS_LOG_E_IF(lruShape.GetDimNum() != 2, context, return ge::GRAPH_FAILED,
                 "[LruCacheInsertByte] lru_state must be 2D [C,W]");
    OPS_LOG_E_IF(uvmShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[LruCacheInsertByte] uvm_cache_stats must be 1D");
    OPS_LOG_E_IF(weightsShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[LruCacheInsertByte] weights must be 1D");
    OPS_LOG_E_IF(dOffShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[LruCacheInsertByte] d_offsets must be 1D");
    OPS_LOG_E_IF(mapShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                 "[LruCacheInsertByte] cache_index_table_map must be 1D");

    int64_t Nbuf = sortedSetsShape.GetDim(0);
    int64_t Nidx = sortedIdxShape.GetDim(0);
    int64_t C = stateShape.GetDim(0);
    int64_t W = stateShape.GetDim(1);
    int64_t cwState = C * W;
    int64_t cacheRows = cacheWShape.GetDim(0);
    int64_t rowBytes = cacheWShape.GetDim(1);
    int64_t uvmLen = uvmShape.GetDim(0);
    int64_t weightsLen = weightsShape.GetDim(0);
    int64_t dOffLen = dOffShape.GetDim(0);
    int64_t hashCumLen = hashCumShape.GetDim(0);
    int64_t mapLen = mapShape.GetDim(0);
    int64_t wOffLen = wOffShape.GetDim(0);
    int64_t wTyLen = wTyShape.GetDim(0);
    OPS_CHECK(
        dOffLen < 2,
        OPS_LOG_E(context, "[LruCacheInsertByte] d_offsets must have length >= 2"),
        return ge::GRAPH_FAILED);
    int64_t numTables = dOffLen - 1;
    OPS_CHECK(
        hashCumLen < numTables,
        OPS_LOG_E(context, "[LruCacheInsertByte] cache_hash_size_cumsum length %lld < numTables %lld", hashCumLen,
                  numTables),
        return ge::GRAPH_FAILED);
    OPS_CHECK(
        wOffLen != numTables,
        OPS_LOG_E(context, "[LruCacheInsertByte] weights_offsets length %lld must equal numTables %lld", wOffLen,
                  numTables),
        return ge::GRAPH_FAILED);
    OPS_CHECK(
        wTyLen != numTables,
        OPS_LOG_E(context, "[LruCacheInsertByte] weights_tys length %lld must equal numTables %lld", wTyLen,
                  numTables),
        return ge::GRAPH_FAILED);

    OPS_CHECK(
        Nbuf != Nidx,
        OPS_LOG_E(context, "[LruCacheInsertByte] sorted_cache_sets and cache_set_sorted_unique_indices "
                           "length mismatch: %lld vs %lld",
                  Nbuf, Nidx),
        return ge::GRAPH_FAILED);

    OPS_CHECK(
        stateShape.GetDim(0) != lruShape.GetDim(0) || stateShape.GetDim(1) != lruShape.GetDim(1),
        OPS_LOG_E(context, "[LruCacheInsertByte] lxu_cache_state and lru_state shape mismatch"),
        return ge::GRAPH_FAILED);

    OPS_CHECK(
        cacheRows != cwState,
        OPS_LOG_E(context,
                  "[LruCacheInsertByte] lxu_cache_weights rows (%lld) must equal C*W (%lld) from cache state",
                  cacheRows, cwState),
        return ge::GRAPH_FAILED);

    // Align with FBGEMM lru_cache_insert_byte_kernel (kWarpSize == 32 ways).
    OPS_CHECK(
        W != 32,
        OPS_LOG_E(context, "[LruCacheInsertByte] num ways W must be 32 (FBGEMM insert kernel), got %lld", W),
        return ge::GRAPH_FAILED);

    const bool gatherStats = *context->GetAttrs()->GetBool(ATTR_INDEX_GATHER_CACHE_STATS);
    if (gatherStats) {
        OPS_CHECK(
            uvmLen < UVM_STATS_MIN_FOR_CONFLICT,
            OPS_LOG_E(context,
                      "[LruCacheInsertByte] gather_cache_stats requires uvm_cache_stats length >= %lld, got %lld",
                      UVM_STATS_MIN_FOR_CONFLICT, uvmLen),
            return ge::GRAPH_FAILED);
    }

    const int64_t timeStamp = *context->GetAttrs()->GetInt(ATTR_INDEX_LRU_TIMESTAMP);
    const int64_t rowAlignment = *context->GetAttrs()->GetInt(ATTR_INDEX_ROW_ALIGNMENT);

    LruCacheInsertByteTilingData tiling;
    tiling.set_bufferLength(Nbuf);
    tiling.set_numCacheSets(C);
    tiling.set_numWays(W);
    tiling.set_cacheWeightsRowBytes(rowBytes);
    tiling.set_weightsTotalLength(weightsLen);
    tiling.set_uvmStatsLength(uvmLen);
    tiling.set_gatherCacheStats(gatherStats ? 1 : 0);
    tiling.set_timeStamp(timeStamp);
    tiling.set_rowAlignment(rowAlignment);
    tiling.set_numTables(numTables);
    tiling.set_hashCumsumLength(hashCumLen);
    tiling.set_cacheIndexMapLength(mapLen);

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
    gert::Shape* out = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("out", out, return ge::GRAPH_FAILED);
    out->SetDimNum(1);
    out->SetDim(0, 1);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class LruCacheInsertByte : public OpDef {
public:
    explicit LruCacheInsertByte(const char* name) : OpDef(name)
    {
        this->Input("weights")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_UINT8})
            .FormatList({ge::FORMAT_ND});
        this->Input("cache_hash_size_cumsum")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("cache_index_table_map")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("weights_offsets")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("weights_tys")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_UINT8})
            .FormatList({ge::FORMAT_ND});
        this->Input("d_offsets")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("sorted_cache_sets")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("cache_set_sorted_unique_indices")
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
        this->Input("lxu_cache_weights")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_UINT8})
            .FormatList({ge::FORMAT_ND});
        this->Input("lru_state")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("uvm_cache_stats")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        this->Output("reserved_out")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        this->Attr("gather_cache_stats").AttrType(OPTIONAL).Bool(false);
        this->Attr("lru_timestamp").AttrType(REQUIRED).Int();
        this->Attr("row_alignment").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(LruCacheInsertByte);

}  // namespace ops
