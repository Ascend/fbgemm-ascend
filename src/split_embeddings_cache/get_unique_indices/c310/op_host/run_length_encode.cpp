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
#include <limits>
#include <vector>

#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"
#include "run_length_encode_tiling.h"

namespace
{
    constexpr int32_t RETURN_COUNTS_IDX = 0;
    constexpr bool DEFAULT_RETURN_COUNTS = false;

    constexpr int64_t MAX_BYTES_SINGLE_CORE = 1024;
    constexpr int64_t MAGIC_GM_PAGE_SIZE = 128;
    constexpr int32_t SHAPE_LEN = 27;

    constexpr uint64_t TILING_KEY_SINGLE_CORE = 10;
    constexpr uint64_t TILING_KEY_MULTI_CORE = 20;
    constexpr uint64_t TILING_KEY_EMPTY = 666;

    constexpr uint32_t DEFAULT_BLOCK_SIZE = 32;

    constexpr int64_t UNKNOWN_SHAPE_DIM = -1;
    constexpr int64_t MAX_SUPPORTED_TOTAL_SIZE = static_cast<int64_t>(std::numeric_limits<int32_t>::max());

    constexpr size_t OUT_IDX = 0;
    constexpr size_t COUNTS_IDX = 1;
    constexpr size_t LENGTH_IDX = 2;

    template <typename T>
    static inline T CeilDivInt(const T x, const T y)
    {
        return (y == 0) ? 0 : (x + y - 1) / y;
    }

    template <typename T>
    static inline T CeilAlignInt(const T x, const T align)
    {
        return CeilDivInt<T>(x, align) * align;
    }

    template <typename T>
    static inline T GetOptionalAttr(const gert::RuntimeAttrs *attrs, const int idx, const T &defaultValue)
    {
        const T *attrPtr = attrs->GetAttrPointer<T>(idx);
        if (attrPtr == nullptr)
        {
            OPS_LOG_I("GetOptionalAttr", "attr[%d] unavailable, fallback to default", idx);
        }
        return (attrPtr == nullptr) ? defaultValue : *attrPtr;
    }

} // namespace

namespace optiling
{

    bool RunLengthEncodeTilingHelper::DoTiling()
    {
        OPS_CHECK(!GetBaseInfo(), OPS_LOG_E("DoTiling", "GetBaseInfo failed."), return false);
        OPS_CHECK(!GetShapeInfo(), OPS_LOG_E("DoTiling", "GetShapeInfo failed."), return false);
        OPS_CHECK(!DoBlockTiling(), OPS_LOG_E("DoTiling", "DoBlockTiling failed."), return false);
        OPS_CHECK(!DoUbTiling(), OPS_LOG_E("DoTiling", "DoUbTiling failed."), return false);
        OPS_CHECK(!ComputeWorkspaces(), OPS_LOG_E("DoTiling", "ComputeWorkspaces failed."), return false);
        return true;
    }

    bool RunLengthEncodeTilingHelper::GetBaseInfo()
    {
        OPS_CHECK(!GetPlatformInfo(), OPS_LOG_E("GetBaseInfo", "GetPlatformInfo failed."), return false);
        OPS_CHECK(!GetAttrs(), OPS_LOG_E("GetBaseInfo", "GetAttrs failed."), return false);
        return true;
    }

    bool RunLengthEncodeTilingHelper::GetPlatformInfo()
    {
        auto platformInfo = context_->GetPlatformInfo();
        OPS_LOG_E_IF_NULL(context_->GetNodeName(), platformInfo, return false);

        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
        aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        blockSize_ = DEFAULT_BLOCK_SIZE;
        sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

        OPS_CHECK((ubSize_ <= 0), OPS_LOG_E(context_->GetNodeName(), "ubSize invalid: %lu", ubSize_), return false);
        OPS_CHECK((aivCoreNum_ <= 0), OPS_LOG_E(context_->GetNodeName(), "aivCoreNum invalid: %u", aivCoreNum_),
                  return false);

        OPS_LOG_I("GetPlatformInfo", "aivCoreNum=%u, ubSize=%lu, blockSize=%u", aivCoreNum_, ubSize_, blockSize_);
        return true;
    }

    bool RunLengthEncodeTilingHelper::GetAttrs()
    {
        auto attrs = context_->GetAttrs();
        if (attrs == nullptr)
        {
            retCounts_ = DEFAULT_RETURN_COUNTS;
            OPS_LOG_I("GetAttrs", "attrs is nullptr, use default return_count=%d", retCounts_);
            return true;
        }

        retCounts_ = GetOptionalAttr<bool>(attrs, RETURN_COUNTS_IDX, DEFAULT_RETURN_COUNTS);
        OPS_LOG_I("GetAttrs", "return_count=%d", retCounts_);
        return true;
    }

    bool RunLengthEncodeTilingHelper::GetShapeInfo()
    {
        auto *inputDesc = context_->GetInputDesc(0);
        OPS_LOG_E_IF_NULL(context_->GetNodeName(), inputDesc, return false);
        auto *inputShape = context_->GetInputShape(0);
        OPS_LOG_E_IF_NULL(context_->GetNodeName(), inputShape, return false);

        dataTypeX_ = inputDesc->GetDataType();
        OPS_CHECK((dataTypeX_ != ge::DT_INT32) && (dataTypeX_ != ge::DT_INT64),
                  OPS_LOG_E(context_->GetNodeName(), "Input sorted_indices dtype must be int32/int64, but got %d",
                            static_cast<int32_t>(dataTypeX_)),
                  return false);
        dtSizeX_ = static_cast<int64_t>(ge::GetSizeByDataType(dataTypeX_));
        OPS_CHECK(
            dtSizeX_ <= 0,
            OPS_LOG_E(context_->GetNodeName(), "Unsupported sorted_indices dtype: %d", static_cast<int32_t>(dataTypeX_)),
            return false);

        totalSize_ = inputShape->GetStorageShape().GetShapeSize();
        OPS_CHECK(totalSize_ > MAX_SUPPORTED_TOTAL_SIZE,
                  OPS_LOG_E(context_->GetNodeName(), "Input totalSize=%ld exceeds max supported length=%ld.", totalSize_,
                            MAX_SUPPORTED_TOTAL_SIZE),
                  return false);

        OPS_LOG_I("GetShapeInfo", "dataTypeX=%d, dtSizeX=%ld, totalSize=%ld", static_cast<int32_t>(dataTypeX_), dtSizeX_,
                  totalSize_);
        return true;
    }

    bool RunLengthEncodeTilingHelper::DoBlockTiling()
    {
        // BlockTiling：统一按 1024B 块切分，避免“核数超限/未超限”两套分支逻辑。
        // 先把每核长度对齐到 1024B 对应的元素粒度，再反推实际使用核数，确保尾核长度始终有效。
        int64_t maxSingleCoreElements = MAX_BYTES_SINGLE_CORE / dtSizeX_;
        OPS_CHECK(maxSingleCoreElements <= 0,
                  OPS_LOG_E(context_->GetNodeName(), "maxSingleCoreElements invalid: %ld", maxSingleCoreElements),
                  return false);

        if (totalSize_ <= 0)
        {
            useCoreNums_ = 1;
            tileLengthPerCore_ = maxSingleCoreElements;
            tileLengthTailCore_ = 0;
            OPS_LOG_I("DoBlockTiling", "empty input, keep default split");
            return true;
        }

        const int64_t estimatedCores = CeilDivInt<int64_t>(totalSize_, maxSingleCoreElements);
        const int64_t targetCores = (estimatedCores > static_cast<int64_t>(aivCoreNum_)) ? static_cast<int64_t>(aivCoreNum_)
                                                                                         : estimatedCores;

        const int64_t avgLength = CeilDivInt<int64_t>(totalSize_, targetCores);
        tileLengthPerCore_ = CeilAlignInt<int64_t>(avgLength, maxSingleCoreElements);
        useCoreNums_ = CeilDivInt<int64_t>(totalSize_, tileLengthPerCore_);

        // 尾核承接剩余元素，保证所有分块长度之和等于 totalSize，且 totalSize_ > 0 时尾核长度 > 0。
        tileLengthTailCore_ = totalSize_ - (useCoreNums_ - 1) * tileLengthPerCore_;
        OPS_CHECK(tileLengthTailCore_ <= 0,
                  OPS_LOG_E(context_->GetNodeName(), "tileLengthTailCore invalid: %ld", tileLengthTailCore_),
                  return false);

        OPS_LOG_I("DoBlockTiling", "useCoreNums=%ld, tileLengthPerCore=%ld, tileLengthTailCore=%ld", useCoreNums_,
                  tileLengthPerCore_, tileLengthTailCore_);
        return true;
    }

    bool RunLengthEncodeTilingHelper::DoUbTiling()
    {
        // UbTiling：先预留固定 UB 区（跨核收集计数、前核尾索引、shape/length 缓冲）。
        collectingCntBufSize_ = useCoreNums_ * sizeof(int64_t);
        prevIdxBufSize_ = static_cast<int64_t>(blockSize_);
        shapeBufSize_ =
            CeilAlignInt(static_cast<int64_t>(SHAPE_LEN * sizeof(uint64_t)), static_cast<int64_t>(blockSize_));
        const int64_t lengthBufSize = static_cast<int64_t>(blockSize_);
        const int64_t tempUbSize = collectingCntBufSize_ + prevIdxBufSize_ + shapeBufSize_ + lengthBufSize;
        OPS_CHECK(
            tempUbSize >= static_cast<int64_t>(ubSize_),
            OPS_LOG_E(context_->GetNodeName(), "fixed UB buffers overflow UB: tempUbSize=%ld, ubSize=%ld", tempUbSize,
                      static_cast<int64_t>(ubSize_)),
            return false);

        constexpr int64_t countBytes = static_cast<int64_t>(sizeof(int32_t));
        // 每个 tile 的字节预算：value 输入 + value 输出 + idx/count 输入输出（均按 int32 count 路径）。
        const int64_t byteSize = dtSizeX_ + dtSizeX_ + countBytes + countBytes;

        // UB 可用空间除以单元素预算，得到 tile 元素数上限。
        const int64_t tileLength = (static_cast<int64_t>(ubSize_) - tempUbSize) / byteSize;
        valueQueueSize_ = (tileLength * dtSizeX_) / static_cast<int64_t>(blockSize_) * static_cast<int64_t>(blockSize_);

        int64_t alignValueLength = valueQueueSize_ / dtSizeX_;
        countQueueSize_ = (tileLength * countBytes) / static_cast<int64_t>(blockSize_) * static_cast<int64_t>(blockSize_);
        idxCopyInQueueSize_ =
            (tileLength * countBytes) / static_cast<int64_t>(blockSize_) * static_cast<int64_t>(blockSize_);
        int64_t alignCountLength = countQueueSize_ / countBytes;

        // Kernel 侧做跨 tile 比较需要额外保留 1 个元素，因此可用 tile 长度必须 > 1。
        adjUbTileLength_ = (alignValueLength < alignCountLength) ? alignValueLength : alignCountLength;
        OPS_CHECK(
            (adjUbTileLength_ <= 1),
            OPS_LOG_E(context_->GetNodeName(), "adjUbTileLength invalid: %ld (alignValueLength=%ld, alignCountLength=%ld)",
                      adjUbTileLength_, alignValueLength, alignCountLength),
            return false);

        // 二次校验：按 kernel 实际 InitBuffer 的总分配量检查 UB 上限，避免越界。
        const int64_t totalUbAlloc =
            valueQueueSize_ * 2 + countQueueSize_ + idxCopyInQueueSize_ + collectingCntBufSize_ + prevIdxBufSize_ +
            shapeBufSize_ + lengthBufSize;
        OPS_CHECK(totalUbAlloc > static_cast<int64_t>(ubSize_),
                  OPS_LOG_E(context_->GetNodeName(), "UB overflow after tiling: totalUbAlloc=%ld, ubSize=%ld",
                            totalUbAlloc, static_cast<int64_t>(ubSize_)),
                  return false);
        return true;
    }

    bool RunLengthEncodeTilingHelper::ComputeWorkspaces()
    {
        // workspace 划分：
        // 1) idxWorkSpace：多核收集阶段暂存 unique 的全局索引（int32）。
        // 2) valueWorkSpace：多核收集阶段暂存 unique 值。
        // 3) coreWorkSpace：每核收集计数区（按页对齐偏移寻址）。
        idxWorkSpace_ = static_cast<int64_t>(sizeof(int32_t)) * totalSize_;
        valueWorkSpace_ = dtSizeX_ * totalSize_;
        coreWorkSpace_ = useCoreNums_ * MAGIC_GM_PAGE_SIZE;

        OPS_LOG_I("ComputeWorkspaces", "idxWorkSpace=%ld, valueWorkSpace=%ld, coreWorkSpace=%ld", idxWorkSpace_,
                  valueWorkSpace_, coreWorkSpace_);
        return true;
    }

    void RunLengthEncodeTilingHelper::SetTilingDataAndTilingKeyAndWorkSpace(RunLengthEncodeTilingData *tiling)
    {
        tiling->set_totalSize(totalSize_);
        tiling->set_useCoreNums(useCoreNums_);
        tiling->set_tileLengthPerCore(tileLengthPerCore_);
        tiling->set_tileLengthTailCore(tileLengthTailCore_);
        tiling->set_adjUbTileLength(adjUbTileLength_);
        tiling->set_valueQueueSize(valueQueueSize_);
        tiling->set_countQueueSize(countQueueSize_);
        tiling->set_idxCopyInQueueSize(idxCopyInQueueSize_);
        tiling->set_collectingCntBufSize(collectingCntBufSize_);
        tiling->set_prevIdxBufSize(prevIdxBufSize_);
        tiling->set_shapeBufSize(shapeBufSize_);

        size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
        OPS_LOG_E_IF_NULL(context_->GetNodeName(), currentWorkspace, return);

        useCoreNums_ = (useCoreNums_ > 0) ? useCoreNums_ : 1;
        // 多核 workspace 布局：
        // [runtime workspace][per-core collect count][value workspace][idx workspace]
        // 单核路径不需要这些中间缓冲，保持最小 workspace=1。
        currentWorkspace[0] =
            (useCoreNums_ == 1) ? 1 : (sysWorkspaceSize_ + idxWorkSpace_ + valueWorkSpace_ + coreWorkSpace_);

        // tiling key 映射：
        // 10/11 -> 单核（不输出/输出 count），20/21 -> 多核（不输出/输出 count），666 -> 空输入。
        uint64_t tilingKey = (useCoreNums_ == 1) ? TILING_KEY_SINGLE_CORE : TILING_KEY_MULTI_CORE;
        if (retCounts_)
        {
            tilingKey += 1;
        }
        if (totalSize_ <= 0)
        {
            tilingKey = TILING_KEY_EMPTY;
        }

        context_->SetTilingKey(tilingKey);
        context_->SetBlockDim(useCoreNums_);
        context_->SetScheduleMode(1);

        OPS_LOG_E_IF_NULL(context_->GetNodeName(), context_->GetRawTilingData(), return);
        tiling->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tiling->GetDataSize());

        OPS_LOG_I("SetTilingData", "tilingKey=%lu, useCoreNums=%ld", tilingKey, useCoreNums_);
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

        RunLengthEncodeTilingData tiling;
        RunLengthEncodeTilingHelper helper(context);

        OPS_CHECK(!helper.DoTiling(), OPS_LOG_E(context->GetNodeName(), "DoTiling failed."), return ge::GRAPH_FAILED);
        helper.SetTilingDataAndTilingKeyAndWorkSpace(&tiling);
        return ge::GRAPH_SUCCESS;
    }

} // namespace optiling

namespace ge
{

    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

        const gert::Shape *xShape = context->GetInputShape(0);
        OPS_LOG_E_IF_NULL("xShape", xShape, return ge::GRAPH_FAILED);
        gert::Shape *outShape = context->GetOutputShape(OUT_IDX);
        OPS_LOG_E_IF_NULL("outShape", outShape, return ge::GRAPH_FAILED);
        gert::Shape *countsShape = context->GetOutputShape(COUNTS_IDX);
        OPS_LOG_E_IF_NULL("countsShape", countsShape, return ge::GRAPH_FAILED);
        gert::Shape *lengthShape = context->GetOutputShape(LENGTH_IDX);
        OPS_LOG_E_IF_NULL("lengthShape", lengthShape, return ge::GRAPH_FAILED);

        int64_t xSize = xShape->GetShapeSize();
        outShape->SetDimNum(1);
        countsShape->SetDimNum(1);
        lengthShape->SetDimNum(1);
        lengthShape->SetDim(0, 1);

        if (xSize == 0)
        {
            outShape->SetDim(0, 0);
            countsShape->SetDim(0, 0);
        }
        else
        {
            outShape->SetDim(0, UNKNOWN_SHAPE_DIM);
            countsShape->SetDim(0, UNKNOWN_SHAPE_DIM);
        }

        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

        const ge::DataType indicesDtype = context->GetInputDataType(0);

        context->SetOutputDataType(OUT_IDX, indicesDtype);
        context->SetOutputDataType(COUNTS_IDX, ge::DT_INT32);
        context->SetOutputDataType(LENGTH_IDX, ge::DT_INT32);
        return ge::GRAPH_SUCCESS;
    }

} // namespace ge

namespace ops
{

    class RunLengthEncode : public OpDef
    {
    public:
        explicit RunLengthEncode(const char *name) : OpDef(name)
        {
            static const std::vector<ge::DataType> dataTypeSortedIndices = {ge::DT_INT32, ge::DT_INT64};
            static const std::vector<ge::DataType> dataTypeUniqueIndices = {ge::DT_INT32, ge::DT_INT64};
            static const std::vector<ge::DataType> dataTypeCount = {ge::DT_INT32, ge::DT_INT32};
            static const std::vector<ge::DataType> dataTypeLength = {ge::DT_INT32, ge::DT_INT32};
            static const std::vector<ge::Format> dataFormat = {ge::FORMAT_ND, ge::FORMAT_ND};

            this->Input("sorted_indices")
                .ParamType(REQUIRED)
                .DataType(dataTypeSortedIndices)
                .Format(dataFormat)
                .UnknownShapeFormat(dataFormat);

            this->Output("unique_indices")
                .OutputShapeDependOnCompute()
                .ParamType(REQUIRED)
                .DataType(dataTypeUniqueIndices)
                .Format(dataFormat)
                .UnknownShapeFormat(dataFormat);

            this->Output("unique_indices_count")
                .OutputShapeDependOnCompute()
                .ParamType(REQUIRED)
                .DataType(dataTypeCount)
                .Format(dataFormat)
                .UnknownShapeFormat(dataFormat);

            this->Output("unique_indices_length")
                .ParamType(REQUIRED)
                .DataType(dataTypeLength)
                .Format(dataFormat)
                .UnknownShapeFormat(dataFormat);

            this->Attr("return_count").AttrType(OPTIONAL).Bool(false);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

            this->AICore().SetTiling(optiling::TilingFunc);

            OpAICoreConfig aicoreConfig;
            aicoreConfig.DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(false)
                .DynamicRankSupportFlag(false)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false)
                .ExtendCfgInfo("opFile.value", "run_length_encode");
            this->AICore().AddConfig("ascend950", aicoreConfig);
        }
    };

    OP_ADD(RunLengthEncode);

} // namespace ops
