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

#ifndef RUN_LENGTH_ENCODE_TILING_H
#define RUN_LENGTH_ENCODE_TILING_H

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"

namespace optiling
{
    BEGIN_TILING_DATA_DEF(RunLengthEncodeTilingData)
    // 输入与多核切分信息
    TILING_DATA_FIELD_DEF(int64_t, totalSize);
    TILING_DATA_FIELD_DEF(int64_t, useCoreNums);
    TILING_DATA_FIELD_DEF(int64_t, tileLengthPerCore);
    TILING_DATA_FIELD_DEF(int64_t, tileLengthTailCore);
    // UB 切分与队列容量
    TILING_DATA_FIELD_DEF(int64_t, adjUbTileLength);
    TILING_DATA_FIELD_DEF(int64_t, valueQueueSize);
    TILING_DATA_FIELD_DEF(int64_t, countQueueSize);
    TILING_DATA_FIELD_DEF(int64_t, idxCopyInQueueSize);
    // 固定 UB 缓冲（跨核计数、前核尾索引、shape）
    TILING_DATA_FIELD_DEF(int64_t, collectingCntBufSize);
    TILING_DATA_FIELD_DEF(int64_t, prevIdxBufSize);
    TILING_DATA_FIELD_DEF(int64_t, shapeBufSize);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(RunLengthEncode, RunLengthEncodeTilingData)

    class RunLengthEncodeTilingHelper
    {
    public:
        explicit RunLengthEncodeTilingHelper(gert::TilingContext *context) : context_(context) {}
        ~RunLengthEncodeTilingHelper() = default;

        bool DoTiling();
        void SetTilingDataAndTilingKeyAndWorkSpace(RunLengthEncodeTilingData *tiling);

    private:
        bool GetBaseInfo();
        bool GetPlatformInfo();
        bool GetAttrs();
        bool GetShapeInfo();
        bool DoBlockTiling();
        bool DoUbTiling();
        bool ComputeWorkspaces();

    private:
        gert::TilingContext *context_{nullptr};

        uint64_t ubSize_{1};
        uint32_t blockSize_{1};
        uint32_t aivCoreNum_{1};
        uint64_t sysWorkspaceSize_{1};

        bool retCounts_{false};

        // 输入与 dtype 信息
        ge::DataType dataTypeX_{ge::DataType::DT_INT32};
        int64_t dtSizeX_{4};
        int64_t totalSize_{1};

        // UB 相关结果
        int64_t adjUbTileLength_{1};
        int64_t valueQueueSize_{1};
        int64_t countQueueSize_{1};
        int64_t idxCopyInQueueSize_{1};
        int64_t collectingCntBufSize_{1};
        int64_t prevIdxBufSize_{1};
        int64_t shapeBufSize_{1};

        // BlockTiling 结果
        int64_t useCoreNums_{1};
        int64_t tileLengthPerCore_{1};
        int64_t tileLengthTailCore_{1};

        // workspace 大小（字节）
        int64_t idxWorkSpace_{1};
        int64_t valueWorkSpace_{1};
        int64_t coreWorkSpace_{1};
    };
} // namespace optiling

#endif // RUN_LENGTH_ENCODE_TILING_H
