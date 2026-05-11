/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "fbgemm_ascend/utils/topology_utils.h"
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "common/pytorch_npu_helper.hpp"
#include <iostream>
#include "fbgemm_ascend/utils/topology_utils.h"

namespace fbgemm_ascend {
constexpr int32_t SUPPORTED_ACCESSPEER = 1;

AdjacencyMatrix<Links> getAscendLinkMatrix()
{
    // dev物理ID与逻辑ID一一对应， aclrtDeviceGetDeviceCount
    auto worldSize = c10_npu::device_count();
    std::vector<Links> links(worldSize * worldSize);
    for (const auto i : c10::irange(worldSize)) {
        auto srcDev = i;  // 根据物理ID映射逻辑ID
        for (const auto j : c10::irange(worldSize)) {
            if (i == j) {
                links[i * worldSize + j] += 1;
                continue;
            }
            auto dstDev = j;
            int32_t canAccessPeer = 0;
            
            aclError err = aclrtDeviceCanAccessPeer(&canAccessPeer, srcDev, dstDev);
            TORCH_CHECK(err == ACL_SUCCESS, "aclrtDeviceCanAccessPeer srcDev: ", srcDev, " dstDev: ", dstDev,
                        " failed, ret: ", err);
            if (canAccessPeer == SUPPORTED_ACCESSPEER) {
                links[i * worldSize + j] += 1;
            }
        }
    }
    return [=](Node i, Node j) {
        TORCH_CHECK_LT(i, worldSize);
        TORCH_CHECK_LT(j, worldSize);
        return links[i * worldSize + j];
    };
}
}  // namespace fbgemm_ascend
