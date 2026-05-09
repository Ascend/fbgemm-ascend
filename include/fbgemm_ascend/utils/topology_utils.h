/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef TOPOLOGY_UTILS_H
#define TOPOLOGY_UTILS_H

#include <cstdint>
#include <functional>

using Node = int64_t;
using Links = int64_t;
template <typename T>
using AdjacencyMatrix = std::function<T(Node, Node)>;

namespace fbgemm_ascend {
    AdjacencyMatrix<Links> getAscendLinkMatrix();
}
#endif