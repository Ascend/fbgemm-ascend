/**
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "../common/pytorch_npu_helper.hpp"
#include "../common/common_utils.h"
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "fbgemm_ascend/utils/topology_utils.h"

using namespace at;
using namespace std;
namespace {
struct DirectConnectedPeer {
    int64_t numPeerLinks;
    int64_t peerId;
    int32_t peerTransfers;
};

struct TwoHopTransferContainer {
    Tensor intermediateTensor;
    uint64_t outputIdx;
};

static std::vector<int8_t> p2pAccessEnabled_;
static int64_t numNpus_ = -1;

AdjacencyMatrix<Node> getIntermediateNode(const AdjacencyMatrix<Links>& links)
{
    const auto worldSize = c10_npu::device_count();
    std::vector<Node> linkVec(static_cast<size_t>(worldSize * worldSize));
    for (const auto i : c10::irange(worldSize)) {
        for (const auto j : c10::irange(worldSize)) {
            linkVec[i * worldSize + j] = links(i, j);
        }
    }
    auto linkTensor = at::from_blob(linkVec.data(), {worldSize, worldSize}, at::TensorOptions().dtype(at::kLong));
    std::vector<Node> assignments(static_cast<size_t>(worldSize * worldSize), -1);
    for (const auto dstRankId : c10::irange(worldSize)) {
        std::vector<int> nonDirectSrcIds;
        nonDirectSrcIds.reserve(worldSize);
        std::vector<DirectConnectedPeer> directConnectedPeers;
        directConnectedPeers.reserve(worldSize);
        for (const auto srcRankId : c10::irange(worldSize)) {
            if (dstRankId == srcRankId) {
                continue;
            }

            const auto numPeerLinks = links(dstRankId, srcRankId);
            if (numPeerLinks > 0) {
                directConnectedPeers.push_back({.numPeerLinks = numPeerLinks, .peerId = srcRankId, .peerTransfers = 1});
            } else {
                nonDirectSrcIds.push_back(srcRankId);
            }
        }

        for (const auto i : c10::irange(nonDirectSrcIds.size())) {
            std::sort(directConnectedPeers.begin(), directConnectedPeers.end(), [](const auto& a, const auto& b) {
                if (a.numPeerLinks > b.numPeerLinks) {
                    return true;
                } else if (a.numPeerLinks == b.numPeerLinks) {
                    return a.peerTransfers < b.peerTransfers;
                } else {
                    return false;
                }
            });
            const auto nonDirectSrcId = nonDirectSrcIds.at(i);
            for (auto& j : directConnectedPeers) {
                const auto potentialHopId = j.peerId;
                const auto potentialHopPeerLinks = links(potentialHopId, nonDirectSrcId);
                if (potentialHopPeerLinks > 0) {
                    assignments[dstRankId * worldSize + nonDirectSrcId] = potentialHopId;
                    j.peerTransfers += 1;
                    break;
                }
            }
        }
    }
    if (std::any_of(assignments.begin(), assignments.end(), [](Node n) { return n != -1; })) {
        auto tensor = at::from_blob(assignments.data(), {worldSize, worldSize}, at::TensorOptions().dtype(at::kLong));
        return [=](Node src, Node dst) {
            return assignments[dst * worldSize + src];
        };
    } else {
        return [](Node, Node) {
            return -1;
        };
    }
}

void initP2PAccessCache()
{
    if (numNpus_ != -1) {
        return;
    }
    numNpus_ = c10_npu::device_count();
    p2pAccessEnabled_.clear();
    p2pAccessEnabled_.resize(numNpus_ * numNpus_, -1);
    for (const auto i : c10::irange(numNpus_)) {
        p2pAccessEnabled_[i * numNpus_ + i] = 1;
    }
}

bool GetP2PAccess(int dev, int devToAccess)
{
    initP2PAccessCache();
    TORCH_CHECK(dev >= 0 && dev < numNpus_, dev, " is not a device");
    TORCH_CHECK(devToAccess >= 0 && devToAccess < numNpus_, devToAccess, " is not a device");
    TORCH_INTERNAL_ASSERT(numNpus_ >= 0, "no device is available");
    auto& cache = p2pAccessEnabled_[dev * numNpus_ + devToAccess];
    if (cache != -1) {
        return cache;
    }

    c10_npu::NPUGuard guard(dev);
    int32_t access = 0;
    aclError err = aclrtDeviceCanAccessPeer(&access, dev, devToAccess);
    TORCH_CHECK(err == ACL_SUCCESS, "aclrtDeviceCanAccessPeer failed, ret: ", err);
    if (access) {
        c10_npu::set_device(dev);
        err = aclrtDeviceEnablePeerAccess(devToAccess, 0);
        if (err == ACL_SUCCESS) {
            cache = 1;
        } else {
            cache = 0;
        }
    }
    return cache;
}

void InitP2PAccess(const std::vector<Tensor>& inputTensors, at::Device targetDevice)
{
    static std::once_flag flag;
    // 只enableP2P src->targetDevice
    std::call_once(flag, [&]() {
        for (const auto i : c10::irange(c10_npu::device_count())) {
            for (const auto j : c10::irange(c10_npu::device_count())) {
                if (i != j) {
                    TORCH_INTERNAL_ASSERT(GetP2PAccess(i, j), "Failed to init p2p access for node ", i, ",", j);
                }
            }
        }
    });
}

void allToOneTargetCpu(const std::vector<Tensor>& inputTensors, std::vector<Tensor>& outputTensors)
{
    TORCH_CHECK(inputTensors.size() == outputTensors.size());
    for (size_t i = 0; i < inputTensors.size(); i++) {
        const auto& src = inputTensors.at(i);
        auto& dst = outputTensors.at(i);
        dst.copy_(src, true);
    }
    for (const auto& t : inputTensors) {
        c10_npu::getCurrentNPUStream(t.device().index()).synchronize();
    }
}
}  // namespace

// 合并所有设备的张量到目标设备, 由于aclrtMemcpy2dAsync不支持d2d,暂用aclrtMemcpyAsync, 所以目前只支持连续内存,
// NPUEvent record block机制暂不支持，采用同步流方式实现
void allToOne(const std::vector<Tensor>& inputTensors, std::vector<Tensor>& outputTensors, at::Device targetDevice,
              bool skipIfSameDevice)
{
    if (targetDevice.is_cpu()) {
        allToOneTargetCpu(inputTensors, outputTensors);
        return;
    }

    auto numNpus = c10_npu::device_count();

    auto targetDeviceIndex = targetDevice.index();
    TORCH_CHECK(targetDeviceIndex != -1, "targetDevice.index() is -1. Please pass targetDevice with device "
                                         "index, e.g., torch.device(\"npu:0\")");
    TORCH_CHECK(targetDeviceIndex < numNpus);
    std::vector<TwoHopTransferContainer> twoHopTransfers;
    twoHopTransfers.reserve(inputTensors.size());
    std::vector<bool> isTwoHopTransfer;
    isTwoHopTransfer.reserve(inputTensors.size());

    static auto intermediateNodes = getIntermediateNode(fbgemm_ascend::getAscendLinkMatrix());
    aclError err = ACL_SUCCESS;
    for (const auto i : c10::irange(inputTensors.size())) {
        const auto& src = inputTensors.at(i);
        auto srcDeviceId = src.get_device();
        auto intermediateNode = intermediateNodes(srcDeviceId, targetDeviceIndex);
        if (intermediateNode != -1) {
            // 创建中间tensor，后续从中间tensor再复制到目标设备
            Tensor dst = at::empty_like(src, src.options().device(at::Device(at::kPrivateUse1, intermediateNode)));
            auto copyStream = c10_npu::getCurrentNPUStream(srcDeviceId);
            // NPUEvent record block机制暂不支持，采用同步流方式实现
            aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(intermediateNode));
            aclrtSynchronizeStream(copyStream);
            twoHopTransfers.push_back({.intermediateTensor = dst, .outputIdx = i});
            c10_npu::set_device(srcDeviceId);
            err = aclrtMemcpyAsync(dst.mutable_data_ptr(), dst.size(0) * dst.element_size() * dst.size(1),
                                   src.const_data_ptr(), src.size(0) * src.element_size() * src.size(1),
                                   aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE, copyStream);
            TORCH_CHECK(err == ACL_SUCCESS, "aclrtMemcpyAsync failed, ret: ", err);
            isTwoHopTransfer.push_back(true);
        } else {
            isTwoHopTransfer.push_back(false);
        }
    }

    for (const auto deviceId : c10::irange(numNpus)) {
        auto srcDevice = at::Device(at::kPrivateUse1, deviceId);
        if (srcDevice == targetDevice) {
            continue;
        }

        auto copyStream = c10_npu::getCurrentNPUStream(deviceId);
        // NPUEvent record block机制暂不支持，采用同步流方式实现
        aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(targetDeviceIndex));
        aclrtSynchronizeStream(copyStream);
        for (const auto i : c10::irange(inputTensors.size())) {
            const auto metadata = isTwoHopTransfer.at(i);
            if (metadata) {
                continue;
            }

            auto& src = inputTensors[i];
            if (src.device() != srcDevice) {
                continue;
            }
            auto& dst = outputTensors[i];
            c10_npu::set_device(deviceId);
            err = aclrtMemcpyAsync(dst.mutable_data_ptr(), dst.size(0) * dst.element_size() * dst.size(1),
                                   src.const_data_ptr(), src.size(0) * src.element_size() * src.size(1),
                                   aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE, copyStream);
            TORCH_CHECK(err == ACL_SUCCESS, "dst src aclrtMemcpyAsync failed, ret: ", err);
        }
    }

    for (auto& twoHopTransfer : twoHopTransfers) {
        // 中间tensor传到targetDevice
        const auto& src = twoHopTransfer.intermediateTensor;
        const auto srcDeviceId = src.get_device();
        const auto srcDevice = at::Device(at::kPrivateUse1, srcDeviceId);
        if (srcDevice == targetDevice) {
            continue;
        }

        auto copyStream = c10_npu::getCurrentNPUStream(srcDeviceId);

        // NPUEvent record block机制暂不支持，采用同步流方式实现
        aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(targetDeviceIndex));
        aclrtSynchronizeStream(copyStream);
        const auto outputIndex = twoHopTransfer.outputIdx;
        auto& dst = outputTensors.at(outputIndex);
        c10_npu::set_device(srcDeviceId);
        err = aclrtMemcpyAsync(dst.mutable_data_ptr(), dst.size(0) * dst.element_size() * dst.size(1),
                               src.const_data_ptr(), src.size(0) * src.element_size() * src.size(1),
                               aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE, copyStream);
        TORCH_CHECK(err == ACL_SUCCESS, "twoHopTransfer aclrtMemcpyAsync failed, ret: ", err);
    }

    if (!skipIfSameDevice) {
        for (const auto i : c10::irange(inputTensors.size())) {
            auto& src = inputTensors[i];
            if (src.device() == targetDevice) {
                auto& dst = outputTensors[i];
                auto copyStream = c10_npu::getCurrentNPUStream(targetDeviceIndex);

                auto err = aclrtMemcpyAsync(dst.mutable_data_ptr(), dst.size(0) * dst.element_size() * dst.size(1),
                                            src.const_data_ptr(), src.size(0) * src.element_size() * src.size(1),
                                            aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE, copyStream);
                TORCH_CHECK(err == ACL_SUCCESS, "skipifSameDevice aclrtMemcpyAsync failed, ret: ", err);
            }
        }
    }

    for (const auto deviceId : c10::irange(numNpus)) {
        if (deviceId != targetDeviceIndex) {
            // NPUEvent record block机制暂不支持，采用同步流方式实现
            auto srcDevice = at::Device(at::kPrivateUse1, deviceId);
            auto copyStream = c10_npu::getCurrentNPUStream(deviceId);
            aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(targetDeviceIndex));
            aclrtSynchronizeStream(copyStream);
        }
    }
}

std::vector<Tensor> allToOneDeviceImplNpu(std::vector<Tensor> inputTensors, at::Device targetDevice)
{
    if (!targetDevice.is_cpu()) {
        InitP2PAccess(inputTensors, targetDevice);
        c10_npu::NPUGuard guard(targetDevice);
    }

    std::vector<Tensor> outputTensors;
    outputTensors.reserve(inputTensors.size());
    for (const auto& tensor : inputTensors) {
        TORCH_CHECK(
            tensor.device().type() == at::kPrivateUse1,
            "input tensor must be on NPU device, but got device type: ", static_cast<int>(tensor.device().type()));
        outputTensors.push_back(tensor.device() != targetDevice
                                    ? at::empty(tensor.sizes(), tensor.options().device(targetDevice))
                                    : tensor);
    }
    allToOne(inputTensors, outputTensors, targetDevice, true);
    return outputTensors;
}

namespace {

// 为merge_pooled_embeddings实现allToOne接口，采用copy_替代aclrtMemcpyAsync，因为outputTensors是非连续
void allToOneMerge(const std::vector<Tensor>& inputTensors, std::vector<Tensor>& outputTensors, at::Device targetDevice,
                   bool skipIfSameDevice)
{
    if (targetDevice.is_cpu()) {
        allToOneTargetCpu(inputTensors, outputTensors);
        return;
    }

    auto numNpus = c10_npu::device_count();

    auto targetDeviceIndex = targetDevice.index();
    TORCH_CHECK(targetDeviceIndex != -1, "targetDevice.index() is -1. Please pass targetDevice with device "
                                         "index, e.g., torch.device(\"npu:0\")");
    TORCH_CHECK(targetDeviceIndex < numNpus);
    std::vector<TwoHopTransferContainer> twoHopTransfers;
    twoHopTransfers.reserve(inputTensors.size());
    std::vector<bool> isTwoHopTransfer;
    isTwoHopTransfer.reserve(inputTensors.size());

    static auto intermediateNodes = getIntermediateNode(fbgemm_ascend::getAscendLinkMatrix());
    aclError err = ACL_SUCCESS;
    for (const auto i : c10::irange(inputTensors.size())) {
        const auto& src = inputTensors.at(i);
        auto srcDeviceId = src.get_device();
        auto intermediateNode = intermediateNodes(srcDeviceId, targetDeviceIndex);
        if (intermediateNode != -1) {
            // 创建中间tensor，后续从中间tensor再复制到目标设备
            Tensor dst = at::empty_like(src, src.options().device(at::Device(at::kPrivateUse1, intermediateNode)));
            auto copyStream = c10_npu::getCurrentNPUStream(srcDeviceId);
            // NPUEvent record block机制暂不支持，采用同步流方式实现
            aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(intermediateNode));
            aclrtSynchronizeStream(copyStream);
            twoHopTransfers.push_back({.intermediateTensor = dst, .outputIdx = i});
            c10_npu::set_device(srcDeviceId);
            err = aclrtMemcpyAsync(dst.mutable_data_ptr(), dst.size(0) * dst.element_size() * dst.size(1),
                                   src.const_data_ptr(), src.size(0) * src.element_size() * src.size(1),
                                   aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE, copyStream);
            TORCH_CHECK(err == ACL_SUCCESS, "aclrtMemcpyAsync failed, ret: ", err);
            isTwoHopTransfer.push_back(true);
        } else {
            isTwoHopTransfer.push_back(false);
        }
    }

    for (const auto deviceId : c10::irange(numNpus)) {
        auto srcDevice = at::Device(at::kPrivateUse1, deviceId);
        if (srcDevice == targetDevice) {
            continue;
        }

        auto copyStream = c10_npu::getCurrentNPUStream(deviceId);
        // NPUEvent record block机制暂不支持，采用同步流方式实现
        aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(targetDeviceIndex));
        aclrtSynchronizeStream(copyStream);
        for (const auto i : c10::irange(inputTensors.size())) {
            const auto metadata = isTwoHopTransfer.at(i);
            if (metadata) {
                continue;
            }

            auto& src = inputTensors[i];
            if (src.device() != srcDevice) {
                continue;
            }
            auto& dst = outputTensors[i];
            // 使用 PyTorch copy_ 替代 aclrtMemcpyAsync，因为 outputTensors[i] 是 view（非连续）
            c10_npu::set_device(deviceId);
            dst.copy_(src, true);
        }
    }

    for (auto& twoHopTransfer : twoHopTransfers) {
        // 中间tensor传到targetDevice
        const auto& src = twoHopTransfer.intermediateTensor;
        const auto srcDeviceId = src.get_device();
        const auto srcDevice = at::Device(at::kPrivateUse1, srcDeviceId);
        if (srcDevice == targetDevice) {
            continue;
        }

        auto copyStream = c10_npu::getCurrentNPUStream(srcDeviceId);

        // NPUEvent record block机制暂不支持，采用同步流方式实现
        aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(targetDeviceIndex));
        aclrtSynchronizeStream(copyStream);
        const auto outputIndex = twoHopTransfer.outputIdx;
        auto& dst = outputTensors.at(outputIndex);
        // 使用 PyTorch copy_ 替代 aclrtMemcpyAsync，因为 outputTensors 是 view
        c10_npu::set_device(srcDeviceId);
        dst.copy_(src, true);
    }

    if (!skipIfSameDevice) {
        for (const auto i : c10::irange(inputTensors.size())) {
            auto& src = inputTensors[i];
            if (src.device() == targetDevice) {
                auto& dst = outputTensors[i];
                dst.copy_(src, true);
            }
        }
        c10_npu::getCurrentNPUStream(targetDeviceIndex).synchronize();
    }

    for (const auto deviceId : c10::irange(numNpus)) {
        if (deviceId != targetDeviceIndex) {
            // NPUEvent record block机制暂不支持，采用同步流方式实现
            auto srcDevice = at::Device(at::kPrivateUse1, deviceId);
            auto copyStream = c10_npu::getCurrentNPUStream(deviceId);
            aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(targetDeviceIndex));
            aclrtSynchronizeStream(copyStream);
        }
    }
}

std::vector<Tensor> allToOneMergeDeviceImplNpu(std::vector<Tensor> inputTensors, at::Device targetDevice)
{
    if (!targetDevice.is_cpu()) {
        InitP2PAccess(inputTensors, targetDevice);
        c10_npu::NPUGuard guard(targetDevice);
    }

    std::vector<Tensor> outputTensors;
    outputTensors.reserve(inputTensors.size());
    for (const auto& tensor : inputTensors) {
        TORCH_CHECK(
            tensor.device().type() == at::kPrivateUse1,
            "input tensor must be on NPU device, but got device type: ", static_cast<int>(tensor.device().type()));
        outputTensors.push_back(tensor.device() != targetDevice
                                    ? at::empty(tensor.sizes(), tensor.options().device(targetDevice))
                                    : tensor);
    }
    allToOneMerge(inputTensors, outputTensors, targetDevice, true);
    return outputTensors;
}

std::tuple<std::array<int64_t, 2>, std::vector<int64_t>, int64_t> catDim2dOutputShape(std::vector<Tensor>& tensors,
                                                                                      int64_t uncatDimSize,
                                                                                      int64_t catDim)
{
    TORCH_CHECK(!tensors.empty());

    TORCH_CHECK(catDim >= 0 && catDim <= 1);

    int64_t totalCatDim = 0;
    std::vector<int64_t> cumulativeDims;
    cumulativeDims.push_back(0);
    for (const auto& t : tensors) {
        TORCH_CHECK(t.dim() == 2);
        TORCH_CHECK(t.size(1 - catDim) == uncatDimSize, "cat dim does not match, ", "compare tensor dim = ", 1 - catDim,
                    ", expect size = ", uncatDimSize, ", but got size = ", t.size(1 - catDim), " on device ",
                    t.get_device());
        totalCatDim += t.size(catDim);
        cumulativeDims.push_back(totalCatDim);
    }

    std::array<int64_t, 2> outputShape;
    if (catDim == 0) {
        outputShape = {totalCatDim, uncatDimSize};
    } else {
        outputShape = {uncatDimSize, totalCatDim};
    }

    return std::make_tuple(outputShape, cumulativeDims, totalCatDim);
}

Tensor catDim2d(std::vector<Tensor>& tensors, int64_t uncatDimSize, at::Device outputDevice, int64_t catDim = 1)
{
    if (tensors.empty()) {
        return at::empty({0}, at::TensorOptions().device(outputDevice));
    }

    // 检查所有 tensor 是否已经在目标设备上
    bool allOnTarget = true;
    for (const auto& t : tensors) {
        if (t.device() != outputDevice) {
            allOnTarget = false;
            break;
        }
    }

    // 如果所有 tensor 已经在目标设备上，直接 cat
    if (allOnTarget) {
        return at::cat(tensors, catDim);
    }

    // NPU 目标设备：先把所有 tensor 汇聚到目标设备
    InitP2PAccess(tensors, outputDevice);
    std::vector<Tensor> gatheredTensors;
    gatheredTensors.reserve(tensors.size());
    for (const auto& t : tensors) {
        if (t.device() == outputDevice) {
            gatheredTensors.push_back(t);
        } else {
            // 使用 allToOneMergeDeviceImplNpu 将数据拷贝到目标设备
            std::vector<Tensor> singleInput = {t};
            auto result = allToOneMergeDeviceImplNpu(singleInput, outputDevice);
            gatheredTensors.push_back(result[0]);
        }
    }

    return at::cat(gatheredTensors, catDim);
}

}  // namespace

Tensor mergePooledEmbeddingsImplNpu(std::vector<Tensor> pooledEmbeddings, int64_t uncatDimSize, at::Device targetDevice,
                                    int64_t catDim = 1)
{
    TORCH_CHECK(!pooledEmbeddings.empty());

    at::Device outDevice = targetDevice;

    bool isSameDevice = true;
    for (const auto& t : pooledEmbeddings) {
        if (t.device() != targetDevice) {
            isSameDevice = false;
            break;
        }
    }
    if (isSameDevice) {
        return at::cat(pooledEmbeddings, catDim);
    }

    if (!targetDevice.is_cpu()) {
        InitP2PAccess(pooledEmbeddings, targetDevice);
        c10_npu::NPUGuard g(targetDevice);
        outDevice = targetDevice.index() == -1 ? at::Device(at::kPrivateUse1, c10_npu::current_device()) : targetDevice;
    }

    return catDim2d(pooledEmbeddings, uncatDimSize, outDevice, catDim);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("all_to_one_device", &allToOneDeviceImplNpu);
    m.impl("merge_pooled_embeddings", &mergePooledEmbeddingsImplNpu);
}
