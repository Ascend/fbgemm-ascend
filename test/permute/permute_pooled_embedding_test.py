#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import inspect
import sys
import unittest

import hypothesis.strategies as st
import torch
import torch.nn as nn
from fbgemm_gpu.permute_pooled_embedding_modules import PermutePooledEmbeddings
from hypothesis import given, HealthCheck, settings

from common import Net, cpu_and_maybe_npu


suppressed_list: list[HealthCheck] = (
    [HealthCheck.not_a_test_method]
    if getattr(HealthCheck, "not_a_test_method", False)
    else []
) + (
    [HealthCheck.differing_executors]
    if getattr(HealthCheck, "differing_executors", False)
    else []
)

INTERN_MODULE = "fbgemm_gpu.permute_pooled_embedding_modules"
FIXED_EXTERN_API = {
    "PermutePooledEmbeddings": {
        "__init__": ["self", "embs_dims", "permute", "device"],
        "__call__": ["self", "pooled_embs"],
    },
}


class PooledEmbeddingModulesTest(unittest.TestCase):
    @settings(deadline=None, suppress_health_check=suppressed_list)
    @given(device_type=cpu_and_maybe_npu())
    def setUp(self, device_type: torch.device) -> None:
        self.device = device_type

    @settings(deadline=None)
    @given(fwd_only=st.booleans())
    def test_permutation(self, fwd_only: bool) -> None:
        net = Net(fwd_only=fwd_only).to(self.device)

        input = torch.Tensor([range(10)]).to(self.device)
        self.assertEqual(
            net.permute_pooled_embeddings(input).view(10).tolist(),
            [6, 7, 8, 9, 0, 1, 5, 2, 3, 4],
        )

    def test_permutation_autograd(self) -> None:
        net = Net().to(self.device)

        input = torch.randn(2, 1).to(self.device)
        input_sum = input.sum().item()

        output = net(input)
        output.sum().backward()

        # check grads for fc1 when permuted, equals to fc2 weights times input_sum
        permute_res = net.permute_pooled_embeddings(net.fc1.weight.grad.view(1, 10))
        permute_ref = input_sum * net.fc2.weight
        torch.testing.assert_close(permute_res, permute_ref, rtol=1e-03, atol=1e-03)

    def test_compatibility(self) -> None:
        members = inspect.getmembers(sys.modules[INTERN_MODULE])
        for name, clazz in members:
            if getattr(clazz, "__module__", None) != INTERN_MODULE:
                continue

            self.assertIn(name, FIXED_EXTERN_API.keys())

            for fn, fixed_params in FIXED_EXTERN_API[name].items():
                current_params = inspect.getfullargspec(getattr(clazz, fn)).args
                self.assertEqual(
                    fixed_params,
                    current_params,
                    msg=f"\nForward incompatible change in {name} : {fn}\n",
                )

    def test_pooled_table_batched_embedding(self) -> None:
        num_emb_bags = 5
        num_embeddings = 10
        embedding_dims = [1, 2, 3, 4, 5]
        emb_weight_range = 1
        embedding_bags = [
            nn.EmbeddingBag(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dims[i],
                mode="sum",
                sparse=True,
            )
            for i in range(num_emb_bags)
        ]
        for emb_bag in embedding_bags:
            torch.nn.init.uniform_(
                emb_bag.weight,
                -emb_weight_range,
                emb_weight_range,
            )
        indices = [[0], [1, 2], [0, 1, 2], [3, 6], [8]]
        indices = [torch.tensor(i).view(-1, len(i)) for i in indices]
        pooled_embs = [emb_bag(indices[i]) for i, emb_bag in enumerate(embedding_bags)]

        cat_pooled_embs = torch.cat(pooled_embs, dim=1)

        permute_order = [2, 1, 3, 0, 4]

        permute_pooled_embeddings = PermutePooledEmbeddings(
            embedding_dims,
            permute_order,
            device=self.device,
        )
        permuted_pooled_emb = permute_pooled_embeddings(cat_pooled_embs.to(self.device))

        ref_permuted_pooled_emb = [pooled_embs[i] for i in permute_order]
        ref_permuted_pooled_emb = torch.cat(ref_permuted_pooled_emb, dim=1)

        assert torch.allclose(
            ref_permuted_pooled_emb.to(self.device), permuted_pooled_emb
        )

    def test_permutation_autograd_meta(self) -> None:
        """
        Test that permute_pooled_embeddings_autograd works with meta tensor and
        dynamo export mode
        """
        input = torch.randn(2, 1)
        net = Net()

        output_cpu = net(input)
        output_meta = net.to("meta")(input.to("meta"))

        assert output_meta.shape == output_cpu.shape
        assert input.shape == output_meta.shape


if __name__ == "__main__":
    unittest.main()
