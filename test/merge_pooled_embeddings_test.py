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
import unittest

import fbgemm_gpu  # noqa:F401
import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given, settings, Verbosity

import torch_npu  # noqa:F401
import fbgemm_ascend  # noqa:F401

from test_utils import npu_unavailable

typed_npu_unavailable: tuple[bool, str] = npu_unavailable
bytes2bit = 8


def make_pitched_tensor(
    height: int,
    width: int,
    dtype: torch.dtype,
    # pyre-fixme[2]: Parameter must be annotated.
    device,
    alignment: int = 256,
) -> torch.Tensor:
    elem_size = (
        torch.finfo(dtype).bits // bytes2bit if dtype.is_floating_point else torch.iinfo(dtype).bits // bytes2bit
    )
    width_bytes = width * elem_size
    pitch_bytes = int(np.ceil(width_bytes / alignment) * alignment)
    pitch_elems = pitch_bytes // elem_size
    storage = torch.randn((height, pitch_elems), dtype=dtype, device=device)
    view = storage[:, :width]  # logical shape
    return view.contiguous() if alignment == 0 else view  # return pitched view


@unittest.skipIf(*typed_npu_unavailable)
class MergePooledEmbeddingsTest(unittest.TestCase):
    @given(
        num_inputs=st.integers(min_value=1, max_value=10),
        num_npus=st.integers(min_value=1, max_value=torch.npu.device_count()),
        r=st.randoms(use_true_random=False),
        use_pitched=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=40, deadline=None)
    def test_all_to_one_device(
        self,
        num_inputs: int,
        num_npus: int,
        # pyre-fixme[2]: Parameter must be annotated.
        r,
        use_pitched: bool,
    ) -> None:
        dst_device = torch.device(f"npu:{r.randint(0, num_npus - 1)}")
        with torch.npu.device(dst_device):
            if use_pitched:
                inputs = [make_pitched_tensor(10, 20, torch.float32, "cpu", alignment=256) for _ in range(num_inputs)]
            else:
                inputs = [torch.randn(10, 20) for _ in range(num_inputs)]

            npu_inputs = [input.to(f"npu:{i % num_npus}") for i, input in enumerate(inputs)]
            npu_outputs = torch.ops.fbgemm.all_to_one_device(npu_inputs, dst_device)
            for i, o in zip(inputs, npu_outputs):
                self.assertEqual(o.device, dst_device)
                torch.testing.assert_close(o.cpu(), i)

    @given(
        num_ads=st.integers(min_value=1, max_value=10),
        embedding_dimension=st.integers(min_value=1, max_value=32),
        ads_tables=st.integers(min_value=1, max_value=32),
        num_npus=st.integers(min_value=1, max_value=torch.npu.device_count()),
        non_default_stream=st.booleans(),
        r=st.randoms(use_true_random=False),
        dim=st.integers(min_value=0, max_value=1),
        source_from_same_device=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=40, deadline=None)
    def test_merge(
        self,
        num_ads: int,
        embedding_dimension: int,
        ads_tables: int,
        num_npus: int,
        non_default_stream: bool,
        # pyre-fixme[2]: Parameter must be annotated.
        r,
        dim: int,
        source_from_same_device: bool,
    ) -> None:
        dst_device = torch.device(f"npu:{r.randint(0, num_npus - 1)}")
        torch.npu.set_device(dst_device)
        ad_ds = [embedding_dimension * ads_tables for _ in range(num_npus)]
        batch_indices = torch.zeros(num_ads).long().to(dst_device)
        pooled_ad_embeddings = [
            (
                torch.randn(num_ads, ad_d, dtype=torch.float16, device=dst_device)
                if source_from_same_device
                else torch.randn(num_ads, ad_d, dtype=torch.float16, device=torch.device(f"npu:{i}"))
            )
            for i, ad_d in enumerate(ad_ds)
        ]
        r.shuffle(pooled_ad_embeddings)

        streams = [torch.npu.Stream(device=i) for i in range(num_npus)]
        import contextlib

        uncat_size = batch_indices.size(0) if dim == 1 else ad_ds[0]

        with contextlib.ExitStack() as stack:
            if non_default_stream:
                for stream in streams:
                    stack.enter_context(torch.npu.stream(stream))
            output = torch.ops.fbgemm.merge_pooled_embeddings(
                pooled_ad_embeddings, uncat_size, batch_indices.device, dim
            )

        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def ref(pooled_ad_embeddings, batch_indices):
            return torch.cat([p.cpu() for p in pooled_ad_embeddings], dim=dim)

        output_ref = ref(pooled_ad_embeddings, batch_indices)
        output_cpu = torch.ops.fbgemm.merge_pooled_embeddings(
            [pe.cpu() for pe in pooled_ad_embeddings],
            uncat_size,
            batch_indices.cpu().device,
            dim,
        )
        self.assertEqual(output.device, torch.device(dst_device))
        torch.testing.assert_close(output_ref, output.cpu())
        torch.testing.assert_close(output_ref, output_cpu)

    def test_merge_pooled_embeddings_npu_to_cpu(self) -> None:
        """Test merge_pooled_embeddings from NPU to CPU"""
        dst_device = torch.device("cpu")
        inputs = [torch.randn(10, 20) for _ in range(4)]
        num_npus = torch.npu.device_count()
        npu_inputs = [input.to(f"npu:{i % num_npus}") for i, input in enumerate(inputs)]
        uncat_size = inputs[0].size(1)
        output = torch.ops.fbgemm.merge_pooled_embeddings(npu_inputs, uncat_size, dst_device, 0)
        ref_output = torch.ops.fbgemm.merge_pooled_embeddings(inputs, uncat_size, dst_device, 0)
        torch.testing.assert_close(output, ref_output)

    def test_merge_pooled_embeddings_cpu_with_different_target_device(self) -> None:
        uncat_size = 2
        pooled_embeddings = [torch.ones(uncat_size, 4), torch.ones(uncat_size, 8)]
        output_meta = torch.ops.fbgemm.merge_pooled_embeddings(
            pooled_embeddings,
            uncat_size,
            torch.device("meta"),
            1,
        )
        self.assertFalse(output_meta.is_cpu)
        self.assertTrue(output_meta.is_meta)

    def test_merge_pooled_embeddings_meta(self) -> None:
        """Test that merge_pooled_embeddings works with meta tensor"""
        uncat_size = 2
        cat_dim = 1
        pooled_embeddings = [torch.ones(uncat_size, 4), torch.ones(uncat_size, 8)]

        output_cpu = torch.ops.fbgemm.merge_pooled_embeddings(
            pooled_embeddings, uncat_size, torch.device("cpu"), cat_dim
        )
        output_meta = torch.ops.fbgemm.merge_pooled_embeddings(
            [p.to("meta") for p in pooled_embeddings], uncat_size, torch.device("meta"), cat_dim
        )

        self.assertFalse(output_meta.is_cpu)
        self.assertTrue(output_meta.is_meta)
        self.assertEqual(output_meta.shape, output_cpu.shape)
