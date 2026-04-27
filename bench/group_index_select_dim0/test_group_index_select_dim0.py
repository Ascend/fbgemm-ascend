#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
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

import sysconfig
import random
import pytest
import torch
import torch_npu
import numpy as np
import fbgemm_gpu
import fbgemm_ascend

from hypothesis import given, settings, Verbosity, strategies as st

torch.npu.config.allow_internal_format = False

DEVICE = "npu:0"
# 检查 NPU 是否可用
npu_available = torch.npu.is_available()
if npu_available:
    torch_npu.npu.set_device(DEVICE)

class TestGroupIndexSelectDim0:
    @given(
        num_indices=st.integers(1, 32),
        max_num_input_rows=st.integers(1, 32),
        shape=st.lists(st.integers(1, 32), min_size=1, max_size=2),
        dtype=st.sampled_from([torch.float, torch.half]),
        use_cpu=st.booleans() if npu_available else st.just(True),
        num_groups=st.integers(1, 32),
        use_var_cols=st.booleans(),
        use_var_num_input_rows=st.booleans(),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    def test_group_index_select_dim0(
        self,
        num_indices: int,
        max_num_input_rows: int,
        shape: list[int],
        dtype: torch.dtype,
        use_cpu: bool,
        num_groups: int,
        use_var_cols: bool,
        use_var_num_input_rows: bool,
    ) -> None:
        device = torch.device("cpu" if use_cpu else DEVICE)
        input_group: list[torch.Tensor] = []
        input_ref_group: list[torch.Tensor] = []
        indices_group: list[torch.Tensor] = []
        grad_group: list[torch.Tensor] = []
        for _ in range(num_groups):
            if use_var_num_input_rows:
                num_input_rows = (
                    random.randint(1, max_num_input_rows)
                    if max_num_input_rows > 1
                    else 1
                )
            else:
                num_input_rows = max_num_input_rows
            indices = torch.randint(num_input_rows, (num_indices,), device=device)
            assert indices.max() < num_input_rows

            if use_var_cols:
                var_dim = random.randint(0, len(shape) - 1)
                new_shape = random.randint(1, 32)
                shape[var_dim] = new_shape
            indices_group.append(indices)
            input = torch.rand(
                (num_input_rows,) + tuple(shape), dtype=dtype, device=device
            )
            input_ref = input.clone().detach()

            input.requires_grad = True
            input_ref.requires_grad = True

            input_group.append(input)
            input_ref_group.append(input_ref)

            grad = torch.rand((num_indices,) + tuple(shape), dtype=dtype, device=device)
            grad_group.append(grad)

        # Test forward
        output_ref_group = []
        for input, indices in zip(input_ref_group, indices_group):
            output_ref_group.append(torch.index_select(input, 0, indices))

        output_group = torch.ops.fbgemm.group_index_select_dim0(
            input_group, indices_group
        )

        # Test backward
        for out, grad in zip(output_ref_group, grad_group):
            out.backward(grad)

        cat_output = torch.concat(
            [output.flatten() for output in output_group]
        )

        cat_grad = torch.concat(
            [grad.flatten() for grad in grad_group]
        )
        cat_output.backward(cat_grad)

        def compare_tensor_groups(
            test_group: list[torch.Tensor],
            ref_group: list[torch.Tensor],
            tensor_type: str,
            tols: dict["str", float],
        ) -> None:
            passed = True
            failure_count = 0
            for i, (test, ref) in enumerate(zip(test_group, ref_group)):
                # pyre-ignore [6]
                if not torch.allclose(test, ref, **tols):
                    passed = False
                    failure_count += 1
                    print(
                        f"FAILED: group {i} {tensor_type} ({dtype}), "
                        f"input shape {input_group[i].shape}, indices "
                        f"{indices_group[i]}, test {test}, ref {ref} "
                        f"grad_group[i] {grad_group[i]}"
                    )
            assert (
                passed
            ), f"{failure_count}/{num_groups} groups of {tensor_type} failed"

        compare_tensor_groups(
            output_group, output_ref_group, "activation", {"rtol": 0, "atol": 0}
        )
        compare_tensor_groups(
            # pyre-ignore [6]
            [i.grad for i in input_group],
            # pyre-ignore [6]
            [i.grad for i in input_ref_group],
            "gradient",
            {"rtol": 1e-02, "atol": 1e-02} if dtype == torch.half else {},
        )

# 直接运行测试（方便调试）
if __name__ == "__main__":
    test = TestGroupIndexSelectDim0()
    test.test_group_index_select_dim0()