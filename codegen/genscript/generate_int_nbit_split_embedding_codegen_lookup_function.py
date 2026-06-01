#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from pathlib import Path
from typing import List

try:
    from .common import CodeTemplate
except ImportError:
    from common import CodeTemplate


SUPPORTED_STEP1_WEIGHT_TYPES = {"FP8", "FP16", "FP32", "INT8", "INT4", "INT2"}


TARGETS = [
    (
        "codegen/inference/nbit_host_op_template.cpp",
        "c310/op_host/int_nbit_split_embedding_codegen_lookup_function.cpp",
        {},
        True,
    ),
    (
        "codegen/inference/nbit_host_tiling_template.h",
        "c310/op_host/int_nbit_split_embedding_codegen_lookup_function_tiling.h",
        {},
        True,
    ),
    (
        "codegen/inference/nbit_kernel_entry_template.cpp",
        "c310/op_kernel/int_nbit_split_embedding_codegen_lookup_function.cpp",
        {},
        True,
    ),
    (
        "codegen/inference/nbit_run_template.sh",
        "c310/run.sh",
        {},
        False,
    ),
    (
        "codegen/inference/nbit_json_template.json",
        "c310/int_nbit_split_embedding_codegen_lookup_function.json",
        {},
        False,
    ),
    (
        "codegen/inference/nbit_common_template.h",
        "c310/op_kernel/common.h",
        {},
        True,
    ),
    (
        "codegen/inference/nbit_kernel_pooling_template.h",
        "c310/op_kernel/int_nbit_split_embedding_pooling_kernel.h",
        {},
        True,
    ),
    (
        "codegen/inference/nbit_kernel_nobag_template.h",
        "c310/op_kernel/int_nbit_split_embedding_nobag_kernel.h",
        {},
        True,
    ),
    (
        "codegen/inference/nbit_adapter_template.cpp",
        "int_nbit_split_embedding_codegen_lookup_function.cpp",
        {},
        True,
    ),
]


class IntNbitSplitEmbeddingCodegenLookupFunctionGenerator:
    @staticmethod
    def validate_weight_types(weight_types: List[str]) -> None:
        unsupported = sorted(set(weight_types) - SUPPORTED_STEP1_WEIGHT_TYPES)
        if unsupported:
            raise ValueError(
                "Codegen only supports FP8/FP16/FP32/INT8/INT4/INT2. "
                f"Unsupported weight types requested: {','.join(unsupported)}"
            )

    @staticmethod
    def generate(
        *,
        repo_root: Path,
        output_root: Path,
        weight_types: List[str],
        check: bool = False,
    ) -> bool:
        IntNbitSplitEmbeddingCodegenLookupFunctionGenerator.validate_weight_types(weight_types)

        is_success = True
        for template_path, output_path, substitutions, add_header in TARGETS:
            template = CodeTemplate(repo_root, template_path)
            is_success = (
                template.write(
                    output_root / output_path,
                    substitutions=substitutions,
                    weight_types=weight_types,
                    check=check,
                    add_header=add_header,
                )
                and is_success
            )
        return is_success
