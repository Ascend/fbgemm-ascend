#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from .generate_int_nbit_split_embedding_codegen_lookup_function import (
        IntNbitSplitEmbeddingCodegenLookupFunctionGenerator,
    )
except ImportError:
    from generate_int_nbit_split_embedding_codegen_lookup_function import (
        IntNbitSplitEmbeddingCodegenLookupFunctionGenerator,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fbgemm-ascend TBE inference sources.")
    parser.add_argument(
        "--install-dir",
        type=Path,
        required=True,
        help="Path to the fbgemm-ascend source root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to write generated TBE inference sources.",
    )
    parser.add_argument(
        "--weight-types",
        default="FP8",
        help="Comma-separated weight types to generate. Step1 supports FP8 only.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check generated outputs without writing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.install_dir.resolve()
    output_root = args.output_dir.resolve()

    if not repo_root.exists() or not repo_root.is_dir():
        logger.error("install-dir does not exist or is not a directory: %s", repo_root)
        return 1
    if not output_root.exists() or not output_root.is_dir():
        logger.error("output-dir does not exist or is not a directory: %s", output_root)
        return 1

    weight_types = [item.strip().upper() for item in args.weight_types.split(",") if item.strip()]

    is_success = IntNbitSplitEmbeddingCodegenLookupFunctionGenerator.generate(
        repo_root=repo_root,
        output_root=output_root,
        weight_types=weight_types,
        check=args.check,
    )
    return 0 if is_success else 1


if __name__ == "__main__":
    sys.exit(main())
