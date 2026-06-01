#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import difflib
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


GENERATED_HEADER_C = """/*
 * GENERATED FILE INFO
 *
 * Template Source: {template_path}
 * Supported Weight Types: {weight_types}
 */

"""


class CodeTemplate:
    def __init__(self, root: Path, relative_path: str) -> None:
        self.root = root
        self.relative_path = relative_path
        self.path = root / relative_path

    def render(self, substitutions: Optional[Dict[str, str]] = None) -> str:
        text = self.path.read_text()
        for key, value in (substitutions or {}).items():
            text = text.replace("{{" + key + "}}", value)
        return text

    def write(
        self,
        output_path: Path,
        *,
        substitutions: Optional[Dict[str, str]] = None,
        weight_types: List[str],
        check: bool = False,
        add_header: bool = True,
    ) -> bool:
        body = self.render(substitutions)
        output = body
        if add_header:
            output = (
                GENERATED_HEADER_C.format(
                    template_path=self.relative_path,
                    weight_types=",".join(weight_types),
                )
                + body
            )

        if check:
            current = output_path.read_text() if output_path.exists() else ""
            if current != output:
                diff = difflib.unified_diff(
                    current.splitlines(keepends=True),
                    output.splitlines(keepends=True),
                    fromfile=str(output_path),
                    tofile=f"{output_path} (generated)",
                )
                logger.info("".join(diff))
                return False
            return True

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output)
        logger.info("Written: %s", output_path)
        return True
