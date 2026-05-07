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
import numpy as np
import torch
import torch_npu


def npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available() and torch.npu.device_count() > 0


def bfloat_quantize(x_float: float) -> np.uint16:
    """Numpy reference implementation of bfloat16 quantization (round-to-nearest-even)."""
    import struct
    bits = struct.unpack('>I', struct.pack('>f', x_float))[0]
    bits += 1 << 15
    bits = bits >> 16
    return np.uint16(bits)


def bfloat_dequantize(x_bfloat: np.uint16) -> float:
    """Numpy reference implementation of bfloat16 dequantization."""
    import struct
    bits = np.int32(x_bfloat) << 16
    return struct.unpack('>f', struct.pack('>I', bits))[0]
