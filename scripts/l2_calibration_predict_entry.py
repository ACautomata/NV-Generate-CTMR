#!/usr/bin/env python3
"""nnUNetv2 推理入口包装（Issue #36 校准）。

仅做一件事：在进入 nnUNetv2 命令行前注册与 scripts/nnunet_l2_instrument.py
一致的 torch weights_only 白名单（checkpoint 的 logging/init_args 携带 numpy
标量与 dtype，torch>=2.6 默认 weights_only 会拒绝），然后调用原生
predict_entry_point —— 推理行为与 nnUNetv2_predict 完全一致。
"""

import sys

import numpy
import torch

torch.serialization.add_safe_globals(
    [numpy.core.multiarray.scalar, numpy.dtype]
    + [type(numpy.dtype(name)) for name in ("bool", "uint8", "int8", "int16", "int32", "int64",
                                            "float16", "float32", "float64", "complex64", "complex128")]
)

from nnunetv2.inference.predict_from_raw_data import predict_entry_point  # noqa: E402

if __name__ == "__main__":
    sys.exit(predict_entry_point())
