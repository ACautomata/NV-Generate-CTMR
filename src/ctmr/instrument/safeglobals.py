# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single definition of the torch ``weights_only`` allowlist for nnU-Net
checkpoints, plus its scoped activation (ADR-0009 decision 4, issue #107).

nnU-Net checkpoints carry numpy scalars (spacing etc.) and their dtypes in
``logging`` / ``init_args``; loading them under ``weights_only=True`` (the
torch>=2.6 default) needs these passive types allowlisted -- arbitrary
executable classes stay rejected. The three byte-identical import-time
``add_safe_globals`` copies (``nnunet_l2_instrument.py``,
``nnunet_l2_closing_verification.py``, ``l2_calibration_predict_entry.py``)
collapse into this single definition; activation is scoped through
``nnunet_safe_globals``, so importing this module never mutates global torch
state.
"""

import numpy
import torch

NUMERIC_DTYPE_NAMES = (
    "bool",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
)

# numpy>=2 deserializes dtypes as numpy.dtypes.*DType subclasses; the concrete
# classes of the numeric dtypes actually carried by checkpoints are allowlisted
# one by one (the verbatim payload of the three collapsed copies).
NNUNET_SAFE_GLOBALS = [numpy.core.multiarray.scalar, numpy.dtype] + [type(numpy.dtype(name)) for name in NUMERIC_DTYPE_NAMES]


class nnunet_safe_globals:
    """Scoped activation of the frozen allowlist on top of the current state.

    Entering adds ``NNUNET_SAFE_GLOBALS``; exiting restores the prior
    allowlist's content -- no residue, no clobbering of whatever was already
    active (the registry is set-like, so ordering is not preserved).
    """

    def __enter__(self) -> "nnunet_safe_globals":
        self._previous = list(torch.serialization.get_safe_globals())  # snapshot copy: the returned list may be the live internal one
        torch.serialization.add_safe_globals(NNUNET_SAFE_GLOBALS)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        torch.serialization.clear_safe_globals()
        if self._previous:
            torch.serialization.add_safe_globals(self._previous)
        return False
