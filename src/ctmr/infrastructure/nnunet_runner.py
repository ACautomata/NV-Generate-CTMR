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

"""Frozen-instrument execution side: the native nnUNetv2 predictor inside the
weights_only allowlist scope (ADR-0009 decisions 3+4, issues #107/#140; ADR-0015 §2).

``MeasurePredictVerb`` backs the canonical entry ``ctmr measure predict`` --
the one execution door shared by Python callers (``subprocess.run`` of
``FrozenInstrumentCommand.build(...)`` argv) and shell orchestration, with
inference behaviour identical to ``nnUNetv2_predict``. The verb passes its
command-line tail straight through to the native nnUNetv2 ``predict_entry_point``
(frozen defaults of that entry: mirror TTA on by omission -- the flag is
``store_true``, passing any value including ``False`` enables it -- overlap 0.5,
fold 0 unless ``-f``, ``nnUNetTrainer250Epochs`` via ``-tr``) and runs it inside
the ``nnunet_safe_globals()`` scope. The scope keeps ``torch.load`` robust under
the torch>=2.6 default ``weights_only=True`` (checkpoints carry numpy scalars /
dtypes); importing this module never mutates global torch state.

Reaching this module needs torch / monai / nnunetv2 (the CI full-dependency
tier installs them all); the ``ctmr.cli`` surface reaches it lazily so the CLI
skeleton stays importable on any machine (ADR-0015 §3).
"""

import sys

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
# one by one (the verbatim payload of the collapsed copies).
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


class MeasurePredictVerb:
    """The ``ctmr measure predict`` verb body: argv pass-through plus scoped activation."""

    def run(self, pass_through):
        """Run the native predictor on the caller-supplied nnUNetv2 flags.

        The verb's own argparse collects everything after ``measure predict``;
        resetting ``sys.argv`` to the tail lets the native ``predict_entry_point``
        parser see exactly the flags a frozen command's argv carries.
        """
        from nnunetv2.inference.predict_from_raw_data import predict_entry_point  # deferred: keeps ctmr.cli importable without nnunetv2

        sys.argv = [sys.argv[0], *pass_through]
        with nnunet_safe_globals():
            return predict_entry_point()


def main(pass_through=None):
    """Module-run convenience form (kept for direct subprocess callers); returns the process exit code."""
    return MeasurePredictVerb().run(list(pass_through or ()))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
