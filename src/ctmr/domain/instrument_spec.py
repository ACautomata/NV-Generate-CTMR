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

"""Frozen instrument command construction (ADR-0009, issue #107).

The single construction point of the frozen instrument call: the nnU-Net
prediction invocation under the frozen inference configuration (fold 0,
``nnUNetTrainer250Epochs``, mirror TTA on; SSA = ``3d_fullres_bs16`` +
``nnUNetPlans_SSA_bs16_v1``).

Mirror TTA ON is a frozen invariant of this module: nnUNetv2's
``--disable_tta`` is a ``store_true`` flag, so TTA stays on by omission --
this interface exposes no TTA parameter at all and ``build`` never emits
``--disable_tta`` (any value passed to that flag, including ``False``, is a
fatal argparse error, #78).

``build`` is a pure transform: it only produces argv for the canonical entry
point ``python -m ctmr measure predict`` (ADR-0009 decision 3, executed by
``ctmr.infrastructure.nnunet_runner`` since #140) -- no execution, no file IO;
running it (subprocess / writing a shell script) stays with the caller.
"""

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InstrumentSpec:
    """Value object of one challenge's frozen inference spec (ADR-0009 decision 1)."""

    dataset_id: str
    config: str
    plans: str
    trainer: str = "nnUNetTrainer250Epochs"
    fold: int = 0


INSTRUMENT_SPECS = {
    "GLI": InstrumentSpec(dataset_id="Dataset501_BraTS2023GLI", config="3d_fullres", plans="nnUNetPlans"),
    "SSA": InstrumentSpec(dataset_id="Dataset502_BraTS2023SSA", config="3d_fullres_bs16", plans="nnUNetPlans_SSA_bs16_v1"),
    "MEN": InstrumentSpec(dataset_id="Dataset503_BraTS2023MEN", config="3d_fullres", plans="nnUNetPlans"),
    "METS": InstrumentSpec(dataset_id="Dataset504_BraTS2023METS", config="3d_fullres", plans="nnUNetPlans"),
    "PED": InstrumentSpec(dataset_id="Dataset505_BraTS2023PED", config="3d_fullres", plans="nnUNetPlans"),
}
"""Per-challenge frozen specs. Both nnUNetv2 spellings of a dataset id resolve
to the same model directory; the unambiguous full name is the canonical form
(the ADR-0002 calibration entry and ``brats_p1_dev_eval`` already use it)."""


class FrozenInstrumentCommand:
    """One challenge's frozen instrument call: an argv builder, nothing else.

    Holds the frozen ``InstrumentSpec``; ``build`` turns a raw input directory
    and an output directory into the canonical argv. Execution, environment
    setup (``nnUNet_raw`` / ``nnUNet_results`` / ...) and file IO stay with
    the caller.
    """

    def __init__(self, spec: InstrumentSpec):
        self._spec = spec

    def build(self, input_dir: Path | str, output_dir: Path | str) -> list[str]:
        """Pure argv construction: no execution, no file IO, no TTA flag ever."""
        spec = self._spec
        return [
            sys.executable,
            "-m",
            "ctmr",
            "measure",
            "predict",
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-d",
            spec.dataset_id,
            "-c",
            spec.config,
            "-p",
            spec.plans,
            "-tr",
            spec.trainer,
            "-f",
            str(spec.fold),
        ]
