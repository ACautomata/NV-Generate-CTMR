# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""L2 shared vocabulary: measurement-face words and the measurement-table CSV
protocol (ADR-0017 decision 1, issue #229).

Everything here moved verbatim out of the terminal-acceptance judge
(``final_acceptance``), which now imports this module -- the judge is no
longer the shared-vocabulary host. Hosted beside the CSV protocol they
belong to:

- ``MODALITIES`` / ``CHANNEL_SUFFIXES``: the four modality words and their
  frozen nnUNet input-channel spellings -- the input half of the measurement
  face (predict scripts, execution-side channel IO, report tabs);
- ``MEASUREMENT_FIELDS`` + ``MeasurementTable``: the wide 27-column
  per-observation CSV protocol (the long 24-column calibration family's
  canonical home is ``ctmr.domain.measurement`` -- the numpy side -- whose
  ``FINAL_ACCEPTANCE_FIELDS`` mirror is pinned equal to ``MEASUREMENT_FIELDS``
  by test_measurement_adoption);
- ``AcceptanceError``: the L2 acceptance-domain error type. It lives here
  because the protocol raises it (unreadable tables) and the diagnostic
  readers catch it across the shared boundary -- hosting it beside the
  protocol keeps every consumer on shared-vocabulary imports only.

The dependency closure is third-party-free -- stdlib only, numpy/scipy/torch
unreachable -- registered by the import-face probe in
``tests/application/acceptance/distribution/test_shared_vocab.py``.
"""

import csv
import math
from pathlib import Path

MODALITIES = ("t1n", "t1c", "t2w", "t2f")
CHANNEL_SUFFIXES = {"t1n": "0000", "t1c": "0001", "t2w": "0002", "t2f": "0003"}

MEASUREMENT_FIELDS = [
    "obs_id",
    "challenge",
    "case",
    "side",
    "anchor",
    "input_fail",
    "run_fail",
    "hier_viol",
    "pred_empty",
    "vol_wt_ml",
    "vol_tc_ml",
    "vol_et_ml",
    "brain_ml",
    "wt_brain",
    "et_wt",
    "cx_wt_mm",
    "cy_wt_mm",
    "cz_wt_mm",
    "cx_tc_mm",
    "cy_tc_mm",
    "cz_tc_mm",
    "cx_et_mm",
    "cy_et_mm",
    "cz_et_mm",
    "cond_dice_wt",
    "cond_dice_tc",
    "cond_dice_et",
]


class AcceptanceError(Exception):
    """Raised when acceptance setup, freeze verification or judgement rules break."""


class MeasurementTable:
    """CSV persistence for per-observation measurements (controlled storage, subject IDs)."""

    @classmethod
    def write(cls, rows, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MEASUREMENT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    @classmethod
    def read(cls, path):
        with open(Path(path), newline="") as handle:
            rows = list(csv.DictReader(handle))
        missing = set(MEASUREMENT_FIELDS) - set(rows[0].keys()) if rows else set(MEASUREMENT_FIELDS)
        if missing:
            raise AcceptanceError(f"measurement table {path} is missing columns: {sorted(missing)}")
        return rows

    @staticmethod
    def number(row, field):
        """Parses a measurement cell; empty/None -> None (undefined quantity)."""
        value = row.get(field)
        if value is None or value == "":
            return None
        parsed = float(value)
        return None if math.isnan(parsed) else parsed

    @staticmethod
    def flag(row, field):
        return str(row.get(field, "")).strip().lower() in ("1", "true", "yes")
