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

"""Canonical execution entry of the frozen instrument call (ADR-0009 decision 3, issue #107).

``python -m ctmr.instrument.predict`` runs the native nnUNetv2
``predict_entry_point`` inside the ``nnunet_safe_globals()`` scope -- the one
entry shared by Python callers (``subprocess.run(cmd.build(...))``) and shell
orchestration, inference behaviour identical to ``nnUNetv2_predict`` (this is
the promotion of ``scripts/l2_calibration_predict_entry.py``). The scope keeps
``torch.load`` robust under the torch>=2.6 default ``weights_only=True``
(checkpoints carry numpy scalars / dtypes). ``src`` must be on ``sys.path``
(PYTHONPATH or a deployment shim) -- that wiring is ADR-0009 decision 1's
call-site rollout, not this module's concern.
"""

import sys

from nnunetv2.inference.predict_from_raw_data import predict_entry_point

from ctmr.instrument.safeglobals import nnunet_safe_globals


def main():
    """Run the native predictor inside the frozen safe-globals scope."""
    with nnunet_safe_globals():
        return predict_entry_point()


if __name__ == "__main__":
    sys.exit(main())
