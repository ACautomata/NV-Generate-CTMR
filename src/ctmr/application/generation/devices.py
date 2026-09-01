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

"""Unified device injection for the diagnostic/inference entries (issue #280, ADR-0019 §8).

The scattered entries used to self-select ``torch.device("cuda" if
torch.cuda.is_available() else "cpu")`` inside their ``main`` -- uninjectable
from the outside. This module is the single definition point for both halves
of the injection: the flag face (``add_device_flag`` on each entry's argparse
surface) and the resolution face (``resolve_device`` in each entry's
``main``). An absent flag resolves exactly like the hardcoded era (cuda when
available, cpu otherwise) so the default invocation path is unchanged; an
explicit ``--device`` value overrides it. Two families stay outside this
cleanup, and any repo-wide sweep for residual device hardcoding must account
for them: the intensity_domain/reencode arms keep their own ``--device``
surfaces with ``default="cpu"`` (CPU-diagnostic by design, predating this
cleanup), and the training-data elastic augmentation
(``infrastructure/dataio/augmentation.py``) keeps its verbatim-migrated bare
``.cuda()`` calls (a training-data path, not a diagnostic/inference entry;
issue #280 scope).
"""

from __future__ import annotations

import torch

DEVICE_FLAG_HELP = "torch device override (default: cuda if available else cpu)"


def add_device_flag(parser):
    """Attach the unified ``--device`` flag to one entry's argparse surface."""
    parser.add_argument("--device", default=None, help=DEVICE_FLAG_HELP)
    return parser


def resolve_device(cli_value=None):
    """Resolve the injected flag value: explicit wins, absent falls back to the hardcoded-era behavior."""
    if cli_value:
        return torch.device(cli_value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
