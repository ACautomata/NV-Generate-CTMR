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

"""Checkpoint ports: the training-shell repository and the instrument reader.

Two persistence faces live here (ADR-0019 §3, #269/#275): ``CheckpointRepository``
is the shell-side weight store (ADR-0015 §4 辖区), ``InstrumentCheckpointReader``
is the judge-chain face onto the frozen instrument's published fold_0 checkpoint.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class CheckpointRepository(Protocol):
    """State_dict payload store/fetch plus tmp atomic publish and latest.json pointer."""

    def save(self, payload, epoch: int) -> Path:
        """Publish one epoch payload; returns the published artifact path."""
        ...

    def load(self, path: str | Path):
        """Fetch a published payload (weights_only: the repository's own artifacts are trusted)."""
        ...


@runtime_checkable
class InstrumentCheckpointReader(Protocol):
    """Reader for the frozen instrument's published fold_0 checkpoint (ADR-0009, #275).

    The judge chain (closing verification) and the instrument trainer read only
    the recorded metadata (``current_epoch`` / ``trainer_name``) out of
    ``checkpoint_final.pth``. The weights_only allowlist scoping that makes the
    load safe is the adapter's guarantee, not the caller's concern -- callers
    depend on this port and never touch torch serialization state.
    """

    def read(self, checkpoint: str | Path) -> dict:
        """Load one checkpoint payload on cpu (weights_only under the scoped allowlist)."""
        ...
