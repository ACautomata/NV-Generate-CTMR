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

"""CheckpointRepository: the persistence port for weight payloads (ADR-0019 §3, #269).

The port application shells drive: ``save(payload, epoch)`` publishes one
epoch's state_dict payload (tmp write + rename, then the ``latest.json``
pointer written only after the renamed file is fully on disk -- it can never
direct a reader at a partial write) and returns the published path;
``load(path)`` fetches a published payload verbatim, the payload key set
repo-transparent. Protocol only -- the filesystem realization lives in
``ctmr.infrastructure.checkpoints`` (ADR-0015 §4 辖区不变).
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
