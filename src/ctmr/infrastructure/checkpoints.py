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

"""CheckpointRepository -- the single persistence protocol for weight payloads (ADR-0015 §4, #135).

Owns the state_dict payload store/fetch and the tmp atomic publication protocol
(``epoch_<N>.pt`` tmp + rename, sunk verbatim out of the training shell) plus
the ``latest.json`` pointer protocol: the pointer is written only after the
renamed file is fully on disk, so it can never direct a reader at a partial
write. Payloads stream straight to the tmp file (no full-artifact RAM copy), and
the payload key set is repo-transparent (``checkpoint_payload`` returns the
storage object verbatim). Provenance is a runtime log, not model state: it stays
in the application layer.
"""

import json
from pathlib import Path

import torch


class CheckpointRepository:
    """State_dict payload store/fetch plus tmp atomic publish and latest.json pointer."""

    def __init__(self, model_dir):
        self._model_dir = Path(model_dir)

    def save(self, payload, epoch):
        """Publish one epoch payload: tmp write + rename, then the latest.json pointer (rank-0 only)."""
        final = self._model_dir / f"epoch_{epoch}.pt"
        tmp = final.with_name(final.name + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(final)
        (self._model_dir / "latest.json").write_text(json.dumps({"epoch": epoch, "checkpoint": str(final)}) + "\n")
        return final

    def load(self, path):
        """Fetch a published payload (weights_only: the repository's own artifacts are trusted)."""
        return torch.load(path, weights_only=True)
