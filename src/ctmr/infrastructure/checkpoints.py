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

"""State_dict-payload checkpoint storage (ADR-0015 section 4, issue #135).

``CheckpointRepository`` is the single persistence protocol for training
checkpoints (CONTEXT.md "Checkpoint Identity"): per-epoch ``epoch_<N>.pt``
payload files published via tmp + rename so a concurrent reader (the dev
sidecar polls ``epoch_*.pt``, and ``latest.json`` names the finished file) can
never observe a partial write. Sunk verbatim from the shell's inline
publication step (previously ``ctmr.harness.train_shell._publish_checkpoint``,
ADR-0011 #111). The payload is whatever a kernel's ``checkpoint_payload`` hook
returns -- the repository carries it opaquely; the payload key set (P1
``unet_state_dict``, P2/P3 ``controlnet_state_dict``, over the shared
epoch/loss/num_train_timesteps/scale_factor skeleton) stays kernel-owned.
Provenance writers are run logs, not model state, and stay out of the
repository (application layer, by ADR).

Torch-level: needs torch for save/load.
"""

import json
from pathlib import Path

import torch


class CheckpointRepository:
    """One payload per epoch, atomically published, ``latest.json`` kept pointing at it."""

    def __init__(self, model_dir):
        self._model_dir = Path(model_dir)

    def checkpoint_path(self, epoch):
        """The published file name for one epoch (the dev sidecar's poll pattern is ``epoch_<N>.pt``)."""
        return self._model_dir / f"epoch_{epoch}.pt"

    def save(self, payload, epoch):
        """Atomically publish one epoch's payload: write ``<name>.tmp``, then rename it final."""
        path = self.checkpoint_path(epoch)
        tmp = path.with_name(path.name + ".tmp")
        # Atomic publication: readers poll for epoch_<N>.pt and must never observe a partial write.
        torch.save(payload, tmp)
        tmp.replace(path)
        return path

    def load(self, path, *, map_location="cpu"):
        """Read back a stored payload."""
        return torch.load(path, map_location=map_location, weights_only=True)

    def point_latest(self, epoch):
        """Point ``latest.json`` at an already-completed epoch publication."""
        pointer_path = self._model_dir / "latest.json"
        pointer_path.write_text(json.dumps({"epoch": epoch, "checkpoint": str(self.checkpoint_path(epoch))}) + "\n")
        return pointer_path

    def publish(self, payload, epoch):
        """The publication protocol in one call: atomic payload first, pointer only after."""
        self.save(payload, epoch)
        self.point_latest(epoch)
        return self.checkpoint_path(epoch)
