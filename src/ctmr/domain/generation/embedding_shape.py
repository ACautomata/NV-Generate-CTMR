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

"""The training embedding's NIfTI shape contract (issue #313, series-③ T3).

The vendored encode chain (``create_training_data.process_file``) writes every
training embedding as a channel-last NIfTI ``(X, Y, Z, 4)``: the frozen VAE's
4-channel latent of an image resized to ``round_number`` multiples of 128
(base_number=128, floor 128) -- so each spatial axis is a multiple of 32
(128 / the VAE's 4x spatial compression) and at least 32.

What the contract can and cannot see: a skipped transpose (channel-first
storage), a non-4D artifact, a foreign tree's file, or any grid no
``round_number`` pass could produce are off contract and rejected here. An
axis-ORDER permutation, though, can dress up as a legal latent -- a
(32, 64, 64, 4) replay file may be the honest 128x256x256 encoding or the
scrambled render of a different case geometry. The series-③ audit's ~65%
replay-arm axis-order damage is rebuilt by the T2 re-encode and its manifest
reconciliation; this contract is the structural backstop at startup, not a
permutation detector.
"""

from __future__ import annotations


class EmbeddingShapeContract:
    """The encode-chain latent artifact contract: validate one embedding's shape, reject off-contract geometry.

    ``check`` raises a ``ValueError`` naming the artifact, its list entry and
    the violated axis -- a rejection the training operator can act on without
    opening the file.
    """

    LATENT_CHANNELS = 4  # the frozen VAE's latent channels (channel-last slot)
    ROUND_BASE = 128  # the encode chain's round_number base_number (grid floor)
    VAE_COMPRESSION = 4  # the frozen VAE's spatial downsampling factor
    MIN_SPATIAL = ROUND_BASE // VAE_COMPRESSION  # 32: the smallest legal axis size

    def check(self, shape, *, path, entry):
        """Validate one embedding's NIfTI shape against the contract.

        Args:
            shape: the on-disk NIfTI shape (nibabel header order).
            path: the artifact path (for the diagnostic).
            entry: the training-list entry (``sub``/``case`` name the case).

        Raises:
            ValueError: the shape is off contract; the message names the
                artifact, the entry and each violated axis.
        """
        violations = []
        if len(shape) != 4:
            violations.append(f"ndim {len(shape)} (expected 4)")
        else:
            if shape[-1] != self.LATENT_CHANNELS:
                violations.append(f"channels {shape[-1]} (expected {self.LATENT_CHANNELS} in the last slot)")
            for axis, size in enumerate(shape[:3], start=1):
                if size < self.MIN_SPATIAL or size % self.MIN_SPATIAL != 0:
                    violations.append(f"axis {axis} size {size} (expected a multiple of {self.MIN_SPATIAL} >= {self.MIN_SPATIAL})")
        if violations:
            sub = entry.get("sub", "?")
            case = entry.get("case", "?")
            raise ValueError(
                f"training embedding shape contract violated: {path} shape={tuple(shape)} "
                f"(entry {sub}:{case}): {'; '.join(violations)} -- "
                f"the encode chain writes channel-last latents (X, Y, Z, {self.LATENT_CHANNELS}) "
                f"with each spatial axis a multiple of {self.MIN_SPATIAL} "
                f"(round_number base {self.ROUND_BASE} / VAE compression {self.VAE_COMPRESSION})"
            )
