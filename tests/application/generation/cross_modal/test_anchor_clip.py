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

"""The P3 anchor-encode clip gate (issue #313, series-③ T3).

The T4 factory moved the training encoding arm to clip=True (job C's measured
verdict -- extrapolated >1.0 inputs leave the frozen VAE's reconstruction
domain). The P3 inference anchor adapter must match that domain: an anchor
volume whose 0-99.5 percentile affine extrapolates its top tail above 1.0
must come out bounded once the anchor chain runs. Same seam family as
tests/infrastructure/maisi_engine/test_intensity_transform_factory.py (the
bounded-output contract); here the reading rides the FULL external behavior
of ``AnchorLatentEncoder.encode`` -- an identity autoencoder stands in for the
frozen VAE (its encoder is out of CPU scope), so the returned "latent" is the
preprocessed intensity itself and the bounded-output assertion is the
encoder's contract, not a transform-internals probe.
"""

from __future__ import annotations

import logging

import nibabel as nib
import numpy as np
import pytest
import torch

from ctmr.application.generation.cross_modal.anchor import AnchorLatentEncoder
from ctmr.infrastructure.engine import MaisiEngine

pytestmark = pytest.mark.torch

CPU = torch.device("cpu")


class _IdentityAutoencoder(torch.nn.Module):
    """Stands in for the frozen VAE: the anchor chain's output passes through
    unchanged (the chain under test is the preprocessing, never the VAE)."""

    def encode_stage_2_inputs(self, x):
        return x


def _extrapolating_volume():
    """A volume whose 0-99.5 percentile affine extrapolates its top tail above 1.0.

    95% of voxels sit at 0.0 and 4.8% at 1.0, so the percentile anchors land at
    (0.0, 1.0); the remaining 0.2% at 10.0 sit past the 99.5th percentile, and
    the affine maps them to 10.0 in the unclipped world -- the extrapolated band
    job C measured. (The tail must stay under 0.5% of the voxels or the upper
    anchor moves into the tail itself and no extrapolation happens.)
    """
    volume = np.zeros((32, 32, 32), dtype=np.float32)
    volume.flat[:1573] = 1.0  # 4.8%: the band the 99.5th percentile anchors in
    volume.flat[1573:1639] = 10.0  # 0.2%: the extrapolating tail
    return volume


def _encoder(tmp_path, output_size=(64, 64, 32)):
    anchor = tmp_path / "anchor.nii.gz"
    nib.save(nib.Nifti1Image(_extrapolating_volume(), np.diag([1.0, 1.0, 1.0, 1.0])), str(anchor))
    encoder = AnchorLatentEncoder(
        autoencoder=_IdentityAutoencoder(),
        device=CPU,
        output_size=output_size,
        logger=logging.getLogger("test-anchor"),
        engine=MaisiEngine(),
    )
    return encoder, str(anchor)


def test_anchor_encode_bounds_the_extrapolating_tail(tmp_path):
    """clip=True aligns the anchor arm with the T4 training domain: the
    >1.0 extrapolated tail truncates exactly at the cap instead of leaving
    the frozen VAE's input domain (T4's bounded-output contract, job C)."""
    encoder, anchor_path = _encoder(tmp_path)
    z = encoder.encode(anchor_path)

    assert float(z.max()) == pytest.approx(1.0)  # the tail truncates at the cap
    assert float(z.min()) == pytest.approx(0.0)  # the floor anchors map to 0.0
    assert z.dtype == torch.float32  # the float contract the denoise loop consumes
