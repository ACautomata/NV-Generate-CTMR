"""The MR/CT intensity-transform factory gates (issue #251, series-② T4).

The training encoding recipe's seam: the mri arm's normalization flag moved
clip=False -> clip=True (job C's measured verdict -- extrapolated >1.0 inputs
leave the frozen VAE's reconstruction domain, self-eval MAE 0.8673 vs 0.0062
in-domain, with intra-tumour negative-value artifacts). The factory tests pin
the bounded-output contract the recipe change buys: a synthetic volume whose
top tail extrapolates above 1.0 under the 0-99.5 percentile affine must come
out bounded once the flag is True. The ct arm's clip=True is pinned against
silent regression; the unsupported-modality warning path stays covered by the
engine smoke module (test_engine_smoke.py).
"""

import numpy as np
import pytest

from ctmr.infrastructure.maisi_engine.instance_definition import define_fixed_intensity_transform

pytestmark = pytest.mark.torch


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


def test_mri_factory_output_is_bounded_above_one():
    transform = define_fixed_intensity_transform(modality="mri")[0]
    out = transform({"image": _extrapolating_volume()})["image"]
    assert float(out.max()) <= 1.0, f"mri normalization must not extrapolate above 1.0, got max {float(out.max())}"
    assert float(out.min()) >= 0.0


def test_mri_factory_output_spans_the_unit_range_below_the_tail():
    """The affine itself is unchanged: in-range voxels still span [0, 1]; only the
    tail is truncated at 1.0 (the clip policy's bounded compression, job C #216
    review -- truncation, not rescaling)."""
    transform = define_fixed_intensity_transform(modality="mri")[0]
    out = np.asarray(transform({"image": _extrapolating_volume()})["image"])
    assert float(out.min()) == pytest.approx(0.0)  # 0.0 anchors map to the range floor
    assert float(out.max()) == pytest.approx(1.0)  # the tail truncates exactly at the cap
    assert float(np.percentile(out, 99.4)) == pytest.approx(1.0)  # the 1.0 band survives at its mapped value
    assert float(np.percentile(out, 50.0)) == pytest.approx(0.0)  # the bulk stays at the floor


def test_mri_factory_transform_carries_clip_true():
    assert define_fixed_intensity_transform(modality="mri")[0].scaler.clip is True


def test_ct_factory_keeps_clip_true():
    """The ct arm was always clip=True ([-1000, 1000] -> [0, 1]); pinned so a
    future factory edit cannot silently flip either arm."""
    transform = define_fixed_intensity_transform(modality="ct")[0]
    assert transform.scaler.clip is True
    out = transform({"image": np.array([-2000.0, -1000.0, 0.0, 1000.0, 2000.0], dtype=np.float32)})["image"]
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
