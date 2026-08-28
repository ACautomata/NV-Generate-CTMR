"""CPU-able behavioral tests for ctmr.infrastructure.dataio.augmentation (migrated from the retired scripts layer (git history; ``augmentation``)).

The tumor-* elastic helpers route through hardcoded ``.cuda()`` calls (GPU-only paths, out of the
"any machine" test line per ADR-0015 §6); everything tested here is the CPU-safe surface.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from ctmr.infrastructure.dataio.augmentation import (
    augmentation_body,
    augmentation_tumor_only,
    dilate3d,
    erode3d,
    finalize_tumor_mask,
    remap_labels,
    remove_tumors,
    remove_tumors_majority_vote,
)

pytestmark = pytest.mark.torch

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_erode3d_dilate3d_keep_shape_and_are_fixed_points_on_solid():
    solid = torch.ones(8, 8, 8)
    empty = torch.zeros(8, 8, 8)
    assert dilate3d(solid).shape == (8, 8, 8)
    assert erode3d(solid).shape == (8, 8, 8)
    assert torch.equal(dilate3d(solid), solid)
    assert torch.equal(erode3d(solid), solid)
    # pad value is 1.0, so the boundary band of an empty volume reads true; the deep interior stays clear
    assert dilate3d(empty)[4, 4, 4] == 0
    assert erode3d(empty)[4, 4, 4] == 0


def test_dilate3d_grows_single_voxel_to_kernel_cube():
    # pad value is 1.0 in the upstream implementation, so the outermost 2-voxel
    # boundary ring reads as true everywhere; the strictly-interior result is the
    # clean 3x3x3 cube around the seed.
    seed = torch.zeros(12, 12, 12)
    seed[6, 6, 6] = 1.0
    grown = dilate3d(seed, erosion=3)
    assert torch.equal(grown[5:8, 5:8, 5:8], torch.ones(3, 3, 3))
    assert grown[2, 2, 2] == 0.0  # far from seed and outside the boundary band


def test_remove_tumors_majority_vote_replaces_tumor_with_organ_ring_mode():
    # upstream contracts on a 4D volume ([1, D, H, W]) for boolean indexing
    volume = torch.full((1, 5, 5, 5), 28)
    tumor_mask = torch.zeros(1, 5, 5, 5)
    tumor_mask[0, 2, 2, 2] = 1
    out = remove_tumors_majority_vote(tumor_mask, volume, organ_label_lists=(28, 29, 30, 31, 32))
    assert out[0, 2, 2, 2] == 28


def test_remove_tumors_majority_vote_falls_back_to_most_common_organ_when_ring_empty():
    volume = torch.zeros(1, 5, 5, 5)
    volume[0, 1:4, 1:4, 1:4] = 32  # only a distant organ, no ring around the tumor
    tumor_mask = torch.zeros(1, 5, 5, 5)
    tumor_mask[0, 0, 0, 0] = 1
    out = remove_tumors_majority_vote(tumor_mask, volume, organ_label_lists=(28, 29, 30, 31, 32))
    assert out[0, 0, 0, 0] == 32


def test_remove_tumors_maps_organ_tumors_to_organs():
    labels = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    labels[0, 1, 1, 1] = 26  # hepatic tumor
    labels[0, 2, 2, 2] = 24  # pancreatic tumor
    out = remove_tumors(labels)
    assert out[0, 1, 1, 1] == 1
    assert out[0, 2, 2, 2] == 4


def test_remove_tumors_with_pseudo_labels_replaces_lesions():
    labels = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    labels[0, 1, 1, 1] = 23  # lung tumor
    pseudo = torch.ones_like(labels) * 29  # every voxel offers a plausible lung label
    out = remove_tumors(labels, pseudo_labels=pseudo)
    assert out[0, 1, 1, 1] == 29


def test_remove_tumors_rejects_2d_input():
    with pytest.raises(ValueError, match="3D/4D"):
        remove_tumors(torch.zeros(2, 2))


def test_remap_labels_applies_mapping():
    x = torch.tensor([[[[3, 1]]]], dtype=torch.long)
    out = remap_labels(x, {3: 200})
    assert out.tolist() == [[[[200, 1]]]]


def test_finalize_tumor_mask_accepts_large_enough_mask_and_rejects_small():
    organ = torch.ones(1, 6, 6, 6)
    big = torch.zeros(1, 6, 6, 6)
    big[0, 1:4, 1:4, 1:4] = 1
    accepted = finalize_tumor_mask(big, organ, threshold_tumor_size=27 * 0.8)
    assert accepted is not None
    assert accepted.shape == (1, 6, 6, 6)
    small = torch.zeros_like(big)
    small[0, 1:3, 1:3, 1:3] = 1
    assert finalize_tumor_mask(small, organ, threshold_tumor_size=8 * 0.8) is not None
    tiny = torch.zeros_like(big)
    tiny[0, 2:3, 2:3, 2:3] = 1
    assert finalize_tumor_mask(tiny, organ, threshold_tumor_size=27 * 0.8) is None


def test_finalize_tumor_mask_counts_voxels_not_label_values():
    """Regression (Codex P1, PR #155): the BraTS path feeds labels 401/402/403 into
    ``finalize_tumor_mask`` while ``threshold_tumor_size`` derives from a binary
    voxel count. Summing raw label values lets two surviving voxels (401+402 = 803)
    clear a voxel-count threshold of 3, so a severely clipped augmentation is
    accepted despite ``min_tumor_size_ratio=0.8``. The check must count voxels."""
    organ = torch.ones(1, 6, 6, 6)
    tiny = torch.zeros(1, 6, 6, 6)
    tiny[0, 2, 2, 2] = 401
    tiny[0, 2, 2, 3] = 402  # two retained voxels, two distinct label values
    assert finalize_tumor_mask(tiny, organ, threshold_tumor_size=3.0) is None
    tiny2 = torch.zeros_like(tiny)
    tiny2[0, 2, 2, 2] = 1
    tiny2[0, 2, 2, 3] = 1  # binary-format anchor: same voxel count, same verdict
    assert finalize_tumor_mask(tiny2, organ, threshold_tumor_size=3.0) is None


class _Stub:
    def __init__(self, tensor):
        self.tensor = tensor

    def as_tensor(self):
        return self.tensor


class _PassThrough:
    def __call__(self, tensor, spatial_size=None):
        return _Stub(tensor)


def test_augmentation_tumor_only_identity_augment_preserves_hit_region():
    tumor = torch.zeros(1, 8, 8, 8)
    tumor[0, 2:5, 2:5, 2:5] = 1  # 27-voxel binary tumor region
    organ = torch.ones(1, 8, 8, 8)
    out = augmentation_tumor_only(tumor, organ, _PassThrough(), tumor_label=1, min_tumor_size_ratio=0.8)
    assert out is not None
    assert out.shape == (1, 8, 8, 8)
    assert out[0, 2:5, 2:5, 2:5].sum() == 27


def test_augmentation_body_preserves_shape_with_seeded_zoom():
    volume = torch.zeros(1, 16, 16, 16)
    volume[0, 4:12, 4:12, 4:12] = 200
    out = augmentation_body(volume.clone(), random_seed=0)
    assert out.shape == volume.shape


def test_augmentation_tumor_liver_loop_is_bounded_when_tumor_cannot_qualify():
    """Regression (Codex P1, PR #155): the specialized tumor-augmentation loops
    retried without bound — when the distorted tumor can never retain the required
    fraction inside the organ mask (e.g. ``output_size`` clips it away) the loop
    never terminates and holds the GPU job. It now fails after MAX_COUNT retries
    like ``augmentation_tumor_only``.

    The elastic transform is stubbed to an all-zero output (a deterministic
    stand-in for "clipped away": sum stays 0 forever, so no random seed can make
    this input qualify). The loop runs in a subprocess so that before the fix the
    unbounded loop shows up as a timeout — red — instead of wedging pytest itself.
    GPU-only code is CPU-thawed via the identity ``.cuda()`` patch.
    """
    script = textwrap.dedent("""\
        import sys, torch
        torch.Tensor.cuda = lambda self, *a, **k: self  # CPU-ize hardcoded .cuda() calls

        from monai.transforms import Rand3DElastic

        class _Stub:
            def __init__(self, tensor): self.tensor = tensor
            def as_tensor(self): return self.tensor

        class _ZeroElastic:
            def __call__(self, img, *args, **kwargs):
                return _Stub(torch.zeros_like(img))

        Rand3DElastic.__call__ = _ZeroElastic.__call__

        from ctmr.infrastructure.dataio.augmentation import augmentation_tumor_liver

        vol = torch.zeros(1, 1, 32, 32, 32, dtype=torch.uint8)
        vol[0, 0, 10, 10, 10] = 26  # hepatic tumor present -> tumor_size > 0
        try:
            augmentation_tumor_liver(vol, output_size=(32, 32, 32), random_seed=0)
        except ValueError as e:
            print("BOUNDED:", e)
            sys.exit(0)
        sys.exit(2)  # returned without raising
    """)
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src") + os.pathsep + str(REPO_ROOT)}
    try:
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail("retry loop did not terminate: unbounded retry (Codex P1 regression)")
    assert proc.returncode == 0, f"expected bounded ValueError, got rc={proc.returncode}: {proc.stderr[-2000:]}"
    assert "BOUNDED" in proc.stdout
