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

"""Diagnostic job C (issue #208, parent #205): t1c intensity-domain discrimination,
observed as pytest.

The L1 encoding-domain hypothesis (RC chain, parent #205): the training pipeline
normalises t1c with ``ScaleIntensityRangePercentilesd(0, 99.5 -> 0, 1, clip=False)``
so the brightest tumour voxels extrapolate above 1.0 -- whether the frozen VAE
reconstructs that extrapolated band faithfully is the one unmeasured link. Job C
produces the three readings: (1) real / VAE-reconstruction / generated tiered
top statistics (intra-tumour vs whole brain, P99 / P99.9 / top-0.5% mean), (2)
VAE reconstruction MAE split by input >1.0 vs within [0, 1] voxels, with a
clip=True normalisation arm as the direct clip-policy verdict, and (3) the
share of generated int16 voxels above 1000 (the >1.0 output-domain probe).

Statistics tests run on synthetic numpy volumes with hand-computed
expectations; the protocol-parity test (marked torch) pins the job's
normalisation and resize to the MONAI transforms the training pipeline itself
uses; the end-to-end test drives ``main`` over synthetic NIfTI fixtures with an
injected fake reconstructor -- no checkpoint, no GPU.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ctmr.application.acceptance.distribution.et_discrimination import DIAGNOSTIC_SEED_BASE as SIBLING_SEED_BASE
from ctmr.application.acceptance.distribution.final_acceptance import CHALLENGE_SEED_OFFSET, GLOBAL_SEED, ClusterBootstrap
from ctmr.application.acceptance.distribution.intensity_domain import (
    DIAGNOSTIC_SEED_BASE,
    GEN_SHAPE_ZYX,
    JOB_C_SEED_SLOT,
    EmbPool,
    GenPool,
    IntensityDomainReport,
    TieredIntensityStats,
    TrainingPreprocessing,
)

# ------------------------------------------------------------------ normalisation arms


def test_normalize_percentile_noclip_extrapolates_above_one():
    """Hand-computed 3-voxel volume: percentile 99.5 of [0, .5, 1] is .995, so the
    maximum maps to 1/.995 > 1 -- the extrapolated training input the job probes."""
    norm, lower, upper = TrainingPreprocessing.normalize_percentile(np.array([[0.0, 0.5, 1.0]]), clip=False)
    assert (lower, upper) == (0.0, pytest.approx(0.995))
    assert norm[0, 1] == pytest.approx(0.5 / 0.995)
    assert norm[0, 2] == pytest.approx(1.0 / 0.995)
    assert norm[0, 2] > 1.0


def test_normalize_percentile_clip_caps_the_same_affine_at_one():
    norm, _lower, upper = TrainingPreprocessing.normalize_percentile(np.array([[0.0, 0.5, 1.0]]), clip=True)
    assert upper == pytest.approx(0.995)
    assert norm[0, 2] == 1.0  # the clip arm keeps the affine, truncates the tail
    assert norm[0, 1] == pytest.approx(0.5 / 0.995)


def test_normalize_percentile_constant_volume_raises():
    with pytest.raises(ValueError):
        TrainingPreprocessing.normalize_percentile(np.zeros((2, 2, 2)), clip=False)


# --------------------------------------------------------------------- tiered top stats


def test_top_stats_hand_computed_on_a_thousand_voxel_volume():
    values = np.arange(1000, dtype=float)
    stats = TieredIntensityStats.top_stats(values)
    assert stats["n"] == 1000
    assert stats["p99"] == pytest.approx(989.01)  # np.quantile linear rule
    assert stats["p99_9"] == pytest.approx(998.001)  # 0.999 * (n-1) = 998.001, linear rule
    # top 0.5% of 1000 voxels = ceil(5) = 5 brightest -> mean of 995..999
    assert stats["top05_mean"] == pytest.approx(997.0)


def test_top_stats_small_volume_keeps_at_least_one_voxel_in_the_top_band():
    stats = TieredIntensityStats.top_stats(np.array([1.0, 2.0]))
    assert stats["n"] == 2
    assert stats["top05_mean"] == pytest.approx(2.0)  # ceil(0.01)=1 brightest voxel


def test_top_stats_empty_is_null_not_an_error():
    assert TieredIntensityStats.top_stats(np.array([])) == {"n": 0, "p99": None, "p99_9": None, "top05_mean": None}


# ---------------------------------------------------------------------- conditioned MAE


def test_conditioned_mae_splits_voxels_by_the_input_domain():
    input_vol = np.array([0.5, 0.5, 1.5, 1.5])
    recon = np.array([0.6, 0.4, 1.0, 2.0])
    reading = TieredIntensityStats.conditioned_mae(input_vol, recon, hi_mask=input_vol > 1.0, lo_mask=(input_vol >= 0.0) & (input_vol <= 1.0))
    assert reading["mae_over"] == pytest.approx(0.5)  # |1.0-1.5|, |2.0-1.5|
    assert reading["mae_within"] == pytest.approx(0.1)
    assert reading["n_over"] == 2
    assert reading["n_within"] == 2


def test_conditioned_mae_accepts_masks_shared_across_arms():
    """The clip=True arm borrows the noclip arm's tier masks -- that shared
    voxel support is what makes the two arms directly comparable."""
    input_vol = np.array([0.5, 1.5])
    reading = TieredIntensityStats.conditioned_mae(input_vol, input_vol, hi_mask=np.array([False, True]), lo_mask=np.array([True, False]))
    assert reading["mae_over"] == pytest.approx(0.0)
    assert reading["n_over"] == 1


def test_conditioned_mae_empty_tier_is_null_not_zero():
    reading = TieredIntensityStats.conditioned_mae(np.array([0.5]), np.array([0.5]), hi_mask=np.array([False]), lo_mask=np.array([True]))
    assert reading["mae_over"] is None
    assert reading["n_over"] == 0
    assert reading["mae_within"] == pytest.approx(0.0)


# ------------------------------------------------------------------- over-threshold count


def test_over_threshold_counts_split_by_tier_masks():
    volume = np.array([[500.0, 1500.0], [0.0, 2000.0]])
    tiers = {
        "brain": volume > 0,
        "tumour": np.array([[True, True], [False, False]]),
        "et": np.array([[False, True], [False, False]]),
    }
    reading = TieredIntensityStats.over_threshold_counts(volume, threshold=1000.0, tier_masks=tiers)
    assert reading["brain"] == {"k": 2, "n": 3}
    assert reading["tumour"] == {"k": 1, "n": 2}
    assert reading["et"] == {"k": 1, "n": 1}


def test_over_threshold_counts_empty_tier_is_zero_over_zero():
    volume = np.array([500.0, 1500.0])
    reading = TieredIntensityStats.over_threshold_counts(volume, threshold=1000.0, tier_masks={"brain": volume > 0, "et": np.zeros(2, bool)})
    assert reading["et"] == {"k": 0, "n": 0}


# --------------------------------------------------------------------- histogram & align


def test_histogram_counts_use_fixed_edges_with_one_overflow_bucket():
    counts, overflow = TieredIntensityStats.histogram_counts(np.array([0.5, 1.5, 2.5, 0.0]), edges=np.array([0.0, 1.0, 2.0]))
    assert list(counts) == [2, 1]  # 0.0 and 0.5 in the first bin, 1.5 in the second
    assert overflow == 1  # 2.5 lands in the overflow bucket, never silently dropped


def test_align_pred_to_generated_grid_is_the_center_crop_pad_inverse():
    """The instrument centred the 256x256x128 generated volume onto 240x240x155
    (crop 8 on x/y, pad 13 in front on z); the tumour mask rides the inverse
    array mapping back onto the generated grid."""
    pred = np.zeros((155, 240, 240), dtype=bool)
    pred[13, 0, 0] = True  # instrument z 13 sits on generated z 0 (pad front = 13)
    pred[30, 20, 10] = True  # mid-volume anchor
    pred[140, 5, 5] = True  # near the generated top: z 140 -> generated z 127
    mask = GenPool.align_pred_to_generated_grid(pred)
    assert mask.shape == GEN_SHAPE_ZYX  # (128, 256, 256)
    assert mask[0, 8, 8]
    assert mask[17, 28, 18]
    assert mask[127, 13, 13]


def test_align_pred_to_generated_grid_leaves_out_of_domain_false():
    pred = np.zeros((155, 240, 240), dtype=bool)
    pred[0, 0, 0] = True  # maps to generated z -13: no such slice
    pred[150, 100, 100] = True  # generated z 137: beyond the 128-slice grid
    assert not GenPool.align_pred_to_generated_grid(pred).any()


# ----------------------------------------------------------------------- distributions


def test_distribution_stats_quantiles_follow_the_cluster_bootstrap_linear_rule():
    values = [0.0, 2.0, 4.0, 8.0, 16.0]
    stats = TieredIntensityStats.distribution_stats(values)
    assert stats["median"] == pytest.approx(ClusterBootstrap.quantile(values, 0.5))
    assert stats["q05"] == pytest.approx(ClusterBootstrap.quantile(values, 0.05))
    assert stats["q95"] == pytest.approx(ClusterBootstrap.quantile(values, 0.95))
    assert stats["mean"] == pytest.approx(6.0)
    assert stats["n_cases"] == 5


def test_distribution_stats_empty_is_null():
    assert TieredIntensityStats.distribution_stats([]) == {"median": None, "mean": None, "q05": None, "q95": None, "n_cases": 0}


# ------------------------------------------------------------------------- seed hygiene


def test_diagnostic_seed_slot_stays_clear_of_the_formal_chain_and_sibling_jobs():
    """Job A occupies slots 0/1 and 100/101, job B slot 200; job C takes the
    next free block so no two diagnostic quantities ever draw one seed."""
    assert JOB_C_SEED_SLOT == 300
    seed = DIAGNOSTIC_SEED_BASE + CHALLENGE_SEED_OFFSET["GLI"] * 1000 + JOB_C_SEED_SLOT
    assert seed >= DIAGNOSTIC_SEED_BASE
    assert GLOBAL_SEED < DIAGNOSTIC_SEED_BASE == SIBLING_SEED_BASE  # one shared diagnostic namespace


# ------------------------------------------------------------------------ pools (fake IO)


class _FakeRepo:
    """Synthetic case repository: volumes are tiny numpy arrays keyed by case."""

    def __init__(self, train_cases=None, gen_cases=None):
        self._train = train_cases or {}
        self._gen = gen_cases or {}

    def train_case(self, entry):
        return self._train.get(entry["case"])

    def gen_case(self, entry):
        data = self._gen.get(entry["case"])
        if data is None:
            return None
        return {"case": entry["case"], "challenge": entry["challenge"], **data}


class _FakeRecon:
    """Deterministic fake VAE in the job's latent layout (channels last): encode
    appends the channel axis, decode takes channel 0 and scales it -- enough to
    pin the arm wiring and the MAE arithmetic without a network."""

    def __init__(self, recon_scale=1.0):
        self._scale = recon_scale

    def decode(self, latent):
        return latent[..., 0] * self._scale

    def encode(self, image):
        return image[..., None]


def _identity_resize(volume, target_shape, mode):
    return volume


def test_emb_pool_reads_both_arms_and_splits_mae_by_input_domain():
    t1c = np.array([[[0.0, 0.5], [1.0, 2.0]]], dtype=float)
    seg = np.array([[[0, 3], [1, 2]]], dtype=np.uint8)
    cases = {
        "c1": {"t1c": t1c, "seg": seg, "emb": np.full((1, 2, 2, 4), 0.5)},
        "c2": {"t1c": t1c, "seg": seg, "emb": None},  # missing embedding -> excluded
    }
    repo = _FakeRepo(train_cases=cases)
    recon = _FakeRecon(recon_scale=1.0)  # identity decode: recon == latent == 0.5 everywhere
    rows = EmbPool().read_cases([{"case": "c1", "sub": "GLI"}, {"case": "c2", "sub": "GLI"}], repo, recon, _identity_resize)
    assert len(rows) == 2

    ok = rows[0]
    assert ok["excluded"] is None
    assert ok["real_native"]["tumour"]["p99"] is not None
    assert ok["recon"]["tumour"]["p99"] is not None
    # input noclip on [0,.5,1,2]: percentile 99.5 = 1.985 (0.995*(n-1)=2.985, between
    # the 1 and 2 voxels), so exactly one voxel (2/1.985 > 1) lands above 1.0; the
    # identity decode keeps a 0.5 offset everywhere.
    mae = ok["mae"]
    assert mae["upper_noclip"] == pytest.approx(1.985)
    assert mae["n_over"] == 1
    assert mae["n_within"] == 3
    assert mae["noclip_over"] == pytest.approx(2.0 / 1.985 - 0.5, abs=1e-12)
    assert mae["noclip_within"] == pytest.approx((0.5 + 0.5 / 1.985) / 3, abs=1e-12)
    # the clip arm reconstructs its own (capped) input exactly under the identity fake
    assert mae["clip_over"] == pytest.approx(0.0, abs=1e-12)
    assert mae["clip_within"] == pytest.approx(0.0, abs=1e-12)

    missing = rows[1]
    assert missing["excluded"] == "missing_embedding"
    assert missing["mae"] is None


def test_gen_pool_counts_over_1000_and_tiers_generation_statistics():
    gen = np.array([[[0, 1500], [500, 1200]]], dtype=np.int16)
    pred = np.array([[[0, 3], [0, 0]]], dtype=np.uint8)  # ET on the 1500 voxel (already aligned)
    real = np.array([[[0.0, 1.2], [0.3, 0.9]]])
    seg = np.array([[[0, 3], [2, 1]]], dtype=np.uint8)
    cases = {"g1": {"gen": gen, "pred": pred, "real": real, "seg": seg}}
    rows = GenPool().read_cases([{"case": "g1", "challenge": "GLI"}], _FakeRepo(gen_cases=cases), align=lambda mask: mask)
    assert len(rows) == 1
    row = rows[0]
    assert row["excluded"] is None
    over = row["over_1000"]
    assert over["brain"] == {"k": 2, "n": 3}  # 1500 and 1200 among the three non-zero voxels
    assert over["tumour"] == {"k": 1, "n": 1}  # the ET-labelled voxel is the 1500 one
    assert over["et"] == {"k": 1, "n": 1}
    assert row["gen"]["brain"]["n"] == 3
    assert row["gen"]["tumour"]["top05_mean"] == pytest.approx(1.5)  # 1500/1000
    assert row["real_native"]["brain"]["p99"] is not None


# ------------------------------------------------------------------------- report & CLI


def test_report_writes_a_skeleton_without_verdicts(tmp_path):
    report = IntensityDomainReport(train_list=Path("t.json"), samples=Path("s.json"), run_id=None)
    json_path, md_path = report.write([], [], tmp_path)
    payload = json.loads(Path(json_path).read_text())
    assert payload["schema"] == "intensity-domain-diagnostic/1"
    assert payload["variant"] == "diagnostic"
    assert payload["issue"] == 208
    assert "不产生任何验收判定" in payload["disclaimer"]
    assert "verdict" not in json.dumps(payload)
    assert "假说" in Path(md_path).read_text()


@pytest.mark.torch
def test_cli_end_to_end_writes_json_and_markdown(tmp_path):
    """Full-chain main() over synthetic NIfTI fixtures with a fake reconstructor:
    one healthy emb-pool case, one missing-embedding exclusion, one gen-pool case."""
    import nibabel as nib

    affine = np.eye(4)
    # emb-pool case 1: healthy (raw t1c + seg + training embedding)
    train_dir = tmp_path / "raw" / "ASNR" / "BraTS-GLI-00001-000"
    train_dir.mkdir(parents=True)
    rng = np.random.default_rng(20260829)
    t1c = rng.uniform(0, 1500, size=(8, 8, 8)).astype(np.float32)
    t1c[:2] = 0.0
    seg = np.zeros((8, 8, 8), dtype=np.uint8)
    seg[4:6, 4:6, 4:6] = 3
    nib.save(nib.Nifti1Image(t1c, affine=affine), train_dir / "BraTS-GLI-00001-000-t1c.nii.gz")
    nib.save(nib.Nifti1Image(seg, affine=affine), train_dir / "BraTS-GLI-00001-000-seg.nii.gz")
    emb_dir = tmp_path / "emb" / "ASNR" / "BraTS-GLI-00001-000"
    emb_dir.mkdir(parents=True)
    nib.save(nib.Nifti1Image(rng.normal(size=(8, 8, 8, 4)).astype(np.float32), affine=affine), emb_dir / "BraTS-GLI-00001-000-t1c_emb.nii.gz")
    # emb-pool case 2: raw + seg but no embedding on disk -> excluded row
    train_dir2 = tmp_path / "raw" / "ASNR" / "BraTS-GLI-00002-000"
    train_dir2.mkdir(parents=True)
    nib.save(nib.Nifti1Image(t1c, affine=affine), train_dir2 / "BraTS-GLI-00002-000-t1c.nii.gz")
    nib.save(nib.Nifti1Image(seg, affine=affine), train_dir2 / "BraTS-GLI-00002-000-seg.nii.gz")

    # gen-pool case: generated int16 + instrument prediction + real companion (L2 layout)
    gen_dir = tmp_path / "gen"
    gen_dir.mkdir()
    gen = rng.uniform(0, 1400, size=(8, 8, 8)).astype(np.int16)
    gen[:2] = 0
    nib.save(nib.Nifti1Image(gen, affine=affine), gen_dir / "g1_t1c.nii.gz")
    pred = np.zeros((8, 8, 8), dtype=np.uint8)
    pred[0, 0, 0] = 3
    pred_dir = tmp_path / "pred" / "GLI"
    pred_dir.mkdir(parents=True)
    nib.save(nib.Nifti1Image(pred, affine=affine), pred_dir / "g1__gen.nii.gz")
    real_dir = tmp_path / "holdout_real" / "GLI" / "g1"
    real_dir.mkdir(parents=True)
    nib.save(nib.Nifti1Image(t1c, affine=affine), real_dir / "g1-t1c.nii.gz")
    nib.save(nib.Nifti1Image(seg, affine=affine), real_dir / "g1-seg.nii.gz")

    train_list = tmp_path / "train.json"
    train_list.write_text(
        json.dumps(
            {
                "training": [
                    {
                        "image": "ASNR/BraTS-GLI-00001-000/BraTS-GLI-00001-000-t1c.nii.gz",
                        "sub": "GLI",
                        "case": "BraTS-GLI-00001-000",
                        "modality": "mri_t1c_skull_stripped",
                    },
                    {
                        "image": "ASNR/BraTS-GLI-00002-000/BraTS-GLI-00002-000-t1c.nii.gz",
                        "sub": "GLI",
                        "case": "BraTS-GLI-00002-000",
                        "modality": "mri_t1c_skull_stripped",
                    },
                ]
            }
        )
    )
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps([{"case_id": "g1", "challenge": "GLI", "samples": {"t1c": {"path": str(gen_dir / "g1_t1c.nii.gz"), "seed": 1}}}]))

    out = tmp_path / "diag"
    from ctmr.application.acceptance.distribution.intensity_domain import main

    rc = main(
        [
            "--train-list",
            str(train_list),
            "--data-root",
            str(tmp_path / "raw"),
            "--emb-root",
            str(tmp_path / "emb"),
            "--samples",
            str(samples),
            "--real-root",
            str(tmp_path / "holdout_real"),
            "--pred-root",
            str(tmp_path / "pred"),
            "-e",
            "env.json",
            "-c",
            "mc.json",
            "-t",
            "md.json",
            "--scale-factor-path",
            "base.pt",
            "--output-dir",
            str(out),
            "--run-id",
            "p1-20260822T131947Z",
        ],
        reconstructor_factory=lambda *a, **k: _FakeRecon(),
        grid=(8, 8, 8),  # the fixtures' training grid (a real run uses 256x256x128)
        align=lambda mask: mask.astype(bool),  # fixtures are already on the generated grid
    )
    assert rc == 0
    payload = json.loads((out / "intensity_domain_diagnostic.json").read_text())
    assert payload["run_id"] == "p1-20260822T131947Z"
    assert [row["case"] for row in payload["emb_pool"]["per_case"]] == ["BraTS-GLI-00001-000", "BraTS-GLI-00002-000"]
    assert payload["emb_pool"]["aggregate"]["real_native"]["brain"]["p99"]["n_cases"] == 1  # only the healthy case
    assert payload["emb_pool"]["aggregate"]["mae"]["noclip_over"]["n_cases"] == 1
    assert payload["gen_pool"]["per_challenge"]["GLI"]["over_1000"]["brain"]["k"] >= 0
    assert payload["hypothesis_over_1000"]["totals"]["brain"]["n"] >= 1
    md = (out / "intensity_domain_diagnostic.md").read_text()
    assert "variant: diagnostic" in md
    assert ">1000" in md
    assert "missing_embedding" in md  # excluded rows stay visible in the detail table


# ------------------------------------------------------------------------- protocol parity


@pytest.mark.torch
def test_normalize_and_resize_match_the_training_monai_transforms():
    """Protocol parity: the job's arms must reuse the training pipeline's own
    MONAI transforms, not a reimplementation (clip=False is the training arm)."""
    from monai.transforms import Resized, ScaleIntensityRangePercentilesd

    rng = np.random.default_rng(7)
    volume = rng.uniform(0, 2, size=(10, 12, 14)).astype(np.float32)

    d_noclip = ScaleIntensityRangePercentilesd(keys="image", lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False)
    norm, _lo, _up = TrainingPreprocessing.normalize_percentile(volume, clip=False)
    assert np.allclose(norm, d_noclip({"image": volume})["image"].numpy(), atol=1e-9)

    d_clip = ScaleIntensityRangePercentilesd(keys="image", lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=True)
    norm_clip, _lo, _up = TrainingPreprocessing.normalize_percentile(volume, clip=True)
    assert np.allclose(norm_clip, d_clip({"image": volume})["image"].numpy(), atol=1e-9)

    d_resize = Resized(keys="image", spatial_size=(8, 8, 8), mode="trilinear")
    expected = d_resize({"image": volume[None]})["image"].numpy()[0]  # channel-first like create_transforms
    assert np.allclose(TrainingPreprocessing.resize_image(volume, (8, 8, 8), "trilinear"), expected, atol=1e-6)
