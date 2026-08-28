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

"""Cross-modal dev-eval (monitor) logic gates and watch/select idempotency guards (issue #61 / ticket 08).

The retired cross-modal dev-eval entry's built-in self-test checks, promoted
into declarative pytest functions against the new home
``ctmr.application.generation.cross_modal.monitor`` plus the shared dev-eval engine
in ``ctmr.application.shell`` (``CheckpointWatcher`` / ``EarlyStopRule`` /
``TrendLedger``). The watch-idempotence and select-idempotence gates below are
acceptance criterion 3: a sidecar restart must not re-evaluate already-scored
epochs (re-appended trend points would corrupt the early-stop patience count), and
the final selection must be stable under a ledger reload.

Torch-level (imports torch/nibabel/skimage at module level), so the module is
torch-marked and runs for real in the CI full-dependency tier (ADR-0015 §6).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ctmr.application.generation.cross_modal.monitor import (
    PSNR_CAP_DB,
    DevCohort,
    DevList,
    PairwiseScorer,
    read_src_latent,
)
from ctmr.application.generation.cross_modal.plan import MODALITY_PAIRS
from ctmr.application.shell import COHORT_QUOTAS, CheckpointWatcher, EarlyStopRule, TrendLedger

pytestmark = pytest.mark.torch

N_DEV_CASES = sum(COHORT_QUOTAS.values())


def _write_src_list(workdir):
    """The synthetic p3_pairs source list: 12 ordered src->tgt entries per dev case."""
    src_entries = []
    for challenge, quota in COHORT_QUOTAS.items():
        for index in range(quota):
            case = f"FIX{challenge}-{index:04d}-000"
            for src in ("t1n", "t1c", "t2w", "t2f"):
                for tgt in ("t1n", "t1c", "t2w", "t2f"):
                    if src == tgt:
                        continue
                    src_entries.append(
                        {
                            "image": f"embeddings/{challenge}/{case}/{case}-{tgt}_emb.nii.gz",
                            "src_image": f"embeddings/{challenge}/{case}/{case}-{src}_emb.nii.gz",
                            "label": f"labels/{challenge}/{case}/{case}-tumor129.nii.gz",
                            "spacing": [1.0, 1.0, 1.0],
                            "modality": MODALITY_PAIRS[tgt][0],
                            "src_modality": MODALITY_PAIRS[src][0],
                            "fold": 0,
                            "sub": challenge,
                            "case": case,
                        }
                    )
    src = workdir / "p3_src.json"
    src.write_text(json.dumps({"training": src_entries}))
    return src


# ------------------------------------------------------------------- dev view / cohort


def test_dev_list_keeps_twelve_pairs_per_case_and_derives_raw_tgt(tmp_path):
    out = DevList(_write_src_list(tmp_path), tmp_path).build()
    entries = json.loads(out.read_text())["training"]
    assert len(entries) == N_DEV_CASES * 12
    assert entries[0]["image"].endswith("-t1c.nii.gz")  # the first ordered pair is t1n->t1c
    assert "_emb" not in entries[0]["image"]  # raw tgt derived from the embedding path
    assert "src_image" in entries[0]  # the src condition survives the dev view


def test_dev_cohort_resolves_each_case_and_every_src_suffix(tmp_path):
    out = DevList(_write_src_list(tmp_path), tmp_path).build()
    cohort_source = DevCohort(out)
    cohort = cohort_source.cases()
    assert len(cohort) == N_DEV_CASES
    assert {item["sub"] for item in cohort} == set(COHORT_QUOTAS)
    # list fields carry the long mapping keys (mri_*); the lookups must translate the BraTS suffixes
    for suffix in MODALITY_PAIRS:
        assert cohort_source.src_image_of(cohort[0]["case"], suffix) is not None


def test_read_src_latent_reads_the_channel_axis(tmp_path):
    import nibabel as nib

    latent = np.zeros((32, 32, 16, 4), dtype=np.float32)
    latent[..., 0] = 1.0
    latent_path = tmp_path / "latent.nii.gz"
    nib.save(nib.Nifti1Image(latent, np.eye(4)), str(latent_path))
    tensor = read_src_latent(latent_path, torch.device("cpu"))
    assert tuple(tensor.shape) == (1, 4, 32, 32, 16)  # (H,W,D,C) -> (1,C,H,W,D)
    assert float(tensor[0, 0].mean()) == 1.0  # channel 0 is the brain-modality slot


# ------------------------------------------------------------- early-stop rule (max direction)


def test_max_direction_rule_does_not_stop_an_improving_trend():
    rule = EarlyStopRule(patience=3, min_epoch=30, max_epoch=100, direction="max")
    improving = [{"epoch": e, "m": 0.1 + 0.01 * e} for e in (5, 10, 15, 20, 25, 30)]
    stop, _reason = rule.should_stop(improving)
    assert not stop


def test_max_direction_rule_stops_a_patience_plateau():
    rule = EarlyStopRule(patience=3, min_epoch=30, max_epoch=100, direction="max")
    improving = [{"epoch": e, "m": 0.1 + 0.01 * e} for e in (5, 10, 15, 20, 25, 30)]
    plateau = improving + [{"epoch": e, "m": 0.4} for e in (35, 40, 45)]
    stop, _reason = rule.should_stop(plateau)
    assert stop


def test_max_direction_selection_picks_the_best_epoch():
    selection = EarlyStopRule.selection(
        [{"epoch": 5, "m": 1.2}, {"epoch": 10, "m": 0.8}, {"epoch": 20, "m": 0.8}], direction="max", metric_name="mean_ssim"
    )
    assert selection["epoch"] == 5
    assert selection["mean_ssim"] == 1.2


# --------------------------------------------------------------------- pairwise scorer


def test_pairwise_scorer_identical_volumes_score_perfect():
    rng = np.random.default_rng(0)
    reference = (50 + 400 * rng.random((32, 32, 32))).astype(np.int16)
    identical = PairwiseScorer.score_arrays(reference, reference)
    assert abs(identical["ssim"] - 1.0) <= 1e-6
    assert identical["psnr"] == PSNR_CAP_DB  # capped, not inf


def test_pairwise_scorer_ranks_degraded_below_identical():
    rng = np.random.default_rng(0)
    reference = (50 + 400 * rng.random((32, 32, 32))).astype(np.int16)
    identical = PairwiseScorer.score_arrays(reference, reference)
    degraded = PairwiseScorer.score_arrays(
        reference, np.clip((reference.astype(np.float64) + rng.normal(0, 80, reference.shape)).round(), 0, None).astype(np.int16)
    )
    assert degraded["ssim"] < identical["ssim"]
    assert degraded["psnr"] < identical["psnr"]
    shuffled = PairwiseScorer.score_arrays(reference, np.roll(reference, 8, axis=0))
    assert shuffled["ssim"] <= 0.9  # substantial structural degradation


# ------------------------------------------- watch idempotence (acceptance criterion 3)


def _touch_checkpoints(ckpt_dir, epochs):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for epoch in epochs:
        (ckpt_dir / f"epoch_{epoch}.pt").write_bytes(b"ckpt")


def test_watcher_only_offers_unevaluated_eval_epochs(tmp_path):
    _touch_checkpoints(tmp_path, epochs=[3, 5, 7, 10, 15, 105])
    watcher = CheckpointWatcher(tmp_path, eval_every=5, max_epoch=100)
    # 3 and 7 are off the eval grid; 105 exceeds the max epoch
    assert [epoch for epoch, _ in watcher.pending()] == [5, 10, 15]


def test_watcher_restart_does_not_reevaluate_seeded_done_epochs(tmp_path):
    _touch_checkpoints(tmp_path, epochs=[5, 10, 15, 20])
    first = CheckpointWatcher(tmp_path, eval_every=5, max_epoch=100)
    assert [epoch for epoch, _ in first.pending()] == [5, 10, 15, 20]
    first.mark_done(5)
    first.mark_done(10)
    # a sidecar restart re-seeds from the ledger's already-scored epochs
    restarted = CheckpointWatcher(tmp_path, eval_every=5, max_epoch=100, done_epochs=first._done)
    assert [epoch for epoch, _ in restarted.pending()] == [15, 20]  # 5/10 not re-evaluated


def test_watcher_pending_is_repeatable_until_marked_done(tmp_path):
    _touch_checkpoints(tmp_path, epochs=[5, 10])
    watcher = CheckpointWatcher(tmp_path, eval_every=5, max_epoch=100)
    assert [epoch for epoch, _ in watcher.pending()] == [5, 10]
    assert [epoch for epoch, _ in watcher.pending()] == [5, 10]  # polling is read-only
    watcher.mark_done(5)
    assert [epoch for epoch, _ in watcher.pending()] == [10]


# ------------------------------------------- select idempotence (acceptance criterion 3)


def test_ledger_append_read_roundtrip_preserves_trend_order(tmp_path):
    ledger = TrendLedger(tmp_path)
    for epoch, m in [(5, 0.5), (10, 0.7), (15, 0.7)]:
        ledger.append({"epoch": epoch, "m": m, "checkpoint": f"epoch_{epoch}.pt"})
    trend = ledger.read()
    assert [point["epoch"] for point in trend] == [5, 10, 15]
    assert [point["m"] for point in trend] == [0.5, 0.7, 0.7]


def test_selection_is_stable_across_a_ledger_reload(tmp_path):
    ledger = TrendLedger(tmp_path)
    for epoch, m in [(5, 0.6), (10, 0.9), (15, 0.7), (20, 0.7)]:
        ledger.append({"epoch": epoch, "m": m, "checkpoint": f"epoch_{epoch}.pt"})
    fresh = EarlyStopRule.selection(ledger.read(), direction="max", metric_name="mean_ssim")
    reloaded = EarlyStopRule.selection(TrendLedger(tmp_path).read(), direction="max", metric_name="mean_ssim")
    assert fresh == reloaded  # the final dev-side selection does not depend on process state
    assert fresh["epoch"] == 10
    assert fresh["mean_ssim"] == 0.9


def test_select_rerun_yields_the_identical_contract_apart_from_the_timestamp(tmp_path):
    """Re-running select over the same ledger reproduces the whole emitted contract;
    only the ``recorded_utc`` audit envelope may differ."""
    ledger = TrendLedger(tmp_path)
    for epoch, m in [(5, 0.6), (10, 0.9), (15, 0.7)]:
        ledger.append({"epoch": epoch, "m": m, "checkpoint": f"epoch_{epoch}.pt"})

    def select_once():
        trend = ledger.read()
        selection = EarlyStopRule.selection(trend, direction="max", metric_name="mean_ssim")
        best = next(point for point in trend if point["epoch"] == selection["epoch"] and point["m"] is not None)
        selection["mean_psnr"] = best.get("mean_psnr")
        selection["rule"] = "argmax mean dev 3D SSIM over eval points (pre-registered; PSNR recorded alongside)"
        selection["trend"] = trend
        selection["recorded_utc"] = "envelope-only"
        return json.dumps(selection, sort_keys=True)

    first = select_once()
    second = select_once()  # a second process replays select against the same on-disk state
    assert first == second
