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

"""Mask dev-eval (monitor) logic gates and watch/select idempotency guards (issue #59 / ticket 09).

The retired mask dev-eval entry's built-in self-test checks, promoted into
declarative pytest functions against the new home
``ctmr.application.generation.mask.monitor`` plus the shared trend extractors
(``ctmr.application.generation.trends``) and the shared dev-eval engine in
``ctmr.application.shell`` (``CheckpointWatcher`` / ``EarlyStopRule`` /
``TrendLedger``). The alignment-parity gate pins the round-trip Dice condition
path onto the terminal-acceptance resampler: both callers enter the same RAS
direction world (ADR-0020 -- the pre-#314 x/y flip is retired on both sides),
so the two paths must stay bit-identical. The
watch-idempotence and select-idempotence gates below are the sidecar-restart
contract: a restart must not re-evaluate already-scored epochs (re-appended
trend points would corrupt the early-stop patience count), and the final
selection must be stable under a ledger reload.

Torch-level (imports torch at module level), so the module is torch-marked and
runs for real in the CI full-dependency tier (ADR-0015 §6).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ctmr.application.generation.mask.monitor import (
    COMBINED_TO_INSTRUMENT,
    CohortSpacingSource,
    ConditionMaskSource,
    DevList,
    L2PostScore,
    RoundTripDice,
)
from ctmr.application.generation.trend import DevCohortBuilder
from ctmr.application.shell import COHORT_QUOTAS, CheckpointWatcher, EarlyStopRule, TrendLedger

pytestmark = pytest.mark.torch


def _write_mask_list(workdir):
    """The synthetic p2_mask_cond source: dev (fold=0) + train (fold=1) entries."""
    src_entries = []
    for challenge, quota in COHORT_QUOTAS.items():
        for index in range(quota + 2):
            src_entries.append(
                {
                    "image": f"embeddings/{challenge}/FIX{challenge}-{index:04d}-000-t1n_emb.nii.gz",
                    "label": f"labels/{challenge}/FIX{challenge}-{index:04d}-000/FIX{challenge}-{index:04d}-000-combined.nii.gz",
                    "spacing": [1.0, 1.0, 1.0],
                    "modality": "mri_t1_skull_stripped",
                    "fold": 0,
                    "sub": challenge,
                    "case": f"FIX{challenge}-{index:04d}-000",
                }
            )
    # one train-side (fold=1) entry per challenge must be dropped.
    for challenge in COHORT_QUOTAS:
        src_entries.append(
            {
                "image": f"embeddings/{challenge}/TRAIN{challenge}-000-t1n_emb.nii.gz",
                "label": f"labels/{challenge}/TRAIN{challenge}-000/TRAIN{challenge}-000-combined.nii.gz",
                "spacing": [1.0, 1.0, 1.0],
                "modality": "mri_t1_skull_stripped",
                "fold": 1,
                "sub": challenge,
                "case": f"TRAIN{challenge}-000",
            }
        )
    src = workdir / "p2_src.json"
    src.write_text(json.dumps({"training": src_entries}))
    return src


# ------------------------------------------------------------------- dev view / cohort


def test_dev_list_keeps_fold_zero_only_and_derives_the_raw_image(tmp_path):
    out = DevList(_write_mask_list(tmp_path), tmp_path).build()
    entries = json.loads(out.read_text())["training"]
    total_dev = sum(quota + 2 for quota in COHORT_QUOTAS.values())
    assert len(entries) == total_dev  # fold=1 not kept
    kept_cases = {entry["case"] for entry in entries}
    assert not any(case.startswith("TRAIN") for case in kept_cases)  # no train-side leak
    first = entries[0]["image"]
    assert first.endswith("-t1n.nii.gz") and "_emb" not in first and not first.startswith("raw/")


def test_dev_list_build_is_idempotent(tmp_path):
    first = DevList(_write_mask_list(tmp_path), tmp_path).build()
    second = DevList(tmp_path / "unrelated.json", tmp_path).build()
    assert first == second


def test_dev_cohort_covers_every_challenge_under_the_quotas(tmp_path):
    out = DevList(_write_mask_list(tmp_path), tmp_path).build()
    cohort = DevCohortBuilder(out).build()
    assert cohort
    assert {item["sub"] for item in cohort} == set(COHORT_QUOTAS)
    assert DevCohortBuilder(out).build() == cohort  # deterministic order


def test_spacing_and_mask_sources_resolve_per_case(tmp_path):
    out = DevList(_write_mask_list(tmp_path), tmp_path).build()
    spacings = CohortSpacingSource(out)
    masks = ConditionMaskSource(out, tmp_path / "phase")
    entry = json.loads(out.read_text())["training"][0]
    assert spacings.spacing_of(entry["case"]) == [1.0, 1.0, 1.0]
    assert masks.path_of(entry["case"]) == tmp_path / "phase" / entry["label"]


# --------------------------------------------------- combined->instrument remap and Dice


def test_combined_to_instrument_remap_is_value_frozen():
    condition = np.array([[[0, 22], [129, 130], [131, 0]]], dtype=np.int16)
    remapped = RoundTripDice.remap_combined_to_instrument(condition)
    expected = np.array([[[0, 0], [1, 2], [3, 0]]], dtype=np.int16)
    assert np.array_equal(remapped, expected)


def test_round_trip_dice_perfect_match_is_one_and_both_empty_is_none():
    remapped = np.array([[[0, 0], [1, 2], [3, 0]]], dtype=np.int16)
    pred = np.array([[[0, 0], [1, 2], [3, 0]]], dtype=np.uint8)
    assert RoundTripDice.dice(pred, remapped, "WT") == 1.0
    empty = RoundTripDice.dice(np.zeros_like(pred), np.zeros_like(remapped), "ET")
    assert empty is None  # both-empty dice must be None, not 0 (spec #51 decision 11)


# ------------------------------------------------------------- alignment parity gate


def test_round_trip_condition_alignment_matches_the_terminal_acceptance_resampler(tmp_path):
    """The monitor's condition alignment must track the L2 final-acceptance path:
    both callers unify onto the RAS direction world (ADR-0020) and align with
    the nearest-neighbour label adapter -- one array world, bit-identical paths."""
    import nibabel as nib

    from ctmr.application.acceptance.distribution.measurement_run import InstrumentInputAssembler

    # a small labelled volume, written on the DM-side RAS-ish grid then upsampled by
    # nibabel to a realistic resolution so the instrument-grid resample has work to do
    labels = np.zeros((24, 26, 28), dtype=np.uint8)
    labels[6:12, 8:14, 10:16] = 129
    labels[12:18, 10:16, 12:20] = 130
    path = tmp_path / "case-combined.nii.gz"
    nib.save(nib.Nifti1Image(labels, np.diag([2.0, 2.0, 2.0, 1.0])), str(path))

    from ctmr.application.generation.mask.monitor import PREDICTION_SHAPE

    reference = InstrumentInputAssembler().label_to_grid(str(path))
    produced = RoundTripDice(None).align_condition(path)  # the monitor applies its remap on top
    assert reference is not None and produced is not None
    assert produced.shape == PREDICTION_SHAPE == reference.shape
    # undo the monitor-side remap on the reference so the two alignment paths compare directly
    reference_remapped = np.zeros_like(reference)
    for source, destination in COMBINED_TO_INSTRUMENT.items():
        reference_remapped[reference == source] = destination
    assert np.array_equal(produced, reference_remapped)


# --------------------------------------------------- early-stop rule and selection (min FID)


def test_min_direction_rule_does_not_stop_an_improving_trend():
    rule = EarlyStopRule(patience=3, min_epoch=30, max_epoch=100)
    improving = [{"epoch": e, "m": 1.0 - 0.01 * e} for e in (5, 10, 15, 20, 25, 30)]
    stop, _ = rule.should_stop(improving)
    assert not stop


def test_min_direction_rule_stops_after_the_patience_plateau():
    rule = EarlyStopRule(patience=3, min_epoch=30, max_epoch=100)
    improving = [{"epoch": e, "m": 1.0 - 0.01 * e} for e in (5, 10, 15, 20, 25, 30)]
    plateau = improving + [{"epoch": e, "m": 0.75} for e in (35, 40, 45)]
    stop, reason = rule.should_stop(plateau)
    assert stop and "no new best" in reason
    short = improving + [{"epoch": 35, "m": 0.75}, {"epoch": 40, "m": 0.75}]
    stop, _ = rule.should_stop(short)
    assert not stop  # patience not exhausted


def test_selection_picks_the_argmin_mean_fid_epoch():
    selection = EarlyStopRule.selection([{"epoch": 5, "m": 1.2}, {"epoch": 10, "m": 0.8}, {"epoch": 20, "m": 0.8}])
    assert selection["epoch"] == 10  # ties break toward the earlier epoch


# ----------------------------------------------------------------- watch idempotence


def test_watcher_restart_does_not_re_evaluate_scored_epochs(tmp_path):
    ledger = TrendLedger(tmp_path / "eval")
    ledger.append({"epoch": 5, "m": 1.0, "checkpoint": "epoch_5.pt"})
    ledger.append({"epoch": 10, "m": 0.9, "checkpoint": "epoch_10.pt"})
    (tmp_path / "ckpt").mkdir()
    for epoch in (5, 10, 15):
        (tmp_path / "ckpt" / f"epoch_{epoch}.pt").write_bytes(b"ckpt")

    watcher = CheckpointWatcher(tmp_path / "ckpt", 5, 100, {r["epoch"] for r in ledger.read()})
    assert [epoch for epoch, _ in watcher.pending()] == [15]  # 5/10 already scored


# --------------------------------------------------- watch engine post-score extension (issue #225)


class _ScriptedL2:
    def run(self, samples, cohort, work_dir):
        return {"ok": True}


class _FailingRoundTrip:
    def run(self, predictions_root, cohort):
        raise RuntimeError("pred missing")


def test_l2_post_score_skip_degrades_to_none_fields(tmp_path):
    assert L2PostScore(_ScriptedL2(), RoundTripDice(None), [], skip=True)(5, [], tmp_path) == {
        "l2_trend": None,
        "round_trip_dice": None,
    }


def test_l2_post_score_keeps_the_l2_trend_when_only_the_round_trip_fails(tmp_path):
    # the pre-#225 loop shared one try across both trends: a round-trip hiccup
    # records None for the dice but keeps the already-measured instrument trend
    extension = L2PostScore(_ScriptedL2(), _FailingRoundTrip(), [], skip=False)
    assert extension(5, [], tmp_path) == {"l2_trend": {"ok": True}, "round_trip_dice": None}


def test_select_is_stable_under_a_ledger_reload(tmp_path):
    ledger = TrendLedger(tmp_path)
    trend = [{"epoch": e, "m": m, "checkpoint": f"epoch_{e}.pt"} for e, m in ((5, 1.2), (10, 0.8), (15, 0.8))]
    for record in trend:
        ledger.append(record)
    assert ledger.read() == trend
    first = EarlyStopRule.selection(ledger.read())
    second = EarlyStopRule.selection(ledger.read())
    assert first == second


# -------------------------------------------------------------- frozen instrument tokens


def test_condition_remap_table_matches_the_terminal_acceptance_table():
    from ctmr.application.acceptance.distribution.measurement_run import COMBINED_TO_INSTRUMENT as ACCEPTANCE_TABLE

    assert COMBINED_TO_INSTRUMENT == ACCEPTANCE_TABLE
