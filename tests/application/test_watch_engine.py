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

"""Behaviour gates for the offline dev watch/select engine (issue #225, #279, ADR-0019 §5).

Fake sampler/scorer injections drive ``WatchEngine`` through the mechanical
sequence the shell owns -- one offline pass over the run's persisted
checkpoints, dedup, ledger append, record assembly, early-stop file write --
and ``SelectionEmitter`` through the select argmin/argmax contract; the event
sequences are pinned item by item against the pre-#225 monitor loops (the
first tests the watch wiring ever had).  The ``TrendLedger`` incremental read
is pinned equal to a full re-read under the append-only protocol.  Torch-level:
the shell module imports torch, so the module is torch-marked and runs for
real in the CI full-dependency tier (ADR-0015 §6).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ctmr.application.shell import STOP_FILE, EarlyStopRule, SelectionEmitter, TrendLedger, WatchEngine

pytestmark = pytest.mark.torch


class ScriptedSamplerFactory:
    """Fake sampler factory: records (checkpoint, out_dir) calls, returns the scripted samples."""

    def __init__(self, samples_by_epoch):
        self.calls = []
        self._samples_by_epoch = samples_by_epoch

    def __call__(self, checkpoint_path, out_dir):
        epoch = int(Path(checkpoint_path).stem.split("_")[1])
        self.calls.append(("sample", epoch, str(out_dir)))
        return self._samples_by_epoch[epoch]


class ScriptedScorer:
    """Fake scorer: records calls, returns the scripted (fields, log_line) per epoch.

    ``fail_once`` epochs raise on their first call and succeed on the retry --
    the skip-and-retry resilience the engine must preserve.
    """

    def __init__(self, score_by_epoch, fail_once=()):
        self.calls = []
        self._score_by_epoch = score_by_epoch
        self._fail_once = set(fail_once)
        self._failed = set()

    def __call__(self, samples):
        epoch = samples[0]["epoch"]
        self.calls.append(("score", epoch))
        if epoch in self._fail_once and epoch not in self._failed:
            self._failed.add(epoch)
            raise RuntimeError(f"score boom at {epoch}")
        return {"m": self._score_by_epoch[epoch]}, f"m={self._score_by_epoch[epoch]}"


def _touch_checkpoints(ckpt_dir, epochs):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for epoch in epochs:
        (ckpt_dir / f"epoch_{epoch}.pt").write_bytes(b"ckpt")


def _records(eval_root):
    return [json.loads(line) for line in (eval_root / "dev_trend.jsonl").read_text().splitlines() if line.strip()]


def _engine(ckpt_dir, eval_root, rule, sampler, scorer, post_score=None):
    return WatchEngine(
        ckpt_dir=ckpt_dir,
        eval_root=eval_root,
        eval_every=5,
        max_epoch=100,
        rule=rule,
        sampler_factory=sampler,
        scorer=scorer,
        post_score=post_score,
    )


def test_watch_engine_drives_the_mechanical_sequence_end_to_end(tmp_path):
    ckpt_dir, eval_root = tmp_path / "ckpt", tmp_path / "eval"
    _touch_checkpoints(ckpt_dir, [5, 10])
    sampler = ScriptedSamplerFactory({5: [{"epoch": 5}], 10: [{"epoch": 10}]})
    scorer = ScriptedScorer({5: 1.0, 10: 0.9})

    code = _engine(ckpt_dir, eval_root, EarlyStopRule(patience=2, min_epoch=0, max_epoch=100), sampler, scorer).run(cohort_file="dev_cohort.json")

    assert code == 0
    # the event sequence, item by item: ledger re-check -> sample -> score -> append -> trend.json
    assert sampler.calls == [
        ("sample", 5, str(eval_root / "epoch_5" / "samples")),
        ("sample", 10, str(eval_root / "epoch_10" / "samples")),
    ]
    assert scorer.calls == [("score", 5), ("score", 10)]
    records = _records(eval_root)
    assert [record["epoch"] for record in records] == [5, 10]
    # the record skeleton is the engine's: eval_utc/epoch/checkpoint open, cohort_file closes
    assert list(records[0]) == ["eval_utc", "epoch", "checkpoint", "m", "cohort_file"]
    assert records[0]["checkpoint"] == str(ckpt_dir / "epoch_5.pt")
    assert records[0]["cohort_file"] == "dev_cohort.json"
    assert json.loads((eval_root / "epoch_5" / "trend.json").read_text()) == records[0]
    assert not (ckpt_dir / STOP_FILE).exists()  # patience never exhausted


def test_watch_engine_dedupes_epochs_already_in_the_ledger(tmp_path):
    ckpt_dir, eval_root = tmp_path / "ckpt", tmp_path / "eval"
    _touch_checkpoints(ckpt_dir, [5, 10])
    ledger = TrendLedger(eval_root)
    ledger.append({"epoch": 5, "m": 1.0, "checkpoint": str(ckpt_dir / "epoch_5.pt")})
    sampler = ScriptedSamplerFactory({10: [{"epoch": 10}]})
    scorer = ScriptedScorer({10: 0.9})

    code = _engine(ckpt_dir, eval_root, EarlyStopRule(patience=2, min_epoch=0, max_epoch=100), sampler, scorer).run()

    assert code == 0
    assert sampler.calls == [("sample", 10, str(eval_root / "epoch_10" / "samples"))]  # 5 never re-evaluated
    assert [record["epoch"] for record in _records(eval_root)] == [5, 10]


def test_watch_engine_rechecks_the_ledger_before_each_point(tmp_path):
    """A point the ledger gains while the pass runs -- e.g. a still-live run's
    #278 embedded validation appending concurrently -- is the trainer's, not
    re-scored: the per-point re-check skips it."""
    ckpt_dir, eval_root = tmp_path / "ckpt", tmp_path / "eval"
    _touch_checkpoints(ckpt_dir, [5, 10])
    ledger = TrendLedger(eval_root)

    class LedgerGrowingSampler:
        """While evaluating epoch 5, a concurrent writer appends epoch 10."""

        def __init__(self, samples_by_epoch):
            self.calls = []
            self._samples_by_epoch = samples_by_epoch

        def __call__(self, checkpoint_path, out_dir):
            epoch = int(Path(checkpoint_path).stem.split("_")[1])
            self.calls.append(("sample", epoch))
            if epoch == 5:
                ledger.append({"epoch": 10, "m": 0.7, "checkpoint": str(ckpt_dir / "epoch_10.pt")})
            return self._samples_by_epoch[epoch]

    sampler = LedgerGrowingSampler({5: [{"epoch": 5}], 10: [{"epoch": 10}]})
    scorer = ScriptedScorer({5: 1.0})

    code = _engine(ckpt_dir, eval_root, EarlyStopRule(patience=2, min_epoch=0, max_epoch=100), sampler, scorer).run()

    assert code == 0
    assert sampler.calls == [("sample", 5)]  # epoch 10 was re-checked and skipped before its sample call
    assert scorer.calls == [("score", 5)]
    records = _records(eval_root)
    assert [record["epoch"] for record in records] == [10, 5]  # the concurrent point stays the trainer's; the watch only appended 5


def test_watch_engine_skips_a_failing_epoch_and_a_rerun_retries_it(tmp_path):
    ckpt_dir, eval_root = tmp_path / "ckpt", tmp_path / "eval"
    _touch_checkpoints(ckpt_dir, [5, 10])
    sampler = ScriptedSamplerFactory({5: [{"epoch": 5}], 10: [{"epoch": 10}]})
    scorer = ScriptedScorer({5: 1.0, 10: 0.9}, fail_once={10})
    engine = _engine(ckpt_dir, eval_root, EarlyStopRule(patience=2, min_epoch=0, max_epoch=100), sampler, scorer)

    assert engine.run() == 0
    assert [record["epoch"] for record in _records(eval_root)] == [5]  # the failed point never reaches the ledger
    assert engine.run() == 0  # the offline re-run retries exactly the skipped point
    assert scorer.calls == [("score", 5), ("score", 10), ("score", 10)]
    records = _records(eval_root)
    assert [record["epoch"] for record in records] == [5, 10]  # exactly one ledger point for the retry
    assert records[1]["m"] == 0.9


def test_watch_engine_merges_post_score_fields_into_the_record(tmp_path):
    ckpt_dir, eval_root = tmp_path / "ckpt", tmp_path / "eval"
    _touch_checkpoints(ckpt_dir, [5])
    sampler = ScriptedSamplerFactory({5: [{"epoch": 5}]})
    scorer = ScriptedScorer({5: 1.0})

    code = _engine(
        ckpt_dir,
        eval_root,
        EarlyStopRule(patience=2, min_epoch=0, max_epoch=100),
        sampler,
        scorer,
        post_score=lambda epoch, samples, epoch_dir: {"l2_trend": None, "round_trip_dice": 0.5},
    ).run(cohort_file="dev_cohort.json")

    assert code == 0
    record = _records(eval_root)[0]
    # skeleton fields open and close, the score and post-score fields sit between, in that order
    assert list(record) == ["eval_utc", "epoch", "checkpoint", "m", "l2_trend", "round_trip_dice", "cohort_file"]
    assert record["l2_trend"] is None and record["round_trip_dice"] == 0.5


def test_watch_engine_writes_the_early_stop_file_and_halts(tmp_path):
    ckpt_dir, eval_root = tmp_path / "ckpt", tmp_path / "eval"
    _touch_checkpoints(ckpt_dir, [5, 10, 15, 20])
    sampler = ScriptedSamplerFactory({epoch: [{"epoch": epoch}] for epoch in (5, 10, 15, 20)})
    scorer = ScriptedScorer({5: 1.0, 10: 0.9, 15: 0.9, 20: 0.5})

    code = _engine(ckpt_dir, eval_root, EarlyStopRule(patience=1, min_epoch=0, max_epoch=100), sampler, scorer).run()

    assert code == 0
    assert scorer.calls == [("score", 5), ("score", 10), ("score", 15)]  # the loop returns the moment stop fires
    stop_text = (ckpt_dir / STOP_FILE).read_text()
    stop = json.loads(stop_text)
    assert stop["epoch"] == 15
    assert "no new best" in stop["reason"]
    assert stop_text.endswith("\n")


def test_watch_engine_without_pending_checkpoints_is_a_no_op(tmp_path):
    ckpt_dir, eval_root = tmp_path / "ckpt", tmp_path / "eval"
    ckpt_dir.mkdir()
    sampler = ScriptedSamplerFactory({})
    scorer = ScriptedScorer({})

    code = _engine(ckpt_dir, eval_root, EarlyStopRule(patience=2, min_epoch=0, max_epoch=100), sampler, scorer).run()

    assert code == 0
    assert sampler.calls == [] and scorer.calls == []
    assert not (eval_root / "dev_trend.jsonl").exists()


def test_ledger_incremental_read_equals_a_full_reread(tmp_path):
    ledger = TrendLedger(tmp_path)
    records = [{"epoch": e, "m": 1.0 - 0.1 * e, "checkpoint": f"epoch_{e}.pt"} for e in (5, 10, 15)]

    ledger.append(records[0])
    ledger.append(records[1])
    assert ledger.read() == records[:2]
    ledger.append(records[2])
    # the incremental view equals a full re-read of the append-only file
    assert ledger.read() == records
    assert TrendLedger(tmp_path).read() == records
    assert ledger.read() == records  # repeated reads are stable


def test_selection_emitter_writes_the_argmin_contract(tmp_path):
    ledger = TrendLedger(tmp_path)
    for record in ({"epoch": 5, "m": 1.2, "checkpoint": "epoch_5.pt"}, {"epoch": 10, "m": 0.8, "checkpoint": "epoch_10.pt"}):
        ledger.append(record)
    out = tmp_path / "out" / "selection.json"

    code = SelectionEmitter(tmp_path).emit(out, rule_text="argmin mean dev FID over eval points (pre-recorded)")

    assert code == 0
    selection = json.loads(out.read_text())
    assert list(selection) == ["epoch", "mean_fid", "checkpoint", "rule", "trend", "recorded_utc"]
    assert selection["epoch"] == 10 and selection["mean_fid"] == 0.8
    assert selection["rule"] == "argmin mean dev FID over eval points (pre-recorded)"
    assert selection["trend"] == ledger.read()


def test_selection_emitter_supports_the_max_mean_ssim_shape(tmp_path):
    """The cross-modal shape: argmax SSIM, the PSNR trail merged from the best point."""
    ledger = TrendLedger(tmp_path)
    for record in (
        {"epoch": 5, "m": 0.6, "checkpoint": "epoch_5.pt", "mean_psnr": 25.0},
        {"epoch": 10, "m": 0.9, "checkpoint": "epoch_10.pt", "mean_psnr": 30.0},
    ):
        ledger.append(record)

    def extra_fields(trend, selection):
        best = next(point for point in trend if point["epoch"] == selection["epoch"] and point["m"] is not None)
        return {"mean_psnr": best.get("mean_psnr")}

    code = SelectionEmitter(tmp_path).emit(
        tmp_path / "selection.json",
        rule_text="argmax mean dev 3D SSIM over eval points (pre-registered; PSNR recorded alongside)",
        direction="max",
        metric_name="mean_ssim",
        extra_fields=extra_fields,
        summary_extra=lambda selection: f", mean_psnr {selection['mean_psnr']:.2f}",
    )

    assert code == 0
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert list(selection) == ["epoch", "mean_ssim", "checkpoint", "mean_psnr", "rule", "trend", "recorded_utc"]
    assert selection["epoch"] == 10 and selection["mean_ssim"] == 0.9 and selection["mean_psnr"] == 30.0


def test_selection_emitter_without_eval_points_fails(tmp_path):
    code = SelectionEmitter(tmp_path).emit(tmp_path / "selection.json", rule_text="argmin mean dev FID over eval points (pre-recorded)")

    assert code == 1
    assert not (tmp_path / "selection.json").exists()
