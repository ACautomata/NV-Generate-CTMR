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

"""Dev-sidecar assertions for the modality_label family and its shared trend machinery (ticket 10).

The retired dev-eval entry's ``selftest`` assertions live here as pytest
functions (cohort determinism, the pre-recorded early-stop rule, the trend
ledger), together with the instrument-adapter / canonical-argv / trend-
preprocessing gates for the shared machinery the modality-label and mask
sidecars build on. Torch-level: runs without a GPU but imports torch --
never skipped around the torch mark.
"""

from __future__ import annotations

import json
import subprocess
import types
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
import torch

from ctmr.application.generation.trend import (
    DevCohortBuilder,
    L2TrendRunner,
    MrTrendFeatures,
    RealReferenceBank,
)
from ctmr.application.shell import COHORT_QUOTAS, EarlyStopRule, TrendLedger
from ctmr.domain.grid import (
    TREND_FEATURE_GRID,
    CenterCropOrPad,
    GridResampler,
    InstrumentGridAdapter,
)
from ctmr.domain.instrument_spec import INSTRUMENT_SPECS, FrozenInstrumentCommand

pytestmark = pytest.mark.torch

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"  # the installed package's src tree


# --------------------------------------------------- cohort / early-stop / ledger


def test_dev_cohort_is_fixed_and_deterministic_within_quotas(tmp_path):
    entries = []
    for challenge, quota in COHORT_QUOTAS.items():
        for index in range(quota + 2):
            entries.append({"sub": challenge, "case": f"FIX{challenge}-{index:04d}-000", "modality": "mri_t1_skull_stripped"})
    dev_list = tmp_path / "p1_dev.json"
    dev_list.write_text(json.dumps({"training": entries}))

    cohort = DevCohortBuilder(dev_list).build()

    assert len(cohort) == sum(COHORT_QUOTAS.values())
    assert DevCohortBuilder(dev_list).build() == cohort  # deterministic
    assert {item["sub"] for item in cohort} == set(COHORT_QUOTAS)  # every challenge represented


def test_early_stop_rule_never_stops_an_improving_trend():
    improving = [{"epoch": e, "m": 1.0 - 0.01 * e} for e in (5, 10, 15, 20, 25, 30)]
    stop, _ = EarlyStopRule(patience=3, min_epoch=30, max_epoch=100).should_stop(improving)
    assert not stop


def test_early_stop_rule_stops_the_registered_plateau_after_patience():
    improving = [{"epoch": e, "m": 1.0 - 0.01 * e} for e in (5, 10, 15, 20, 25, 30)]
    plateau = improving + [{"epoch": e, "m": 0.75} for e in (35, 40, 45)]
    rule = EarlyStopRule(patience=3, min_epoch=30, max_epoch=100)

    stop, reason = rule.should_stop(plateau)
    assert stop, reason

    short = improving + [{"epoch": 35, "m": 0.75}, {"epoch": 40, "m": 0.75}]
    stop, _ = rule.should_stop(short)
    assert not stop  # patience not exhausted yet


def test_selection_is_the_argmin_mean_fid_eval_point():
    trend = [{"epoch": 5, "m": 1.2}, {"epoch": 10, "m": 0.8}]
    plateau = trend + [{"epoch": 15, "m": 0.75}, {"epoch": 20, "m": 0.75}]

    selection = EarlyStopRule.selection(plateau)

    assert selection["epoch"] == 15
    assert selection["mean_fid"] == pytest.approx(0.75)


def test_trend_ledger_roundtrip(tmp_path):
    ledger = TrendLedger(tmp_path)
    trend = [{"epoch": 5, "m": 1.2, "checkpoint": "epoch_5.pt"}, {"epoch": 10, "m": 0.8, "checkpoint": "epoch_10.pt"}]
    for record in trend:
        ledger.append(record)

    assert ledger.read() == trend


# --------------------------------------------------- shared trend machinery gates


def test_prep_inputs_matches_the_instrument_adapter(tmp_path):
    """L2TrendRunner.prep_inputs delegates to the frozen instrument adapter
    (ADR-0008 adoption: the registered linear->B-spline + centred crop/pad changes
    land the nnU-Net inputs on the terminal-acceptance geometry)."""
    samples = []
    for modality in sorted(L2TrendRunner.NN_CHANNELS):
        path = tmp_path / f"{modality}.nii.gz"
        sitk.WriteImage(_trend_volume(), str(path))  # v1-DM-ish footprint, mixed pad/crop axes
        samples.append({"sub": "GLI", "case": "SYNTH-9001", "modality": modality, "path": str(path)})

    L2TrendRunner(None, None, None).prep_inputs(samples, tmp_path / "inputs")

    for modality in sorted(L2TrendRunner.NN_CHANNELS):
        suffix = L2TrendRunner.NN_CHANNELS[modality]
        produced = sitk.ReadImage(str(tmp_path / "inputs" / "GLI" / f"SYNTH-9001_{suffix}.nii.gz"))
        expected = InstrumentGridAdapter.continuum().align(sitk.ReadImage(str(tmp_path / f"{modality}.nii.gz")))
        assert np.array_equal(sitk.GetArrayFromImage(produced), sitk.GetArrayFromImage(expected))
        assert produced.GetSize() == (240, 240, 155)


def test_predict_uses_the_canonical_instrument_argv(tmp_path, monkeypatch):
    """ADR-0009 #108 adoption: ``L2TrendRunner.predict`` produces exactly
    ``FrozenInstrumentCommand.build`` -- the dev-side sidecar shares the single
    construction point (canonical ``python -m ctmr.instrument.predict`` entry,
    no TTA token anywhere, SSA derived config inside the spec)."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    for challenge in ("GLI", "SSA"):
        log_path = tmp_path / f"predict-{challenge}.log"
        results = {"GLI": "/results/gli", "SSA": "/results/ssa"}
        rc = L2TrendRunner(results, None, None).predict(challenge, "/in", "/out", log_path)
        assert rc == 0
        assert captured["cmd"] == FrozenInstrumentCommand(INSTRUMENT_SPECS[challenge]).build("/in", "/out")
        assert "--disable_tta" not in captured["cmd"]
        assert captured["env"]["nnUNet_raw"]  # the nnU-Net env wiring stays with the executor
        # the child process gets the module's src tree on PYTHONPATH (process-local
        # sys.path entries do not reach a fresh `python -m ctmr.instrument.predict`)
        assert str(SRC_ROOT) in captured["env"]["PYTHONPATH"]


def test_trend_preprocess_lands_on_the_pinned_engine_composition(tmp_path):
    """ADR-0008 decision 4: MrTrendFeatures.preprocess keeps the percentile
    normalisation, then resamples (linear) and centre-crops/pads via the generic
    engine onto the 240x240x160 trend grid.  This pins the xyz<->zyx wiring: the
    nibabel input is (x, y, z) and the preprocessed output stays (x, y, z), while
    the engine works in sitk zyx."""
    sample = tmp_path / "sample.nii.gz"
    zz, yy, xx = np.mgrid[0:80, 0:150, 0:100]  # zyx
    volume = 100.0 * np.exp(-(((xx - 33.0) ** 2 + (yy - 75.0) ** 2 + (zz - 40.0) ** 2) / 3.0e3))
    nib.save(nib.Nifti1Image(volume.astype(np.float32), np.diag([1.0, 1.5, 2.0, 1.0])), str(sample))

    produced = MrTrendFeatures.preprocess(str(sample))
    assert produced.shape == (240, 240, 160)  # xyz, the pinned grid

    # replicate the percentile step, then the engine composition the method delegates to
    data = np.asarray(nib.load(str(sample)).dataobj, dtype=np.float32)
    values = data[data > 0] if (data > 0).any() else data.ravel()
    lo, hi = np.percentile(values, [0.0, 99.5])
    normalized = np.clip((data - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    sitk_image = sitk.GetImageFromArray(np.ascontiguousarray(normalized.transpose(2, 1, 0)).astype(np.float32))
    sitk_image.SetSpacing((1.0, 1.5, 2.0))
    aligned = CenterCropOrPad().crop_or_pad(GridResampler(sitk.sitkLinear).resample(sitk_image, TREND_FEATURE_GRID), TREND_FEATURE_GRID)
    expected = sitk.GetArrayFromImage(aligned).transpose(2, 1, 0)
    assert np.allclose(produced, expected, atol=1e-5)


def test_real_reference_bank_roundtrip_loads_under_weights_only(tmp_path):
    """The cached bank reload branch: numpy-array payloads load with the
    weights_only allowlist applied at the load point (no import-time mutation)."""
    dev_list = tmp_path / "dev.json"
    dev_list.write_text(json.dumps({"training": []}))
    before = list(torch.serialization.get_safe_globals())
    payload = {
        "t1n": {"xy": np.stack([np.ones(4), np.zeros(4)]), "yz": np.stack([np.ones(4), np.zeros(4)]), "zx": np.stack([np.ones(4), np.zeros(4)])}
    }
    torch.save(payload, tmp_path / "real_reference_bank.pt")

    bank = RealReferenceBank(dev_list, "unused", None, tmp_path).build()

    assert set(bank) == {"t1n"}
    assert bank["t1n"]["xy"].shape == (2, 4)
    # the load-point exposure is additive: whatever was allowlisted before stays
    assert set(before) <= set(torch.serialization.get_safe_globals())


def _trend_volume():
    zz, yy, xx = np.mgrid[0:80, 0:130, 0:300]  # zyx -> xyz size (300, 130, 80): crop x/y, pad z
    image = sitk.GetImageFromArray((100.0 * np.exp(-(((xx - 100.0) ** 2 + (yy - 65.0) ** 2 + (zz - 40.0) ** 2) / 2.0e4))).astype(np.float32))
    image.SetSpacing((1.0, 2.0, 1.5))  # xyz mm
    return image
