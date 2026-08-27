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

"""Pytest wrapper for the P1 dev-eval selftest (issue #104 / ADR-0013 §2).

Calls the resident ``DevEvalSelfTest`` directly (the implementation stays in
the production script; the ``selftest`` subcommand remains the sugon-side
integration-gate entry and must not forward pytest). Torch-level: runs without
a GPU but imports torch, so it skips itself on light stacks via
``pytest.importorskip`` (ADR-0013 §4).
"""

import subprocess
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("monai")  # transitively imported at module level via diff_model_setting / utils_infer

import nibabel as nib  # noqa: E402  (importorskip must precede the torch-dependent import)
import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402

from ctmr.domain.grid import (  # noqa: E402
    TREND_FEATURE_GRID,
    CenterCropOrPad,
    GridResampler,
    InstrumentGridAdapter,  # noqa: E402
)
from ctmr.domain.instrument_spec import INSTRUMENT_SPECS, FrozenInstrumentCommand  # noqa: E402
from scripts.brats_p1_dev_eval import (  # noqa: E402  (importorskip must precede the torch-dependent import)
    DevEvalSelfTest,
    L2TrendRunner,
    MrTrendFeatures,
)


@pytest.mark.torch
def test_p1_dev_eval_selftest(tmp_path):
    failures = DevEvalSelfTest(tmp_path).run()

    assert failures == []


@pytest.mark.torch
def test_p1_dev_eval_prep_inputs_matches_the_instrument_adapter(tmp_path):
    """L2TrendRunner.prep_inputs now delegates to the frozen instrument adapter
    (ADR-0008 adoption: the registered linear->B-spline + centred crop/pad changes
    land the nnU-Net inputs on the terminal-acceptance geometry)."""
    samples = []
    for modality, suffix in sorted(L2TrendRunner.NN_CHANNELS.items()):
        path = tmp_path / f"{modality}.nii.gz"
        sitk.WriteImage(_trend_volume(), str(path))  # v1-DM-ish footprint, mixed pad/crop axes
        samples.append({"sub": "GLI", "case": "SYNTH-9001", "modality": modality, "path": str(path)})

    L2TrendRunner(None, None, None).prep_inputs(samples, tmp_path / "inputs")

    for modality, suffix in sorted(L2TrendRunner.NN_CHANNELS.items()):
        produced = sitk.ReadImage(str(tmp_path / "inputs" / "GLI" / f"SYNTH-9001_{suffix}.nii.gz"))
        expected = InstrumentGridAdapter.continuum().align(sitk.ReadImage(str(tmp_path / f"{modality}.nii.gz")))
        assert np.array_equal(sitk.GetArrayFromImage(produced), sitk.GetArrayFromImage(expected))
        assert produced.GetSize() == (240, 240, 155)


@pytest.mark.torch
def test_p1_dev_eval_predict_uses_the_canonical_instrument_argv(tmp_path, monkeypatch):
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
        # sys.path.insert does not reach a fresh `python -m ctmr.instrument.predict`)
        assert str(Path(__file__).resolve().parents[2] / "src") in captured["env"]["PYTHONPATH"]


def _trend_volume():
    zz, yy, xx = np.mgrid[0:80, 0:130, 0:300]  # zyx -> xyz size (300, 130, 80): crop x/y, pad z
    image = sitk.GetImageFromArray((100.0 * np.exp(-(((xx - 100.0) ** 2 + (yy - 65.0) ** 2 + (zz - 40.0) ** 2) / 2.0e4))).astype(np.float32))
    image.SetSpacing((1.0, 2.0, 1.5))  # xyz mm
    return image


@pytest.mark.torch
def test_p1_dev_eval_trend_preprocess_lands_on_the_pinned_engine_composition(tmp_path):
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
