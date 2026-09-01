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

from ctmr.application.acceptance.distribution.token_dilution import ARM_ORDER
from ctmr.application.generation.modality_label.monitor import CandidateSampler, CohortFeatureScorer, FidTrendScorer, L2PostScore
from ctmr.application.generation.modality_label.token_swap_sampling import TokenSwapSampler
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
from ctmr.domain.measurement import HierarchyChecker

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


# --------------------------------------------------- watch engine collaborators (issue #225)


class _ToyVolumeNet(torch.nn.Module):
    """One-conv stand-in for the sampler's load path (state roundtrip only)."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(4, 4, 3, padding=1)


class _ScriptedSamplerEngine:
    """Engine-port stand-in: hands back the prebuilt networks, counts the calls."""

    def __init__(self, autoencoder, unet):
        self._autoencoder = autoencoder
        self._unet = unet
        self.calls = []

    def define_instance(self, args, instance_def_key):
        self.calls.append(f"define_instance:{instance_def_key}")
        return self._autoencoder if instance_def_key == "autoencoder_def" else self._unet

    def recon_model(self, autoencoder, scale_factor):
        self.calls.append("recon_model")
        from ctmr.infrastructure.maisi_engine.utils_infer import ReconModel

        return ReconModel(autoencoder=autoencoder, scale_factor=scale_factor)


def test_candidate_sampler_loads_models_through_the_engine_port(tmp_path):
    """The sidecar sampler reaches its networks only through the injected
    GenerationEngine port (issue #272): define_instance for both networks,
    recon_model for the VAE decode wrapper, checkpoints loaded verbatim."""
    autoencoder, unet = _ToyVolumeNet(), _ToyVolumeNet()
    with torch.no_grad():
        autoencoder.conv.weight.fill_(0.25)
    torch.save(autoencoder.state_dict(), tmp_path / "ae.pt")
    torch.save({"unet_state_dict": unet.state_dict(), "scale_factor": 0.5}, tmp_path / "candidate.pt")
    args = types.SimpleNamespace(
        trained_autoencoder_path=str(tmp_path / "ae.pt"),
        noise_scheduler={
            "_target_": "monai.networks.schedulers.rectified_flow.RFlowScheduler",
            "num_train_timesteps": 1000,
            "use_discrete_timesteps": False,
            "use_timestep_transform": True,
            "sample_method": "uniform",
            "scale": 1.4,
        },
    )
    engine = _ScriptedSamplerEngine(autoencoder, unet)

    model, recon = CandidateSampler(args, torch.device("cpu"), None, engine).load_models(tmp_path / "candidate.pt")

    assert engine.calls == ["define_instance:autoencoder_def", "define_instance:diffusion_unet_def", "recon_model"]
    assert recon.scale_factor == pytest.approx(0.5)
    assert float(model.scale_factor) == pytest.approx(0.5)
    assert torch.allclose(recon.autoencoder.conv.weight, torch.full_like(recon.autoencoder.conv.weight, 0.25))


class _ScriptedFeatures:
    def volume_features(self, path):
        return {"xy": np.zeros(2), "yz": None, "zx": np.ones(2)}


class _ScriptedFid:
    def __init__(self):
        self.seen = None

    def trend_fields(self, generated):
        self.seen = generated
        return {"fid": {"fid": "report"}, "m": 0.42}, "mean_fid=0.42"


def test_fid_trend_scorer_assembles_the_plane_means():
    fid = _ScriptedFid()
    samples = [{"modality": "t1n", "path": "a.nii.gz"}, {"modality": "t1c", "path": "b.nii.gz"}]

    fields, log_line = FidTrendScorer(_ScriptedFeatures(), fid)(samples)

    assert fields == {"fid": {"fid": "report"}, "m": 0.42}
    assert log_line == "mean_fid=0.42"
    # the None plane is skipped, per-sample planes mean into the per-modality buckets
    assert list(fid.seen) == ["t1n", "t1c", "t2w", "t2f"]  # all four target modalities, in order
    assert fid.seen["t1n"]["yz"] == []
    assert len(fid.seen["t1n"]["zx"]) == 1
    assert fid.seen["t2w"] == {"xy": [], "yz": [], "zx": []}  # no samples -> empty buckets


def test_cohort_feature_scorer_assembles_the_gathered_entries():
    """The embedded-validation scorer (issue #278): the all_gathered per-item
    plane-mean features fold into the same {modality: {plane: [...]}} view the
    sidecar scorer built -- the gathered ordering survives (summation order is
    the only allowed drift vs the single-card full cohort)."""
    fid = _ScriptedFid()
    entries = [
        {
            "sub": "GLI",
            "case": "case-a",
            "modality": "t1n",
            "path": "a.nii.gz",
            "features": {"xy": np.ones(4), "yz": None, "zx": np.zeros(4)},
        },
        {
            "sub": "GLI",
            "case": "case-b",
            "modality": "t1n",
            "path": "b.nii.gz",
            "features": {"xy": np.full(4, 2.0), "yz": None, "zx": np.ones(4)},
        },
    ]

    fields, log_line = CohortFeatureScorer(fid)(entries)

    assert fields == {"fid": {"fid": "report"}, "m": 0.42}
    assert log_line == "mean_fid=0.42"
    xy = fid.seen["t1n"]["xy"]
    assert len(xy) == 2 and np.all(xy[0] == 1.0) and np.all(xy[1] == 2.0)  # gathered order kept
    assert fid.seen["t1n"]["yz"] == []  # the None plane is skipped
    assert fid.seen["t1n"]["zx"][1].sum() == 4.0
    assert fid.seen["t2w"] == {"xy": [], "yz": [], "zx": []}


class _ScriptedL2:
    def __init__(self, fail=False):
        self.calls = []
        self._fail = fail

    def run(self, samples, cohort, work_dir):
        self.calls.append((len(samples), len(cohort), work_dir))
        if self._fail:
            raise RuntimeError("nnunet boom")
        return {"median_wt": 1.0}


def test_l2_post_score_skip_degrades_to_none_and_success_passes_through(tmp_path):
    l2 = _ScriptedL2()
    assert L2PostScore(l2, [{"case": "c"}], skip=True)(5, [{"path": "s"}], tmp_path) == {"l2_trend": None}
    assert l2.calls == []  # --skip-l2 never reaches the instruments

    assert L2PostScore(l2, [{"case": "c"}], skip=False)(5, [{"path": "s"}], tmp_path) == {"l2_trend": {"median_wt": 1.0}}
    assert l2.calls == [(1, 1, tmp_path)]


def test_l2_post_score_failure_records_none_instead_of_dying(tmp_path):
    # a single-epoch instrument hiccup must not kill the sidecar
    assert L2PostScore(_ScriptedL2(fail=True), [], skip=False)(5, [], tmp_path) == {"l2_trend": None}


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
    construction point (canonical ``ctmr measure predict`` entry,
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
        # sys.path entries do not reach a fresh child process)
        assert str(SRC_ROOT) in captured["env"]["PYTHONPATH"]


def test_l2_trend_hierarchy_count_matches_the_canonical_checker(tmp_path):
    """#223 convergence: the P1 dev monitor's per-case ``hier_viol`` equals
    ``HierarchyChecker.violates`` on crafted inputs (the canonical containment
    single expression, ADR-0010 decision 3) -- the dev chain and the terminal-
    acceptance chain share one measurement semantics. run_fail rows keep their
    placeholder shape."""
    well_formed = np.zeros((4, 4, 4), dtype=np.uint8)
    well_formed[0, 0, 0] = 3
    well_formed[0, 0, 1] = 1
    escaped = well_formed.copy()
    escaped[2, 2, 2] = 4
    preds = {"well_formed": well_formed, "domain_escape": escaped}
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    for name, array in preds.items():
        sitk.WriteImage(sitk.GetImageFromArray(array), str(pred_dir / f"CASE-{name}.nii.gz"))

    cohort = [{"sub": "GLI", "case": "CASE-well_formed"}, {"sub": "GLI", "case": "CASE-domain_escape"}]
    runner = L2TrendRunner(None, None, None)
    rows = runner.measure(cohort, tmp_path, pred_dir)

    by_case = {row["case"]: row for row in rows}
    for name, array in preds.items():
        row = by_case[f"CASE-{name}"]
        assert row["run_fail"] is False, name
        assert row["hier_viol"] == HierarchyChecker.violates(array), name
    assert by_case["CASE-domain_escape"]["hier_viol"] is True  # the equivalence is not vacuous
    assert not by_case["CASE-well_formed"]["hier_viol"]
    # the run_fail placeholder row keeps its shape (call-site concern, not measurement)
    fail_rows = runner.measure([{"sub": "GLI", "case": "MISSING"}], tmp_path, pred_dir)
    assert fail_rows == [{"sub": "GLI", "case": "MISSING", "run_fail": True}]


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


# --------------------------------------------------- write-path affine (issue #249)

# The real sampling spacing the v1 DM natively generates at (ruling #6 / #247). The
# pre-fix sidecar writer declared unit 1 mm, which made the instrument chain's 1 mm
# resample a no-op and produced the centroid pad-world artefacts the audit quantified.
TRUE_DM_SPACING_AFFINE = np.diag([0.94, 0.94, 1.36, 1.0])


class _StubSpacings:
    """Stands in for CohortSpacingSource; the write path only reads spacing_of."""

    def spacing_of(self, case):
        return [0.94, 0.94, 1.36]


def _stub_gpu_sampler(monkeypatch):
    """Stub the GPU-bound model load / denoise so the write path runs CPU-only.

    The seam under test is the on-disk affine, not the sampling, so the sampler
    returns a synthetic volume and never touches a checkpoint or a device.
    """
    monkeypatch.setattr(CandidateSampler, "load_models", lambda self, checkpoint_path: (object(), object()))
    monkeypatch.setattr(
        CandidateSampler,
        "sample_one",
        lambda self, model, recon_model, modality_token, spacing, seed: np.zeros((256, 256, 128), dtype=np.int16),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def test_modality_label_write_stamps_the_true_dm_spacing(tmp_path, monkeypatch):
    """The dev-sidecar write path declares the v1 DM's real spacing, not unit 1 mm.

    Pin: a synthetic volume written through ``CandidateSampler.generate_cohort``
    reads back with the true-spacing affine (never ``np.diag([1, 1, 1])``), so a
    future write-path refactor cannot silently regress to the 1 mm convention.
    """
    _stub_gpu_sampler(monkeypatch)
    cohort = [{"sub": "GLI", "case": "BraTS-GLI-0001-000"}]

    # the injected engine port stays unused here: _stub_gpu_sampler replaced load_models wholesale
    samples = CandidateSampler(types.SimpleNamespace(), torch.device("cpu"), None, None).generate_cohort(
        "/ckpt/epoch_20.pt", cohort, _StubSpacings(), tmp_path
    )

    assert samples  # one sample per target modality was written
    for sample in samples:
        affine = nib.load(sample["path"]).affine
        assert np.allclose(affine, TRUE_DM_SPACING_AFFINE), sample["path"]
        assert not np.allclose(np.diag(affine)[:3], (1.0, 1.0, 1.0)), sample["path"]  # the retired convention


def test_token_swap_write_stamps_the_true_dm_spacing(tmp_path, monkeypatch):
    """The job-D diagnostic arm shares the modality-label write path (#249): its
    five per-case products carry the same true-spacing affine."""
    _stub_gpu_sampler(monkeypatch)
    sampler = TokenSwapSampler(types.SimpleNamespace(), torch.device("cpu"), None)

    written = sampler.sample_cohort("/ckpt/epoch_20.pt", [{"case": "BraTS-GLI-0001-000"}], _StubSpacings(), tmp_path)

    assert written == len(ARM_ORDER)
    for path in tmp_path.glob("*.nii.gz"):
        assert np.allclose(nib.load(str(path)).affine, TRUE_DM_SPACING_AFFINE), path.name
