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

"""Device-injection smoke for every diagnostic/inference entry main (issue #280, ADR-0019 §8).

Each scattered entry used to self-select its device inside ``main``; the
unified injection (``devices.resolve_device`` behind ``--device``) must keep
both behaviors: an explicit ``--device cpu`` reaches the device-consuming
collaborators verbatim (the CPU smoke path), and an absent flag resolves to
the hardcoded-era fallback -- here proven by forcing availability true and
expecting ``cuda`` where the old code would have picked it. The heavy
collaborators (engine assembly, samplers, writers, the watch shell) are
replaced by recording fakes; every touched file lives under ``tmp_path``.
Torch-level: runs without a GPU -- never skipped around the torch mark.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ctmr.application.generation.cross_modal import baseline as cross_modal_baseline
from ctmr.application.generation.cross_modal import candidate as cross_modal_candidate
from ctmr.application.generation.cross_modal import monitor as cross_modal_monitor
from ctmr.application.generation.mask import monitor as mask_monitor
from ctmr.application.generation.mask import sample as mask_sample
from ctmr.application.generation.modality_label import monitor as ml_monitor
from ctmr.application.generation.modality_label import token_swap_sampling as token_swap

pytestmark = pytest.mark.torch

GRID = (256, 256, 128)


# --------------------------------------------------------------------- fakes


class Recording:
    """Stands in for a device-consuming collaborator: records its init args, any method is a no-op."""

    def __init__(self, *args, **kwargs):
        self.init_args = args

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _fake_engine():
    return SimpleNamespace(
        load_config=lambda *paths: SimpleNamespace(diffusion_unet_inference={"dim": list(GRID), "num_inference_steps": 10}, cfg_guidance_scale=1.0)
    )


def _fake_runtime(monkeypatch):
    runtime = SimpleNamespace(weights_ref_of_file=lambda: object(), engine=lambda: _fake_engine(), logger=lambda name: object())
    monkeypatch.setattr(cross_modal_baseline, "GenerateRuntime", lambda: runtime)
    monkeypatch.setattr(cross_modal_candidate, "GenerateRuntime", lambda: runtime)
    monkeypatch.setattr(cross_modal_monitor, "GenerateRuntime", lambda: runtime)
    return runtime


def _write_json(path, payload):
    path.write_text(json.dumps(payload))
    return path


# ------------------------------------------------- modality_label dev-eval entry


def test_modality_label_reference_injects_the_device(monkeypatch, tmp_path):
    reference_argv = ["reference", "--dev-list", "lists/dev.json", "--raw-root", str(tmp_path), "--eval-root", str(tmp_path)]
    features_box = []
    monkeypatch.setattr(ml_monitor, "MrTrendFeatures", lambda device: (features_box.append(device), Recording(device))[1])
    monkeypatch.setattr(ml_monitor, "RealReferenceBank", lambda *args: SimpleNamespace(build=lambda: object()))

    # explicit injection: cpu reaches the feature extractor verbatim
    assert ml_monitor.main([*reference_argv, "--device", "cpu"]) == 0
    assert features_box[-1] == torch.device("cpu")

    # absent flag: the hardcoded-era fallback (availability forced true -> cuda)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert ml_monitor.main(reference_argv) == 0
    assert features_box[-1] == torch.device("cuda")


def test_modality_label_watch_injects_the_device(monkeypatch, tmp_path):
    argv = [
        "watch",
        "--ckpt-dir",
        str(tmp_path / "ckpt"),
        "--eval-root",
        str(tmp_path),
        "--dev-list",
        str(_write_json(tmp_path / "dev.json", {"training": []})),
        "--raw-root",
        str(tmp_path),
        "--emb-root",
        str(tmp_path),
        "-e",
        "env.json",
        "-c",
        "config.json",
        "-t",
        "net.json",
    ]

    monkeypatch.setattr(ml_monitor, "DevCohortBuilder", lambda dev_list: SimpleNamespace(write=lambda path: []))
    monkeypatch.setattr(ml_monitor, "CohortSpacingSource", lambda *args: object())
    monkeypatch.setattr(ml_monitor, "modality_label_engine", lambda: _fake_engine())
    features_box, sampler_box = [], []
    monkeypatch.setattr(ml_monitor, "MrTrendFeatures", lambda device: (features_box.append(device), Recording(device))[1])
    monkeypatch.setattr(ml_monitor, "RealReferenceBank", lambda *args: SimpleNamespace(build=lambda: object()))
    monkeypatch.setattr(ml_monitor, "CandidateSampler", lambda *args: (sampler_box.append(args[1]), Recording(*args))[1])
    monkeypatch.setattr(ml_monitor, "L2TrendRunner", lambda *args, **kwargs: object())  # real __init__ takes autodiscover_specs
    monkeypatch.setattr(ml_monitor, "WatchEngine", lambda **kwargs: SimpleNamespace(run=lambda cohort_file: 0))

    assert ml_monitor.main([*argv, "--device", "cpu"]) == 0
    assert features_box[-1] == torch.device("cpu")
    assert sampler_box[-1] == torch.device("cpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert ml_monitor.main(argv) == 0
    assert features_box[-1] == torch.device("cuda")
    assert sampler_box[-1] == torch.device("cuda")


# --------------------------------------------------------- mask dev-eval entry


def test_mask_reference_injects_the_device(monkeypatch, tmp_path):
    reference_argv = ["reference", "--dev-list", "lists/dev.json", "--raw-root", str(tmp_path), "--eval-root", str(tmp_path)]
    features_box = []
    monkeypatch.setattr(mask_monitor, "DevList", lambda dev_list, eval_root: SimpleNamespace(build=lambda: []))
    monkeypatch.setattr(mask_monitor, "MrTrendFeatures", lambda device: (features_box.append(device), Recording(device))[1])
    monkeypatch.setattr(mask_monitor, "RealReferenceBank", lambda *args: SimpleNamespace(build=lambda: object()))

    # explicit injection: cpu reaches the feature extractor verbatim
    assert mask_monitor.main([*reference_argv, "--device", "cpu"]) == 0
    assert features_box[-1] == torch.device("cpu")

    # absent flag: the hardcoded-era fallback (availability forced true -> cuda)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert mask_monitor.main(reference_argv) == 0
    assert features_box[-1] == torch.device("cuda")


def test_mask_monitor_injects_the_device(monkeypatch, tmp_path):
    argv = [
        "watch",
        "--ckpt-dir",
        str(tmp_path / "ckpt"),
        "--eval-root",
        str(tmp_path),
        "--dev-list",
        str(_write_json(tmp_path / "dev.json", {"training": []})),
        "--raw-root",
        str(tmp_path),
        "--label-root",
        str(tmp_path),
        "-e",
        "env.json",
        "-c",
        "config.json",
        "-t",
        "net.json",
    ]

    monkeypatch.setattr(mask_monitor, "DevList", lambda dev_list, eval_root: SimpleNamespace(build=lambda: []))
    monkeypatch.setattr(mask_monitor, "DevCohortBuilder", lambda dev_list: SimpleNamespace(write=lambda path: []))
    monkeypatch.setattr(mask_monitor, "CohortSpacingSource", lambda *args: object())
    monkeypatch.setattr(mask_monitor, "ConditionMaskSource", lambda *args: object())
    monkeypatch.setattr(mask_monitor, "mask_engine", lambda: _fake_engine())
    features_box, sampler_box = [], []
    monkeypatch.setattr(mask_monitor, "MrTrendFeatures", lambda device: (features_box.append(device), Recording(device))[1])
    monkeypatch.setattr(mask_monitor, "RealReferenceBank", lambda *args: SimpleNamespace(build=lambda: object()))
    monkeypatch.setattr(mask_monitor, "CandidateSampler", lambda *args: (sampler_box.append(args[1]), Recording(*args))[1])
    monkeypatch.setattr(mask_monitor, "L2TrendRunner", lambda *args, **kwargs: object())  # real __init__ takes autodiscover_specs
    monkeypatch.setattr(mask_monitor, "WatchEngine", lambda **kwargs: SimpleNamespace(run=lambda cohort_file: 0))

    assert mask_monitor.main([*argv, "--device", "cpu"]) == 0
    assert features_box[-1] == torch.device("cpu")
    assert sampler_box[-1] == torch.device("cpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert mask_monitor.main(argv) == 0
    assert features_box[-1] == torch.device("cuda")
    assert sampler_box[-1] == torch.device("cuda")


# -------------------------------------------------------- mask holdout entry


def test_mask_holdout_sample_injects_the_device(monkeypatch, tmp_path):
    _write_json(tmp_path / "run.json", {"selection": {"epoch": 1}})
    _write_json(tmp_path / "manifest.json", {})
    (tmp_path / "out").mkdir()  # the entry writes the samples json into the out root
    argv = [
        "--run",
        str(tmp_path / "run.json"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--out-root",
        str(tmp_path / "out"),
        "--raw-root",
        str(tmp_path),
        "--label-root",
        str(tmp_path),
        "-e",
        "env.json",
        "-c",
        "config.json",
        "-t",
        "net.json",
    ]

    monkeypatch.setattr(mask_sample, "mask_engine", lambda: _fake_engine())
    monkeypatch.setattr(mask_sample, "HoldoutCohortBuilder", lambda *args: SimpleNamespace(build=lambda: [{"case": "c"}]))
    monkeypatch.setattr(mask_sample, "HoldoutSpacingSource", lambda *args: object())
    monkeypatch.setattr(mask_sample, "HoldoutMaskSource", lambda *args: object())
    writer_box = []
    monkeypatch.setattr(
        mask_sample,
        "HoldoutSampleWriter",
        lambda *args: (writer_box.append(args[4]), SimpleNamespace(write=lambda *write_args: [{"case": "c"}]))[1],
    )

    assert mask_sample.main([*argv, "--device", "cpu"]) == 0
    assert writer_box[-1] == torch.device("cpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert mask_sample.main(argv) == 0
    assert writer_box[-1] == torch.device("cuda")


# -------------------------------------------------- cross-modal dev-eval entry


def test_cross_modal_watch_injects_the_device(monkeypatch, tmp_path):
    argv = [
        "watch",
        "--ckpt-dir",
        str(tmp_path / "ckpt"),
        "--eval-root",
        str(tmp_path),
        "--dev-list",
        str(_write_json(tmp_path / "dev.json", {"training": []})),
        "--raw-root",
        str(tmp_path),
        "--phase-root",
        str(tmp_path),
        "-e",
        "env.json",
        "-c",
        "config.json",
        "-t",
        "net.json",
    ]
    _fake_runtime(monkeypatch)
    monkeypatch.setattr(cross_modal_monitor, "DevList", lambda dev_list, eval_root: SimpleNamespace(build=lambda: []))
    monkeypatch.setattr(cross_modal_monitor, "DevCohort", lambda dev_list: SimpleNamespace(cases=lambda: []))
    sampler_box = []
    monkeypatch.setattr(cross_modal_monitor, "CandidateSampler", lambda *args: (sampler_box.append(args[1]), Recording(*args))[1])
    monkeypatch.setattr(cross_modal_monitor, "WatchEngine", lambda **kwargs: SimpleNamespace(run=lambda cohort_file: 0))

    assert cross_modal_monitor.main([*argv, "--device", "cpu"]) == 0
    assert sampler_box[-1] == torch.device("cpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert cross_modal_monitor.main(argv) == 0
    assert sampler_box[-1] == torch.device("cuda")


# ------------------------------------------- cross-modal baseline/candidate entries


def _baseline_argv(tmp_path):
    _write_json(tmp_path / "run.json", {"run_id": "fake"})
    _write_json(tmp_path / "manifest.json", {})
    (tmp_path / "out").mkdir()  # the entries write samples/pairs json into the out root
    return [
        "--run",
        str(tmp_path / "run.json"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--out-root",
        str(tmp_path / "out"),
        "--raw-root",
        str(tmp_path),
        "--infer-config",
        str(tmp_path / "infer.json"),
        "-e",
        "env.json",
        "-c",
        "config.json",
        "-t",
        "net.json",
    ]


def test_cross_modal_baseline_injects_the_device(monkeypatch, tmp_path):
    argv = _baseline_argv(tmp_path)
    _fake_runtime(monkeypatch)
    monkeypatch.setattr(cross_modal_baseline, "BaselineRunGuard", lambda *args: SimpleNamespace(check=lambda: Path(tmp_path / "epoch.pt")))
    monkeypatch.setattr(cross_modal_baseline, "BaselineInferenceConfig", SimpleNamespace(from_path=lambda path: SimpleNamespace(grid=GRID)))
    monkeypatch.setattr(cross_modal_baseline, "SideCohortBuilder", lambda *args: SimpleNamespace(build=lambda: [{"case": "c"}]))
    monkeypatch.setattr(cross_modal_baseline, "RawCaseLayout", lambda *args: object())
    writer_box = []
    monkeypatch.setattr(
        cross_modal_baseline,
        "BaselineSampleWriter",
        lambda *args: (writer_box.append(args[4]), SimpleNamespace(write=lambda *write_args: ([], {"records": []})))[1],
    )

    assert cross_modal_baseline.main([*argv, "--device", "cpu"]) == 0
    assert writer_box[-1] == torch.device("cpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert cross_modal_baseline.main(argv) == 0
    assert writer_box[-1] == torch.device("cuda")


def test_cross_modal_candidate_injects_the_device(monkeypatch, tmp_path):
    argv = _baseline_argv(tmp_path)
    argv[argv.index("--infer-config") + 1] = str(tmp_path / "infer.json")
    stage0_pairs = _write_json(tmp_path / "pairs.json", {"records": [{"case": "c"}]})
    argv += ["--stage0-pairs", str(stage0_pairs)]
    _fake_runtime(monkeypatch)
    monkeypatch.setattr(cross_modal_candidate, "CandidateRunGuard", lambda *args: SimpleNamespace(check=lambda: Path(tmp_path / "epoch.pt")))
    monkeypatch.setattr(cross_modal_candidate, "CandidateInferenceConfig", SimpleNamespace(from_path=lambda path: SimpleNamespace(grid=GRID)))
    monkeypatch.setattr(cross_modal_candidate, "SideCohortBuilder", lambda *args: SimpleNamespace(build=lambda: [{"case": "c"}]))
    monkeypatch.setattr(cross_modal_candidate, "RawCaseLayout", lambda *args: object())
    writer_box = []
    monkeypatch.setattr(
        cross_modal_candidate,
        "CandidateSampleWriter",
        lambda *args: (writer_box.append(args[4]), SimpleNamespace(write=lambda *write_args: ([], {"records": []})))[1],
    )

    assert cross_modal_candidate.main([*argv, "--device", "cpu"]) == 0
    assert writer_box[-1] == torch.device("cpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert cross_modal_candidate.main(argv) == 0
    assert writer_box[-1] == torch.device("cuda")


# ------------------------------------------------- token-swap diagnostic entry


def test_token_swap_injects_the_device(monkeypatch, tmp_path):
    argv = [
        "--dev-list",
        str(_write_json(tmp_path / "dev.json", {"training": []})),
        "--emb-root",
        str(tmp_path),
        "--ckpt",
        str(tmp_path / "epoch.pt"),
        "-e",
        "env.json",
        "-c",
        "config.json",
        "-t",
        "net.json",
        "--samples-dir",
        str(tmp_path / "samples"),
    ]

    monkeypatch.setattr(token_swap, "modality_label_engine", lambda: _fake_engine())
    monkeypatch.setattr(token_swap, "DevCohortBuilder", lambda dev_list: SimpleNamespace(build=lambda: []))
    monkeypatch.setattr(token_swap, "CohortSpacingSource", lambda *args: object())
    sampler_box = []
    monkeypatch.setattr(
        token_swap,
        "TokenSwapSampler",
        lambda *args: (sampler_box.append(args[1]), SimpleNamespace(sample_cohort=lambda *sample_args: 0))[1],
    )

    assert token_swap.main([*argv, "--device", "cpu"]) == 0
    assert sampler_box[-1] == torch.device("cpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert token_swap.main(argv) == 0
    assert sampler_box[-1] == torch.device("cuda")
