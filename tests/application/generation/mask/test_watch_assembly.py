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

"""The mask dev-eval watch assembly contract (issue #316).

The watch main() hands ``WatchEngine`` a ``sampler_factory``; the engine's
contract (ADR-0019 §5, #279) is a positional ``(checkpoint_path, out_dir)``
call. The mask ``CandidateSampler.generate_cohort`` signature is
``(checkpoint_path, cohort, spacings, masks, out_dir)`` — a partial binding
that lets the engine's second positional argument land on ``cohort`` raises
``TypeError: got multiple values for argument 'cohort'`` on every eval point
(first end-to-end consumer: the #316 P2 retrain watch). This gate drives
``main()`` with the engine replaced by a recorder and asserts the factory
delivers each argument to its own name.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ctmr.application.generation.mask import monitor

DEV_ENTRIES = [
    {
        "image": "embeddings/x/case1-t1n_emb.nii.gz",
        "label": "labels/x/case1-combined.nii.gz",
        "spacing": [1.0, 1.0, 1.0],
        "modality": "mri_t1_skull_stripped",
        "fold": 0,
        "sub": "GLI",
        "case": "case1",
    },
    {
        "image": "embeddings/x/case2-t1n_emb.nii.gz",
        "label": "labels/x/case2-combined.nii.gz",
        "spacing": [1.0, 1.0, 1.0],
        "modality": "mri_t1_skull_stripped",
        "fold": 1,
        "sub": "GLI",
        "case": "case2",
    },
]


class RecordingSampler:
    """Stands in for CandidateSampler; records the generate_cohort call shape."""

    def __init__(self):
        self.calls = []

    def generate_cohort(self, checkpoint_path, cohort, spacings, masks, out_dir):
        self.calls.append({"checkpoint_path": checkpoint_path, "cohort": cohort, "spacings": spacings, "masks": masks, "out_dir": out_dir})
        return ["sample"]


@pytest.fixture
def sampler(monkeypatch, tmp_path):
    """Drive main() in watch mode with every heavy collaborator faked; yield the recording sampler."""
    dev_list = tmp_path / "dev.json"
    dev_list.write_text(json.dumps({"training": DEV_ENTRIES}))
    eval_root = tmp_path / "eval"
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()

    fake = RecordingSampler()
    monkeypatch.setattr(monitor, "mask_engine", lambda: SimpleNamespace(load_config=lambda *a: SimpleNamespace()))
    monkeypatch.setattr(monitor, "resolve_device", lambda device: "cpu")
    monkeypatch.setattr(monitor, "MrTrendFeatures", lambda device: object())
    monkeypatch.setattr(monitor, "RealReferenceBank", lambda *a, **k: SimpleNamespace(build=lambda: {}))
    monkeypatch.setattr(monitor, "CandidateSampler", lambda *a, **k: fake)

    def fake_watch_engine(**kwargs):
        recorded = SimpleNamespace(**kwargs)

        def run(cohort_file=None):
            recorded.sampler_factory("epoch_5.pt", eval_root / "epoch_5" / "samples")
            return 0

        return SimpleNamespace(run=run)

    monkeypatch.setattr(monitor, "WatchEngine", lambda **kwargs: fake_watch_engine(**kwargs))

    argv = [
        "watch",
        "--ckpt-dir",
        str(ckpt_dir),
        "--eval-root",
        str(eval_root),
        "--dev-list",
        str(dev_list),
        "--raw-root",
        str(tmp_path / "raw"),
        "--label-root",
        str(tmp_path / "raw"),
        "-e",
        "unused.json",
        "-c",
        "unused.json",
        "-t",
        "unused.json",
        "--skip-l2",
        "--device",
        "cpu",
    ]
    assert monitor.main(argv) == 0
    return fake


def test_sampler_factory_delivers_checkpoint_path_and_out_dir_to_their_names(sampler):
    """The engine's positional (checkpoint_path, out_dir) call must land each argument on its own name."""
    assert len(sampler.calls) == 1
    call = sampler.calls[0]
    assert call["checkpoint_path"] == "epoch_5.pt"
    assert str(call["out_dir"]).endswith("samples")
    # the dev-side cohort / spacing / mask companions ride by keyword, not by positional collision
    assert [item["case"] for item in call["cohort"]] == ["case1"]
    assert call["spacings"].spacing_of("case1") == [1.0, 1.0, 1.0]
    assert str(call["masks"].path_of("case1")).endswith("case1-combined.nii.gz")
