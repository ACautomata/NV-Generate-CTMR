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

"""The mask watch L2 face's instrument-spec live-tree override (issue #316).

The frozen ``INSTRUMENT_SPECS`` anchor awaits re-pinning alongside the
instrument v2 calibration rerun (#310, ADR-0002) and is not touched by
diagnostic consumers. The instrument results tree moved to the v2 layout
(``nnUNetTrainer_250epochs_bf16__nnUNetPlans_v2bs8__3d_fullres_bs8``) in
series ②, so the mask watch's ``L2TrendRunner`` — the P1 dev monitor's etwt
script adapts via script rewriting, but the mask watch drives
``FrozenInstrumentCommand`` straight from the frozen spec — resolves a
nonexistent ``dataset.json`` and fails every prediction. The override reads
each challenge's live ``<trainer>__<plans>__<config>`` directory (exactly
one, three segments — never a guess, the etwt precedent) and the runner
builds its argv from the live spec.
"""

from __future__ import annotations

import pytest

from ctmr.application.generation.trend import L2TrendRunner
from ctmr.domain.instrument_spec import INSTRUMENT_SPECS

V2_DIR = "nnUNetTrainer_250epochs_bf16__nnUNetPlans_v2bs8__3d_fullres_bs8"


@pytest.fixture
def results_root(tmp_path):
    """A v2-layout results tree: one frozen-spec-matching challenge (SSA keeps its
    special plans/config in the frozen anchor but the live tree is uniform) plus
    the four the override must rewrite."""
    root = tmp_path / "nnunet_results"
    for spec in INSTRUMENT_SPECS.values():
        (root / spec.dataset_id / V2_DIR / "fold_0").mkdir(parents=True)
    return root


def test_autodiscover_rewrites_every_frozen_spec_to_the_live_tree(results_root):
    runner = L2TrendRunner(
        {challenge: str(results_root) for challenge in INSTRUMENT_SPECS},
        "raw-unused",
        "prep-unused",
        autodiscover_specs=True,
    )
    for challenge, frozen in INSTRUMENT_SPECS.items():
        live = runner.spec_of(challenge)
        assert live.trainer == "nnUNetTrainer_250epochs_bf16"
        assert live.plans == "nnUNetPlans_v2bs8"
        assert live.config == "3d_fullres_bs8"
        assert live.dataset_id == frozen.dataset_id  # dataset identity never moves
        assert live.fold == frozen.fold


def test_frozen_specs_stay_untouched_by_the_discovery(results_root):
    """The override is per-runner state: the module-level frozen anchor is not mutated."""
    before = {c: (s.trainer, s.plans, s.config) for c, s in INSTRUMENT_SPECS.items()}
    L2TrendRunner(
        {challenge: str(results_root) for challenge in INSTRUMENT_SPECS},
        "raw-unused",
        "prep-unused",
        autodiscover_specs=True,
    )
    after = {c: (s.trainer, s.plans, s.config) for c, s in INSTRUMENT_SPECS.items()}
    assert before == after


def test_ambiguous_trainer_tree_fails_loudly(results_root):
    """Two trainer dirs under one dataset is a deployment fact the runner refuses to guess."""
    (results_root / INSTRUMENT_SPECS["GLI"].dataset_id / "nnUNetTrainerOther__nnUNetPlans__3d_fullres").mkdir()
    with pytest.raises(RuntimeError, match="refuse to guess|exactly one"):
        L2TrendRunner(
            {challenge: str(results_root) for challenge in INSTRUMENT_SPECS},
            "raw-unused",
            "prep-unused",
            autodiscover_specs=True,
        )


def test_missing_dataset_dir_fails_loudly(tmp_path):
    """A half-migrated tree (dataset dir absent) is a loud preflight-style failure."""
    with pytest.raises(FileNotFoundError):
        L2TrendRunner({"GLI": str(tmp_path / "empty")}, "raw-unused", "prep-unused", autodiscover_specs=True)


def test_default_construction_keeps_the_frozen_specs():
    """Without the opt-in the runner behaves exactly as before (frozen anchor argv)."""
    runner = L2TrendRunner({}, "raw-unused", "prep-unused")
    assert runner.spec_of("GLI") is INSTRUMENT_SPECS["GLI"]
