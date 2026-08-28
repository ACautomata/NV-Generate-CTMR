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

"""Mask holdout sample-plan gates: cohort sharding, spacing formula, manifest shape (ticket 09).

The plan-layer pieces of ``ctmr.application.generation.mask.sample`` are pure
logic over the pinned phase manifest; the gates below exercise them on synthetic
fixtures without a GPU. The #52 spacing-companion formula
(``spacing_i = pixdim_i * shape_i / GRID_i``) and the deterministic shard
arithmetic are value-frozen: holdout manifests already in the controlled
storage were produced with them. Torch-marked (module imports torch through
``sample``), CPU-real in the CI full-dependency tier (ADR-0015 §6).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

from ctmr.application.generation.mask.sample import (
    GRID,
    CandidateSampler,
    HoldoutCohortBuilder,
    HoldoutMaskSource,
    HoldoutSampleWriter,
    HoldoutSpacingSource,
)

pytestmark = pytest.mark.torch


def _manifest():
    return {
        "challenges": {
            "GLI": {"source_dir": "", "cases": {"holdout": [f"BraTS-GLI-{i:04d}-000" for i in range(1, 6)]}},
            "MEN": {"source_dir": "", "cases": {"holdout": [f"BraTS-MEN-{i:04d}-000" for i in range(1, 4)]}},
            "SSA": {"source_dir": "", "cases": {"holdout": ["BraTS-SSA-0001-000"]}},
        }
    }


# --------------------------------------------------------------------- cohort / sharding


def test_holdout_cohort_lists_every_manifest_case_in_order():
    cohort = HoldoutCohortBuilder(_manifest()).build()
    assert [(item["sub"], item["case"]) for item in cohort] == [("GLI", f"BraTS-GLI-{i:04d}-000") for i in range(1, 6)] + [
        ("MEN", f"BraTS-MEN-{i:04d}-000") for i in range(1, 4)
    ] + [("SSA", "BraTS-SSA-0001-000")]


def test_holdout_sharding_takes_every_nth_case_of_the_deterministic_order():
    manifest = _manifest()
    whole = HoldoutCohortBuilder(manifest).build()
    shards = [HoldoutCohortBuilder(manifest, shard=i, num_shards=3).build() for i in range(3)]
    # every case lands in exactly one shard; the shards partition the cohort
    flattened = [item for shard in shards for item in shard]
    assert sorted((item["sub"], item["case"]) for item in flattened) == sorted((item["sub"], item["case"]) for item in whole)
    assert all(len(shard) <= 3 for shard in shards)


def test_holdout_filters_limit_challenge_and_only_cases():
    manifest = _manifest()
    cohort = HoldoutCohortBuilder(manifest, limit=2, only_challenge="GLI", only_cases=["BraTS-GLI-0002-000", "BraTS-GLI-0003-000"]).build()
    assert [(item["sub"], item["case"]) for item in cohort] == [("GLI", "BraTS-GLI-0002-000"), ("GLI", "BraTS-GLI-0003-000")]


# --------------------------------------------------------------------- spacing + masks


def test_holdout_spacing_source_replicates_the_companion_formula(tmp_path):
    """spacing_i = pixdim_i * shape_i / GRID_i (issue #52), from the raw t1n header."""
    challenge, case = "GLI", "BraTS-GLI-0001-000"
    case_dir = tmp_path / challenge / case
    case_dir.mkdir(parents=True)
    zooms, shape = (1.6, 0.8, 2.5), (300, 130, 80)  # xyz
    volume = np.zeros(shape, dtype=np.float32)
    image = nib.Nifti1Image(volume, np.diag([zooms[0], zooms[1], zooms[2], 1.0]))
    image.header.set_zooms(zooms)
    nib.save(image, str(case_dir / f"{case}-t1n.nii.gz"))

    spacings = HoldoutSpacingSource(tmp_path, _manifest())
    assert spacings.directory_of(case) == Path(challenge) / case  # the raw-relative case directory
    expected = [zooms[i] * shape[i] / GRID[i] for i in range(3)]
    # nibabel stores zooms as float32, so the header round-trip carries float32 noise
    assert spacings.spacing_of(case) == pytest.approx(expected)


def test_holdout_mask_source_lands_on_the_phase_label_root():
    masks = HoldoutMaskSource("/phase", _manifest())
    assert str(masks.path_of("BraTS-MEN-0002-000")) == "/phase/labels/MEN/BraTS-MEN-0002-000/BraTS-MEN-0002-000-combined.nii.gz"


# --------------------------------------------------------------------- seed rule (frozen)


def test_seed_rule_matches_the_dev_sidecar_convention():
    """sha256(case|modality) truncated -- holdout names must match the dev-trend sample names."""
    for modality in ("t1n", "t1c", "t2w", "t2f"):
        expected = int(hashlib.sha256(f"BraTS-GLI-0001-000|{modality}".encode()).hexdigest()[:8], 16) % (2**31 - 1)
        assert CandidateSampler.seed_of("BraTS-GLI-0001-000", modality) == expected


# ------------------------------------------------------------------ samples manifest shape


def test_samples_manifest_entry_shape(tmp_path, monkeypatch):
    """The L2 assembly-samples manifest keys, per case (one condition mask, four modalities)."""
    challenge, case = "GLI", "BraTS-GLI-0001-000"
    manifest = _manifest()
    raw_root = tmp_path / "raw"
    (raw_root / challenge / case).mkdir(parents=True)
    # the spacing source really loads the t1n header (the #52 formula), so write a real NIfTI
    t1n = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.diag([1.0, 1.0, 1.0, 1.0]))
    t1n.header.set_zooms((1.0, 1.0, 1.0))
    nib.save(t1n, str(raw_root / challenge / case / f"{case}-t1n.nii.gz"))
    for modality in ("t1c", "t2w", "t2f"):
        (raw_root / challenge / case / f"{case}-{modality}.nii.gz").write_bytes(b"nii")  # path-only for real_paths
    (tmp_path / "labels" / challenge / case).mkdir(parents=True)

    class _FakeSampler:
        """Stands in for the GPU sampler; the writer's manifest assembly is the gate."""

        def __init__(self, *args):
            pass

        @staticmethod
        def load_models(checkpoint_path):
            assert checkpoint_path == "/ckpt/epoch_30.pt"
            return object(), object()

        @staticmethod
        def load_condition_mask(mask_source, case_id, device):
            return object()

        @staticmethod
        def seed_of(case_id, modality):
            return CandidateSampler.seed_of(case_id, modality)

        def sample_one(self, *args):
            return np.zeros((256, 256, 128), dtype=np.int16)

    monkeypatch.setattr("ctmr.application.generation.mask.sample.CandidateSampler", _FakeSampler)
    run_record = {"selection": {"checkpoint": {"path": "/ckpt/epoch_30.pt", "epoch": 30}}}
    writer = HoldoutSampleWriter(SimpleNamespace(), run_record, raw_root, tmp_path / "out", torch.device("cpu"), print)
    entries = writer.write([{"sub": challenge, "case": case}], HoldoutSpacingSource(raw_root, manifest), HoldoutMaskSource(tmp_path, manifest))

    assert len(entries) == 1
    entry = entries[0]
    assert list(entry) == ["case_id", "challenge", "phase", "condition_mask", "samples", "real_paths"]
    assert entry["phase"] == "P2"  # the frozen assembly marker the L2 acceptor keys on
    assert set(entry["samples"]) == {"t1n", "t1c", "t2w", "t2f"}
    assert set(entry["real_paths"]) == {"t1n", "t1c", "t2w", "t2f"}
    assert entry["samples"]["t1n"]["seed"] == CandidateSampler.seed_of(case, "t1n")
    assert json.dumps(entries)  # the manifest is JSON-serialisable end to end
