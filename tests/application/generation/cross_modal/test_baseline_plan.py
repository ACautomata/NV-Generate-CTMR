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

"""Baseline plan-layer contract gates (issue #60 / ticket 08).

These are the retired cross-modal baseline plan entry's built-in self-test
checks, promoted from the self-check ``failures`` list into declarative pytest
functions against the new home ``ctmr.application.generation.cross_modal.baseline``.
The plan layer is pure logic, but the module imports torch/monai/nibabel at module
level (the generation driver lives alongside), so the module is torch-marked and
runs for real in the CI full-dependency tier (ADR-0015 §6).

The ``variant=stage0-baseline`` marker string and the two schema strings are frozen
contract bytes (ADR-0015 §2): the byte-identity gate below is acceptance criterion 2.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from ctmr.application.generation.cross_modal.baseline import (  # noqa: E402
    BASELINE_VARIANT,
    INFER_SCHEMA,
    PAIRS_SCHEMA,
    BaselineInferenceConfig,
    BaselinePlanError,
    BaselineRunGuard,
    BaselineSamplePlanBuilder,
)
from ctmr.application.generation.cross_modal.plan import MODALITIES, seed_of  # noqa: E402

pytestmark = pytest.mark.torch

CASES = [
    {"sub": "FIXGLI", "case": "FIXGLI-0200-000"},
    {"sub": "FIXSSA", "case": "FIXSSA-0200-000"},
]


def _config_payload(**overrides):
    payload = {
        "schema": INFER_SCHEMA,
        "scheduler": "RFlowScheduler",
        "num_inference_steps": 30,
        "cfg_guidance_scale": 10.0,
        "strength": 0.9,
        "grid": [256, 256, 128],
        "modality_tokens": {"t1n": 29, "t2w": 30, "t2f": 31, "t1c": 34},
        "seed_rule": "int(sha256(f'{case}|{src}->{tgt}')[:8], 16) % (2**31 - 1)",
    }
    payload.update(overrides)
    return payload


def _real_of(challenge, case, modality):
    return Path("/ctrl/raw") / challenge / case / f"{case}-{modality}.nii.gz"


def _builder(config=None):
    return BaselineSamplePlanBuilder("baseline-fixture", "d" * 64, "holdout", config or BaselineInferenceConfig(_config_payload()))


# ------------------------------------------------- frozen contract markers (criterion 2)


def test_variant_and_schema_markers_are_byte_frozen():
    assert BASELINE_VARIANT == "stage0-baseline"
    assert INFER_SCHEMA == "brats-p3-stage0-infer/1"
    assert PAIRS_SCHEMA == "brats-p3-stage0-pairs/1"


def test_seed_rule_is_shared_and_direction_sensitive():
    # baseline and candidate must share the identical (case, src, tgt) noise schedule
    assert seed_of("FIXGLI-0200-000", "t1n", "t1c") == seed_of("FIXGLI-0200-000", "t1n", "t1c")
    assert seed_of("FIXGLI-0200-000", "t1n", "t1c") != seed_of("FIXGLI-0200-000", "t1c", "t1n")


# ------------------------------------------------------------ inference-config validation


def test_inference_config_accepts_the_official_payload():
    config = BaselineInferenceConfig(_config_payload())
    assert config.scheduler == "RFlowScheduler"
    assert config.grid == (256, 256, 128)
    assert config.strength == 0.9


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema": "brats-p3-stage0-infer/2"},  # wrong schema
        {"scheduler": "DDPMScheduler"},  # not the rectified-flow scheduler
        {"num_inference_steps": 0},
        {"cfg_guidance_scale": -1.0},
        {"strength": 1.0},  # erases the src latent
        {"strength": 0.0},  # copies the src latent
        {"modality_tokens": {"t1n": 29, "t2w": 30, "t2f": 31}},  # missing t1c
        {"modality_tokens": {"t1n": 29, "t2w": 30, "t2f": 31, "t1c": "34"}},  # string token
        {"grid": [256, 256]},  # not three dims
        {"grid": [256, 256, -128]},  # not positive
        {"seed_rule": "  "},  # empty
    ],
)
def test_inference_config_rejects_contract_violations(overrides):
    with pytest.raises(BaselinePlanError):
        BaselineInferenceConfig(_config_payload(**overrides))


# ------------------------------------------------------------------ plan coverage


def test_entries_carry_mislabel_guards_and_cover_twelve_ordered_pairs(tmp_path):
    builder = _builder()
    generated_root = tmp_path / "generated"
    entries = builder.entries(CASES, _real_of, generated_root)

    assert len(entries) == len(CASES)
    for entry in entries:
        assert entry["phase"] == "P3"
        assert entry["variant"] == BASELINE_VARIANT  # the stage0-baseline mislabel guard
        assert len(entry["dm_checkpoint_sha256"]) == 64
        covered = set()
        for anchor, info in entry["anchors"].items():
            generated = info["generated"]
            assert set(generated) == {m for m in MODALITIES if m != anchor}  # the other three modalities
            for tgt, sample in generated.items():
                covered.add((anchor, tgt))
                expected = generated_root / entry["challenge"] / entry["case_id"] / f"a{anchor}" / f"{tgt}_seed{seed_of(entry['case_id'], anchor, tgt)}.nii.gz"
                assert sample["path"] == str(expected)
                assert sample["seed"] == seed_of(entry["case_id"], anchor, tgt)
        assert covered == set(builder.ordered_pairs(entry["case_id"]))  # all 12 ordered pairs per case


def test_pairs_document_declares_schema_and_matches_the_anchor_volume_set(tmp_path):
    builder = _builder()
    generated_root = tmp_path / "generated"
    entries = builder.entries(CASES, _real_of, generated_root)
    pairs_doc = builder.pairs(CASES, _real_of, generated_root)

    assert pairs_doc["schema"] == PAIRS_SCHEMA
    assert pairs_doc["variant"] == BASELINE_VARIANT
    records = pairs_doc["records"]
    assert len(records) == 12 * len(CASES)  # 12 ordered pairs per case
    pair_keys = {(r["challenge"], r["case"], r["src_modality"], r["target_modality"]) for r in records}
    assert len(pair_keys) == len(records)  # unique per (case, src, tgt)
    baseline_paths = {r["baseline"] for r in records}
    anchor_paths = {s["path"] for entry in entries for info in entry["anchors"].values() for s in info["generated"].values()}
    assert baseline_paths == anchor_paths  # pairs and anchors reference the identical volume set
    for record in records:
        assert record["reference"].endswith(f"-{record['target_modality']}.nii.gz")  # real target volume


# ---------------------------------------------------------------------- run guard


def _run_guard_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "p1-dm.pt"
    checkpoint.write_bytes(b"frozen-p1-dm-fixture")
    infer_path = root / "infer.json"
    infer_path.write_text(json.dumps(_config_payload()))
    dm_sha = BaselineRunGuard.file_sha256(checkpoint)
    infer_sha = BaselineRunGuard.file_sha256(infer_path)

    def record(**overrides):
        payload = {
            "run_id": "baseline-fixture",
            "phase": "P3",
            "variant": BASELINE_VARIANT,
            "status": "frozen",
            "selection": {"checkpoint": {"path": str(checkpoint), "sha256": dm_sha}},
            "upstream": {"checkpoint": {"path": str(checkpoint), "sha256": dm_sha}},
            "configs": [{"role": "inference", "path": str(infer_path), "sha256": infer_sha}],
        }
        payload.update(overrides)
        return payload

    return checkpoint, infer_path, infer_sha, record


def test_run_guard_positive_path_returns_the_pinned_checkpoint(tmp_path):
    checkpoint, infer_path, _infer_sha, record = _run_guard_fixture(tmp_path / "run-guard")
    assert BaselineRunGuard(record(), infer_path).check() == checkpoint


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record, checkpoint, infer_sha, root: record(variant="controlnet-candidate"),  # not the baseline variant
        lambda record, checkpoint, infer_sha, root: record(status="open"),  # unfrozen
        lambda record, checkpoint, infer_sha, root: record(selection={"checkpoint": {"path": str(checkpoint), "sha256": "0" * 64}}),
        lambda record, checkpoint, infer_sha, root: record(upstream={"checkpoint": {"path": str(checkpoint), "sha256": "0" * 64}}),
        lambda record, checkpoint, infer_sha, root: record(configs=[]),  # no inference config pinned
        lambda record, checkpoint, infer_sha, root: record(
            configs=[
                {"role": "inference", "path": str(root / "infer.json"), "sha256": infer_sha},
                {"role": "inference", "path": str(root / "other.json"), "sha256": "0" * 64},
            ]
        ),  # two inference configs pinned
    ],
)
def test_run_guard_rejects_contract_violations(tmp_path, mutate):
    root = tmp_path / "run-guard"
    _checkpoint, infer_path, infer_sha, record = _run_guard_fixture(root)
    with pytest.raises(BaselinePlanError):
        BaselineRunGuard(mutate(record, root / "p1-dm.pt", infer_sha, root), infer_path).check()


def test_run_guard_rejects_infer_config_drift_from_the_pinned_provenance(tmp_path):
    root = tmp_path / "run-guard"
    _checkpoint, _infer_path, _infer_sha, record = _run_guard_fixture(root)
    drifted = root / "drifted.json"
    drifted.write_text(json.dumps(_config_payload(cfg_guidance_scale=0.0)))
    with pytest.raises(BaselinePlanError):
        BaselineRunGuard(record(), drifted).check()
