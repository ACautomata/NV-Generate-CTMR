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

"""Cross-modal ControlNet-candidate plan-layer contract gates (issue #61 / ticket 08).

The retired cross-modal candidate plan entry's built-in self-test checks,
promoted into declarative pytest functions against the new home
``ctmr.application.generation.cross_modal.candidate``. The legacy selftest's
``_check_pairs_merge`` was never wired into its ``run()`` (dead on arrival); it is
promoted here for the first time as ``test_pairs_merge_*`` so the merged
``brats-l1-pairs/1`` triplet contract is actually gated. The module imports
torch/nibabel at module level, so it is torch-marked and runs for real in the CI
full-dependency tier (ADR-0015 §6).

The ``variant=controlnet-candidate`` marker string and the three schema strings are
frozen contract bytes (ADR-0015 §2): the byte-identity gate below pairs with the
baseline side for acceptance criterion 2.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from ctmr.application.generation.cross_modal.candidate import (  # noqa: E402
    CANDIDATE_VARIANT,
    INFER_SCHEMA,
    PAIRS_SCHEMA,
    SAMPLES_SCHEMA,
    CandidateInferenceConfig,
    CandidatePlanError,
    CandidateRunGuard,
    CandidateSamplePlanBuilder,
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
        "cfg_guidance_scale": 0.0,  # CFG off (issue #61 acceptance criterion 1)
        "grid": [256, 256, 128],
        "modality_tokens": {"t1n": 29, "t2w": 30, "t2f": 31, "t1c": 34},
        "seed_rule": "int(sha256(f'{case}|{src}->{tgt}')[:8], 16) % (2**31 - 1)",
    }
    payload.update(overrides)
    return payload


def _real_of(challenge, case, modality):
    return Path("/ctrl/raw") / challenge / case / f"{case}-{modality}.nii.gz"


def _builder(config=None):
    return CandidateSamplePlanBuilder("p3-candidate-fixture", "d" * 64, "c" * 64, "holdout", config or CandidateInferenceConfig(_config_payload()))


# ------------------------------------------------- frozen contract markers (criterion 2)


def test_variant_and_schema_markers_are_byte_frozen():
    assert CANDIDATE_VARIANT == "controlnet-candidate"
    assert INFER_SCHEMA == "brats-p3-controlnet-infer/1"
    assert SAMPLES_SCHEMA == "brats-p3-candidate-samples/1"
    assert PAIRS_SCHEMA == "brats-l1-pairs/1"


# ------------------------------------------------------------ inference-config validation


def test_inference_config_accepts_the_official_payload():
    config = CandidateInferenceConfig(_config_payload())
    assert config.scheduler == "RFlowScheduler"
    assert config.cfg_guidance_scale == 0.0
    assert config.grid == (256, 256, 128)


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema": "brats-p3-controlnet-infer/2"},  # wrong schema
        {"scheduler": "DDPMScheduler"},  # not the rectified-flow scheduler
        {"num_inference_steps": 0},
        {"cfg_guidance_scale": 10.0},  # CFG must be off
        {"cfg_guidance_scale": -1.0},
        {"strength": 0.9},  # an img2img start must not leak into the candidate
        {"modality_tokens": {"t1n": 29, "t2w": 30, "t2f": 31}},  # missing t1c
        {"modality_tokens": {"t1n": 29, "t2w": 30, "t2f": 31, "t1c": "34"}},  # string token
        {"grid": [256, 256]},  # not three dims
        {"grid": [256, 256, -128]},  # not positive
        {"seed_rule": "  "},  # empty
    ],
)
def test_inference_config_rejects_contract_violations(overrides):
    with pytest.raises(CandidatePlanError):
        CandidateInferenceConfig(_config_payload(**overrides))


# ------------------------------------------------------------------ plan coverage


def test_entries_carry_mislabel_guards_and_cover_twelve_ordered_pairs(tmp_path):
    builder = _builder()
    generated_root = tmp_path / "generated"
    entries = builder.entries(CASES, _real_of, generated_root)

    assert len(entries) == len(CASES)
    for entry in entries:
        assert entry["phase"] == "P3"
        assert entry["variant"] == CANDIDATE_VARIANT  # the controlnet-candidate mislabel guard
        assert len(entry["candidate_checkpoint_sha256"]) == 64
        assert len(entry["dm_checkpoint_sha256"]) == 64
        covered = set()
        for anchor, info in entry["anchors"].items():
            generated = info["generated"]
            assert set(generated) == {m for m in MODALITIES if m != anchor}
            for tgt, sample in generated.items():
                covered.add((anchor, tgt))
                expected = (
                    generated_root
                    / entry["challenge"]
                    / entry["case_id"]
                    / f"a{anchor}"
                    / f"{tgt}_seed{seed_of(entry['case_id'], anchor, tgt)}.nii.gz"
                )
                assert sample["path"] == str(expected)
                assert sample["seed"] == seed_of(entry["case_id"], anchor, tgt)
        assert covered == set(builder.ordered_pairs(entry["case_id"]))


# ------------------------------------------------------- pairs merge (the dead-code check)


def _stage0_records():
    records = []
    for item in CASES:
        for src in MODALITIES:
            for tgt in MODALITIES:
                if src == tgt:
                    continue
                records.append(
                    {
                        "challenge": item["sub"],
                        "case": item["case"],
                        "src_modality": src,
                        "target_modality": tgt,
                        "baseline": f"/stage0/{item['sub']}/{item['case']}/a{src}/{tgt}.nii.gz",
                        "reference": f"/refgrid/{item['sub']}/{item['case']}/{tgt}.nii.gz",
                    }
                )
    return records


def test_pairs_merge_carries_reference_baseline_candidate_triplets(tmp_path):
    builder = _builder()
    generated_root = tmp_path / "generated"
    doc = builder.pairs(_stage0_records(), generated_root)

    assert doc["schema"] == PAIRS_SCHEMA
    assert doc["variant"] == CANDIDATE_VARIANT
    assert len(doc["records"]) == 12 * len(CASES)
    pair_keys = {(r["challenge"], r["case"], r["src_modality"], r["target_modality"]) for r in doc["records"]}
    assert len(pair_keys) == len(doc["records"])  # unique per (case, src, tgt)
    for record in doc["records"]:
        assert {"reference", "baseline", "candidate"} <= set(record)
        assert record["baseline"].startswith("/stage0")
        assert record["candidate"].startswith(f"{generated_root}/")  # candidate under the candidate root


def test_pairs_merge_rejects_malformed_stage0_records(tmp_path):
    builder = _builder()
    bad = [dict(record) for record in _stage0_records()]
    del bad[0]["baseline"]  # a baseline record missing a required key
    with pytest.raises(CandidatePlanError):
        builder.pairs(bad, tmp_path / "generated")
    with pytest.raises(CandidatePlanError):
        builder.pairs({"not": "a list"}, tmp_path / "generated")


# ---------------------------------------------------------------------- run guard


def _run_guard_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "p3-controlnet.pt"
    checkpoint.write_bytes(b"frozen-p3-controlnet-fixture")
    dm = root / "p1-dm.pt"
    dm.write_bytes(b"p1-dm-fixture")
    infer_path = root / "infer.json"
    infer_path.write_text(json.dumps(_config_payload()))
    cn_sha = CandidateRunGuard.file_sha256(checkpoint)
    dm_sha = CandidateRunGuard.file_sha256(dm)
    infer_sha = CandidateRunGuard.file_sha256(infer_path)

    def record(**overrides):
        payload = {
            "run_id": "p3-candidate-fixture",
            "phase": "P3",
            "variant": CANDIDATE_VARIANT,
            "status": "frozen",
            "selection": {"checkpoint": {"path": str(checkpoint), "sha256": cn_sha}},
            "upstream": {"checkpoint": {"path": str(dm), "sha256": dm_sha}},
            "configs": [{"role": "inference", "path": str(infer_path), "sha256": infer_sha}],
        }
        payload.update(overrides)
        return payload

    return checkpoint, dm, dm_sha, infer_path, infer_sha, record


def test_run_guard_positive_path_returns_the_pinned_candidate_checkpoint(tmp_path):
    checkpoint, _dm, _dm_sha, infer_path, _infer_sha, record = _run_guard_fixture(tmp_path / "run-guard")
    assert CandidateRunGuard(record(), infer_path).check() == checkpoint


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record, dm, dm_sha, infer_sha, root: record(variant="stage0-baseline"),  # not the candidate variant
        lambda record, dm, dm_sha, infer_sha, root: record(status="open"),  # unfrozen
        lambda record, dm, dm_sha, infer_sha, root: record(selection={}),  # no candidate checkpoint
        lambda record, dm, dm_sha, infer_sha, root: record(
            selection={"checkpoint": {"path": str(dm), "sha256": dm_sha}}
        ),  # zero-training variant in disguise
        lambda record, dm, dm_sha, infer_sha, root: record(selection={"checkpoint": {"path": str(dm), "sha256": "0" * 64}}),
        lambda record, dm, dm_sha, infer_sha, root: record(configs=[]),  # no inference config pinned
        lambda record, dm, dm_sha, infer_sha, root: record(
            configs=[
                {"role": "inference", "path": str(root / "infer.json"), "sha256": infer_sha},
                {"role": "inference", "path": str(root / "other.json"), "sha256": "0" * 64},
            ]
        ),  # two inference configs pinned
    ],
)
def test_run_guard_rejects_contract_violations(tmp_path, mutate):
    root = tmp_path / "run-guard"
    _checkpoint, dm, dm_sha, infer_path, infer_sha, record = _run_guard_fixture(root)
    with pytest.raises(CandidatePlanError):
        CandidateRunGuard(mutate(record, dm, dm_sha, infer_sha, root), infer_path).check()


def test_run_guard_rejects_infer_config_drift_from_the_pinned_cfg0_provenance(tmp_path):
    root = tmp_path / "run-guard"
    _checkpoint, _dm, _dm_sha, _infer_path, _infer_sha, record = _run_guard_fixture(root)
    drifted = root / "drifted.json"
    drifted.write_text(json.dumps(_config_payload(cfg_guidance_scale=10.0)))
    with pytest.raises(CandidatePlanError):
        CandidateRunGuard(record(), drifted).check()
