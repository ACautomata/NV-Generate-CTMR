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

"""P3 image-conditioned ControlNet candidate sample-plan assembly (issue #61).

Pure manifest logic — stdlib only, no torch/nibabel — so the plan layer is
testable off the DCU box. The generation driver (``scripts.brats_p3_controlnet_generate``)
imports this module and feeds it resolved paths. The stage-0 sibling
(``brats_p3_stage0_manifest``) writes the *baseline* side of the L1 pairs; this
module writes the *candidate* side and merges the two into the ``brats-l1-pairs/1``
triplets the L1 ``P3DirectionAssessor`` consumes.

Outputs two controlled manifests per run side:

- ``samples.json``: a top-level list whose entries match the L2 final-acceptance
  ``P3FourAnchorPlan`` layout (``phase=P3`` + per-anchor ``real``/``generated``
  channels) so ``nnunet_l2_final_acceptance assemble --phase P3`` consumes it
  directly. Each entry carries the explicit ``variant`` / ``run_id`` /
  ``candidate_checkpoint_sha256`` / ``dm_checkpoint_sha256`` markers so a trained
  candidate manifest can never be mistaken for a stage-0 baseline manifest.
- ``pairs.json``: the L1-side flat view with a one-record-per-ordered-direction
  ``reference`` (real target on the generation grid), ``baseline`` (stage-0
  zero-training volume) and ``candidate`` (this ControlNet volume) triple. The
  triplets are consumed directly by ``brats_l1_quantitative evaluate --pairs``.

The candidate recipe is frozen by the P1-DM + P3 ControlNet (issue #61 acceptance
criterion 1-2): DM/VAE frozen, ``conditioning_embedding_in_channels=4`` (src
latent, no mask) and CFG=0 (``cfg_guidance_scale == 0`` — no latent unconditional
branch). The run guard refuses anything that deviates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MODALITIES = ("t1n", "t1c", "t2w", "t2f")
INFER_SCHEMA = "brats-p3-controlnet-infer/1"
SAMPLES_SCHEMA = "brats-p3-candidate-samples/1"
PAIRS_SCHEMA = "brats-l1-pairs/1"
CANDIDATE_VARIANT = "controlnet-candidate"
SEED_MODULUS = 2**31 - 1


class P3CandidatePlanError(Exception):
    """Raised when the inference config or the assembled plan breaks the P3 candidate contract."""


class P3CandidateInferenceConfig:
    """The official P3 candidate inference configuration (configs/config_p3_controlnet_infer.json).

    The candidate is a trained ControlNet that conditions on a 4ch src latent and
    denoises to the target modality from noise with CFG OFF. It differs from the
    stage-0 config (``brats-p3-stage0-infer/1``): no ``strength`` (the candidate is
    not an interpolated img2img start) and ``cfg_guidance_scale`` is pinned to 0.
    """

    def __init__(self, payload):
        if payload.get("schema") != INFER_SCHEMA:
            raise P3CandidatePlanError(f"P3 candidate inference config schema must be {INFER_SCHEMA!r}")
        self.scheduler = self._text(payload, "scheduler")
        if self.scheduler != "RFlowScheduler":
            raise P3CandidatePlanError("P3 candidate only runs on the RFlow scheduler (rectified flow)")
        self.num_inference_steps = self._positive_int(payload, "num_inference_steps")
        self.cfg_guidance_scale = self._number(payload, "cfg_guidance_scale")
        if self.cfg_guidance_scale != 0:
            raise P3CandidatePlanError(
                "P3 candidate cfg_guidance_scale must be 0 (default CFG off, zero latent unconditional branch; "
                "issue #61 acceptance criterion 1)"
            )
        if "strength" in payload:
            raise P3CandidatePlanError(
                "P3 candidate inference config must not carry strength (it conditions from noise, not an img2img start)"
            )
        self.grid = self._grid(payload)
        self.modality_tokens = payload.get("modality_tokens")
        if not isinstance(self.modality_tokens, dict) or set(self.modality_tokens) != set(MODALITIES):
            raise P3CandidatePlanError(f"modality_tokens must cover exactly {list(MODALITIES)} (P1-DM tokens)")
        if not all(isinstance(token, int) and token > 0 for token in self.modality_tokens.values()):
            raise P3CandidatePlanError("modality tokens must be positive integers (P1-DM class labels)")
        self.seed_rule = self._text(payload, "seed_rule")

    @classmethod
    def from_path(cls, path):
        return cls(json.loads(Path(path).read_text()))

    @staticmethod
    def _text(payload, key):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise P3CandidatePlanError(f"P3 candidate inference config needs a non-empty {key}")
        return value

    @staticmethod
    def _positive_int(payload, key):
        value = payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise P3CandidatePlanError(f"P3 candidate inference config {key} must be a positive integer")
        return value

    @staticmethod
    def _number(payload, key):
        value = payload.get(key)
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise P3CandidatePlanError(f"P3 candidate inference config {key} must be numeric")
        return float(value)

    @staticmethod
    def _grid(payload):
        value = payload.get("grid")
        if not isinstance(value, list | tuple) or len(value) != 3 or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in value):
            raise P3CandidatePlanError("P3 candidate inference config grid must be three positive integers (SRI24 256x256x128)")
        return tuple(value)

    @staticmethod
    def seed_of(case, src, tgt):
        """Deterministic per (case, direction) seed: sha256('<case>|<src>-><tgt>') truncated.

        Identical to the stage-0 seed rule so baseline and candidate share the same
        (case, src, tgt) noise schedule and are directly comparable.
        """
        digest = hashlib.sha256(f"{case}|{src}->{tgt}".encode()).hexdigest()[:8]
        return int(digest, 16) % SEED_MODULUS


class P3CandidateRunGuard:
    """Checks the frozen P3 candidate run record and pins the ControlNet checkpoint + inference config.

    The generation-time recheck of the phase-run contract markers (issue #61
    acceptance criterion 1-3): the run must be the trained P3 ``controlnet-candidate``
    variant, its selection must pin the trained ControlNet checkpoint (never the
    upstream P1-DM — that would be a stage-0 baseline in disguise), and the inference
    config used on the command line must byte-match the ``role=inference`` config the
    run pinned at init.
    """

    def __init__(self, run_record, infer_config_path):
        self._record = run_record
        self._infer_config_path = Path(infer_config_path)

    def check(self):
        record = self._record
        if record.get("phase") != "P3" or record.get("variant") != CANDIDATE_VARIANT:
            raise P3CandidatePlanError(
                f"run {record.get('run_id')} is not a P3 {CANDIDATE_VARIANT} run; "
                "candidate generation must hang off its own trained ControlNet run record"
            )
        if record.get("status") != "frozen":
            raise P3CandidatePlanError(f"run {record['run_id']} is {record.get('status')}; generate samples only after the freeze")
        selection = record.get("selection") or {}
        checkpoint = selection.get("checkpoint") or {}
        if not checkpoint.get("path") or not checkpoint.get("sha256"):
            raise P3CandidatePlanError(f"run {record['run_id']} selection carries no candidate checkpoint (the trained ControlNet)")
        path = Path(checkpoint["path"])
        if not path.is_file():
            raise P3CandidatePlanError(f"candidate ControlNet checkpoint missing: {path}")
        if self.file_sha256(path) != checkpoint["sha256"]:
            raise P3CandidatePlanError(f"candidate ControlNet checkpoint changed on disk: {path}")
        # anti-confusion (issue #61 acceptance criterion 3): a trained candidate pins its own
        # ControlNet checkpoint, never the upstream P1-DM (that would be a stage-0 baseline in
        # disguise, and load_image_models would KeyError on the DM ckpt's missing state_dict)
        upstream = record.get("upstream") or {}
        upstream_sha = upstream.get("checkpoint", {}).get("sha256")
        if upstream_sha and checkpoint["sha256"] == upstream_sha:
            raise P3CandidatePlanError(
                f"run {record['run_id']} selection pins the upstream P1-DM checkpoint; "
                "a controlnet-candidate must pin its own trained ControlNet checkpoint"
            )
        inference_entries = [entry for entry in record.get("configs", []) if entry.get("role") == "inference"]
        if len(inference_entries) != 1:
            raise P3CandidatePlanError(
                "the P3 candidate run must pin exactly one --config inference=<P3 candidate inference config> "
                "(the recorded inference provenance, issue #61 acceptance criterion 1)"
            )
        if self.file_sha256(self._infer_config_path) != inference_entries[0]["sha256"]:
            raise P3CandidatePlanError(
                f"--infer-config sha256 does not match the config pinned by run {record['run_id']}; "
                "regenerate with the recorded official P3 candidate inference config"
            )
        return path

    @staticmethod
    def file_sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()


class P3CandidateSamplePlanBuilder:
    """Builds the L2-compatible four-anchor entries and the L1 candidate triplets.

    One real anchor per modality per case; each anchor round conditions the other
    three modalities (12 ordered src->tgt pairs per case). The candidate pairs are
    the merge of the stage-0 baseline/reference volumes with this candidate volume,
    emitted as ``brats-l1-pairs/1`` for the L1 ``P3DirectionAssessor``. Every carried
    path marker keeps the ``controlnet-candidate`` variant explicit.
    """

    def __init__(self, run_id, dm_checkpoint_sha256, candidate_checkpoint_sha256, side, config):
        self._run_id = run_id
        self._dm_sha = dm_checkpoint_sha256
        self._candidate_sha = candidate_checkpoint_sha256
        self._side = side
        self._config = config

    def generated_path(self, generated_root, challenge, case, src, tgt):
        seed = self._config.seed_of(case, src, tgt)
        return Path(generated_root) / challenge / case / f"a{src}" / f"{tgt}_seed{seed}.nii.gz"

    def entries(self, cohort, real_of, generated_root):
        """``real_of(challenge, case, modality) -> path`` maps each case's real volumes."""
        entries = []
        for item in cohort:
            challenge, case = item["sub"], item["case"]
            anchors = {}
            for anchor in MODALITIES:
                generated = {}
                for tgt in MODALITIES:
                    if tgt == anchor:
                        continue
                    path = self.generated_path(generated_root, challenge, case, anchor, tgt)
                    generated[tgt] = {"path": str(path), "seed": self._config.seed_of(case, anchor, tgt)}
                anchors[anchor] = {"real": str(real_of(challenge, case, anchor)), "generated": generated}
            entries.append(
                {
                    "case_id": case,
                    "challenge": challenge,
                    "phase": "P3",
                    "variant": CANDIDATE_VARIANT,
                    "run_id": self._run_id,
                    "side": self._side,
                    "candidate_checkpoint_sha256": self._candidate_sha,
                    "dm_checkpoint_sha256": self._dm_sha,
                    "anchors": anchors,
                }
            )
        return entries

    def pairs(self, stage0_records, generated_root):
        """Merge stage-0 baseline/reference with this candidate volume into ``brats-l1-pairs/1``.

        ``stage0_records`` is the stage-0 ``brats-p3-stage0-pairs/1`` ``records`` list:
        each carries ``challenge``/``case``/``src_modality``/``target_modality`` plus the
        stage-0 ``baseline`` and ``reference`` (real target on the generation grid).
        """
        if not isinstance(stage0_records, list):
            raise P3CandidatePlanError("stage-0 pairs records must be a list")
        records = []
        for record in stage0_records:
            missing = {"challenge", "case", "src_modality", "target_modality", "baseline", "reference"} - set(record)
            if missing:
                raise P3CandidatePlanError(f"stage-0 pair record missing {sorted(missing)}: {record.get('case')}")
            challenge, case = record["challenge"], record["case"]
            src, tgt = record["src_modality"], record["target_modality"]
            candidate = self.generated_path(generated_root, challenge, case, src, tgt)
            records.append(
                {
                    "challenge": challenge,
                    "case": case,
                    "src_modality": src,
                    "target_modality": tgt,
                    "seed": self._config.seed_of(case, src, tgt),
                    "reference": record["reference"],
                    "baseline": record["baseline"],
                    "candidate": str(candidate),
                }
            )
        return {
            "schema": PAIRS_SCHEMA,
            "run_id": self._run_id,
            "variant": CANDIDATE_VARIANT,
            "side": self._side,
            "candidate_checkpoint_sha256": self._candidate_sha,
            "records": records,
        }

    def ordered_pairs(self, case):
        """The 12 ordered src!=tgt modality pairs this plan must cover per case."""
        return [(src, tgt) for src in MODALITIES for tgt in MODALITIES if src != tgt]


class P3CandidatePlanSelfTest:
    """Fixture-driven checks with synthetic non-subject ids (stdlib only)."""

    CASES = [
        {"sub": "FIXGLI", "case": "FIXGLI-0200-000"},
        {"sub": "FIXSSA", "case": "FIXSSA-0200-000"},
    ]

    def __init__(self, workdir):
        self._workdir = Path(workdir)
        self.failures = []

    def expect_reject(self, action, label):
        try:
            action()
        except P3CandidatePlanError:
            return
        self.failures.append(f"expected rejection but succeeded: {label}")

    def _config_payload(self, **overrides):
        payload = {
            "schema": INFER_SCHEMA,
            "scheduler": "RFlowScheduler",
            "num_inference_steps": 30,
            "cfg_guidance_scale": 0.0,
            "grid": [256, 256, 128],
            "modality_tokens": {"t1n": 29, "t2w": 30, "t2f": 31, "t1c": 34},
            "seed_rule": "int(sha256(f'{case}|{src}->{tgt}')[:8], 16) % (2**31 - 1)",
        }
        payload.update(overrides)
        return payload

    def run(self):
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._check_config_validation()
        self._check_plan_coverage()
        self._check_run_guard()
        return self.failures

    def _check_config_validation(self):
        P3CandidateInferenceConfig(self._config_payload())  # positive path
        for label, overrides in (
            ("wrong schema", {"schema": "brats-p3-controlnet-infer/2"}),
            ("ddpm scheduler", {"scheduler": "DDPMScheduler"}),
            ("zero steps", {"num_inference_steps": 0}),
            ("cfg != 0", {"cfg_guidance_scale": 10.0}),
            ("cfg negative", {"cfg_guidance_scale": -1.0}),
            ("carries strength (img2img start leaked in)", {"strength": 0.9}),
            ("missing modality token", {"modality_tokens": {"t1n": 29, "t2w": 30, "t2f": 31}}),
            ("string modality token", {"modality_tokens": {"t1n": 29, "t2w": 30, "t2f": 31, "t1c": "34"}}),
            ("grid not three dims", {"grid": [256, 256]}),
            ("grid not positive ints", {"grid": [256, 256, -128]}),
            ("empty seed rule", {"seed_rule": "  "}),
        ):
            self.expect_reject(lambda overrides=overrides: P3CandidateInferenceConfig(self._config_payload(**overrides)), label)

    def _check_plan_coverage(self):
        config = P3CandidateInferenceConfig(self._config_payload())
        builder = P3CandidateSamplePlanBuilder("p3-candidate-fixture", "d" * 64, "c" * 64, "holdout", config)
        generated_root = self._workdir / "generated"

        def real_of(challenge, case, modality):
            return Path("/ctrl/raw") / challenge / case / f"{case}-{modality}.nii.gz"

        entries = builder.entries(self.CASES, real_of, generated_root)
        if len(entries) != len(self.CASES):
            self.failures.append(f"expected one entry per case, got {len(entries)}")
        for entry in entries:
            if entry["phase"] != "P3" or entry["variant"] != CANDIDATE_VARIANT:
                self.failures.append("every entry must carry the explicit P3 controlnet-candidate mislabel guards")
            if not isinstance(entry["candidate_checkpoint_sha256"], str) or len(entry["candidate_checkpoint_sha256"]) != 64:
                self.failures.append("entry must carry the candidate ControlNet checkpoint sha256")
            if not isinstance(entry["dm_checkpoint_sha256"], str) or len(entry["dm_checkpoint_sha256"]) != 64:
                self.failures.append("entry must carry the pinned DM checkpoint sha256")
            covered = set()
            for anchor, info in entry["anchors"].items():
                generated = info["generated"]
                if set(generated) != {m for m in MODALITIES if m != anchor}:
                    self.failures.append(f"anchor {anchor} must generate exactly the other three modalities")
                for tgt, sample in generated.items():
                    covered.add((anchor, tgt))
                    expected = (
                        generated_root
                        / entry["challenge"]
                        / entry["case_id"]
                        / f"a{anchor}"
                        / (f"{tgt}_seed{config.seed_of(entry['case_id'], anchor, tgt)}.nii.gz")
                    )
                    if sample["path"] != str(expected) or sample["seed"] != config.seed_of(entry["case_id"], anchor, tgt):
                        self.failures.append(f"generated sample for {anchor}->{tgt} must use the deterministic seeded path")
            if covered != set(builder.ordered_pairs(entry["case_id"])):
                self.failures.append("the four anchor rounds must cover all 12 ordered modality pairs per case")

    def _check_run_guard(self):
        root = self._workdir / "run-guard"
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = root / "p3-controlnet.pt"
        checkpoint.write_bytes(b"frozen-p3-controlnet-fixture")
        dm = root / "p1-dm.pt"
        dm.write_bytes(b"p1-dm-fixture")
        infer_path = root / "infer.json"
        infer_path.write_text(json.dumps(self._config_payload()))
        cn_sha = P3CandidateRunGuard.file_sha256(checkpoint)
        dm_sha = P3CandidateRunGuard.file_sha256(dm)
        infer_sha = P3CandidateRunGuard.file_sha256(infer_path)

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

        if P3CandidateRunGuard(record(), infer_path).check() != checkpoint:
            self.failures.append("guard positive path must return the pinned P3 ControlNet checkpoint")
        for label, mutated, infer_override in (
            ("stage0 baseline run", record(variant="stage0-baseline"), infer_path),
            ("open (unfrozen) run", record(status="open"), infer_path),
            ("selection carries no candidate checkpoint", record(selection={}), infer_path),
            ("selection == upstream DM (stage-0 in disguise)", record(selection={"checkpoint": {"path": str(dm), "sha256": dm_sha}}), infer_path),
            ("selection pinning the upstream DM path but wrong sha", record(selection={"checkpoint": {"path": str(dm), "sha256": "0" * 64}}), infer_path),
            ("run pinning no inference config", record(configs=[]), infer_path),
            (
                "run pinning two inference configs",
                record(
                    configs=[
                        {"role": "inference", "path": str(infer_path), "sha256": infer_sha},
                        {"role": "inference", "path": str(root / "other.json"), "sha256": "0" * 64},
                    ]
                ),
                infer_path,
            ),
        ):
            self.expect_reject(lambda mutated=mutated, infer_override=infer_override: P3CandidateRunGuard(mutated, infer_override).check(), label)
        drifted = root / "drifted.json"
        drifted.write_text(json.dumps(self._config_payload(cfg_guidance_scale=10.0)))
        self.expect_reject(lambda: P3CandidateRunGuard(record(), drifted).check(), "infer config drifting from the pinned CFG=0 provenance")

    def _check_pairs_merge(self):
        """The merged brats-l1-pairs/1 triplets carry reference+baseline+candidate per direction."""
        config = P3CandidateInferenceConfig(self._config_payload())
        builder = P3CandidateSamplePlanBuilder("p3-candidate-fixture", "d" * 64, "c" * 64, "holdout", config)
        generated_root = self._workdir / "generated"
        stage0_records = []
        for item in self.CASES:
            for src in MODALITIES:
                for tgt in MODALITIES:
                    if src == tgt:
                        continue
                    stage0_records.append(
                        {
                            "challenge": item["sub"],
                            "case": item["case"],
                            "src_modality": src,
                            "target_modality": tgt,
                            "baseline": f"/stage0/{item['sub']}/{item['case']}/a{src}/{tgt}.nii.gz",
                            "reference": f"/refgrid/{item['sub']}/{item['case']}/{tgt}.nii.gz",
                        }
                    )
        doc = builder.pairs(stage0_records, generated_root)
        if doc.get("schema") != PAIRS_SCHEMA or doc.get("variant") != CANDIDATE_VARIANT:
            self.failures.append("pairs document must declare its schema and the controlnet-candidate variant")
        if len(doc["records"]) != 12 * len(self.CASES):
            self.failures.append(f"pairs must hold 12 ordered pairs per case, got {len(doc['records'])}")
        pair_keys = {(r["challenge"], r["case"], r["src_modality"], r["target_modality"]) for r in doc["records"]}
        if len(pair_keys) != len(doc["records"]):
            self.failures.append("pair records must be unique per (case, src, tgt)")
        for record in doc["records"]:
            if not {"reference", "baseline", "candidate"} <= set(record):
                self.failures.append("every brats-l1-pairs/1 record must carry reference+baseline+candidate")
            if record["baseline"].startswith("/stage0") and record["candidate"].startswith(f"{generated_root}/"):
                continue
            self.failures.append("candidate path must live under the candidate generated root")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["selftest"])
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args(argv)
    failures = P3CandidatePlanSelfTest(args.workdir).run()
    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        return 1
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
