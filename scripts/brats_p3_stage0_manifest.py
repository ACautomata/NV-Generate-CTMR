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

"""Stage-0 (zero-training img2img) sample-plan assembly for the P3 baseline (issue #60).

Pure manifest logic — stdlib only, no torch/nibabel — so the plan layer is
testable off the DCU box. The GPU driver (``scripts.brats_p3_stage0_generate``)
imports this module and feeds it resolved paths.

Outputs two controlled manifests per run side:

- ``samples.json``: a top-level list whose entries match the L2 final-acceptance
  ``P3FourAnchorPlan`` layout (``phase=P3`` + per-anchor ``real``/``generated``
  channels), so ``nnunet_l2_final_acceptance assemble --phase P3`` consumes it
  directly. Each entry additionally carries the explicit ``variant`` /
  ``run_id`` / ``dm_checkpoint_sha256`` markers so a baseline manifest can never
  be mistaken for a trained-candidate manifest.
- ``pairs.json``: the L1-side flat view — one record per ordered (src -> tgt)
  pair with the stage-0 ``baseline`` volume and the real ``reference`` volume.
  The P3 candidate's final acceptance merges these with its own candidate
  volumes into ``brats-l1-pairs/1`` records (issue #54).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MODALITIES = ("t1n", "t1c", "t2w", "t2f")
INFER_SCHEMA = "brats-p3-stage0-infer/1"
PAIRS_SCHEMA = "brats-p3-stage0-pairs/1"
STAGE0_VARIANT = "stage0-baseline"
SEED_MODULUS = 2**31 - 1


class Stage0PlanError(Exception):
    """Raised when the inference config or the assembled plan breaks the stage-0 contract."""


class Stage0InferenceConfig:
    """The official stage-0 inference configuration (configs/config_p3_stage0_infer.json)."""

    def __init__(self, payload):
        if payload.get("schema") != INFER_SCHEMA:
            raise Stage0PlanError(f"stage-0 inference config schema must be {INFER_SCHEMA!r}")
        self.scheduler = self._text(payload, "scheduler")
        if self.scheduler != "RFlowScheduler":
            raise Stage0PlanError("stage-0 img2img only runs on the RFlow scheduler (rectified flow interpolation start)")
        self.num_inference_steps = self._positive_int(payload, "num_inference_steps")
        self.cfg_guidance_scale = self._number(payload, "cfg_guidance_scale")
        if self.cfg_guidance_scale < 0:
            raise Stage0PlanError("cfg_guidance_scale must be >= 0")
        self.strength = self._number(payload, "strength")
        if not 0.0 < self.strength < 1.0:
            raise Stage0PlanError("strength must lie strictly in (0, 1); 1 would erase the src latent, 0 would copy it")
        self.grid = self._grid(payload)
        self.modality_tokens = payload.get("modality_tokens")
        if not isinstance(self.modality_tokens, dict) or set(self.modality_tokens) != set(MODALITIES):
            raise Stage0PlanError(f"modality_tokens must cover exactly {list(MODALITIES)} (P1-DM tokens)")
        if not all(isinstance(token, int) and token > 0 for token in self.modality_tokens.values()):
            raise Stage0PlanError("modality tokens must be positive integers (P1-DM class labels)")
        self.seed_rule = self._text(payload, "seed_rule")

    @classmethod
    def from_path(cls, path):
        return cls(json.loads(Path(path).read_text()))

    @staticmethod
    def _text(payload, key):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise Stage0PlanError(f"stage-0 inference config needs a non-empty {key}")
        return value

    @staticmethod
    def _positive_int(payload, key):
        value = payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise Stage0PlanError(f"stage-0 inference config {key} must be a positive integer")
        return value

    @staticmethod
    def _number(payload, key):
        value = payload.get(key)
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise Stage0PlanError(f"stage-0 inference config {key} must be numeric")
        return float(value)

    @staticmethod
    def _grid(payload):
        value = payload.get("grid")
        if not isinstance(value, list | tuple) or len(value) != 3 or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in value):
            raise Stage0PlanError("stage-0 inference config grid must be three positive integers (SRI24 256x256x128)")
        return tuple(value)

    @staticmethod
    def seed_of(case, src, tgt):
        """Deterministic per (case, direction) seed: sha256('<case>|<src>-><tgt>') truncated."""
        digest = hashlib.sha256(f"{case}|{src}->{tgt}".encode()).hexdigest()[:8]
        return int(digest, 16) % SEED_MODULUS


class Stage0RunGuard:
    """Checks the frozen stage-0 run record and pins the DM checkpoint + inference config.

    The generation-time recheck of the phase-run contract markers (issue #60
    acceptance criterion 1): the run must be the zero-training P3 variant, its
    selection must pin the upstream P1-DM checkpoint, and the inference config
    used on the command line must byte-match the ``role=inference`` config the
    run pinned at init — so every generated volume traces back to the recorded
    official inference provenance.
    """

    def __init__(self, run_record, infer_config_path):
        self._record = run_record
        self._infer_config_path = Path(infer_config_path)

    def check(self):
        record = self._record
        if record.get("phase") != "P3" or record.get("variant") != STAGE0_VARIANT:
            raise Stage0PlanError(
                f"run {record.get('run_id')} is not a P3 {STAGE0_VARIANT} run; "
                "stage-0 generation must hang off its own zero-training baseline record"
            )
        if record.get("status") != "frozen":
            raise Stage0PlanError(f"run {record['run_id']} is {record.get('status')}; generate samples only after the freeze")
        selection = record.get("selection") or {}
        upstream = record.get("upstream") or {}
        if selection.get("checkpoint", {}).get("sha256") != upstream.get("checkpoint", {}).get("sha256"):
            raise Stage0PlanError("stage-0 selection does not pin the upstream P1-DM checkpoint; the record is inconsistent")
        checkpoint = Path(upstream["checkpoint"]["path"])
        if not checkpoint.is_file():
            raise Stage0PlanError(f"pinned P1-DM checkpoint missing: {checkpoint}")
        if self.file_sha256(checkpoint) != upstream["checkpoint"]["sha256"]:
            raise Stage0PlanError(f"pinned P1-DM checkpoint changed on disk: {checkpoint}")
        inference_entries = [entry for entry in record.get("configs", []) if entry.get("role") == "inference"]
        if len(inference_entries) != 1:
            raise Stage0PlanError(
                "the stage-0 run must pin exactly one --config inference=<official stage-0 inference config> "
                "(the recorded inference provenance, issue #60 acceptance criterion 1)"
            )
        if self.file_sha256(self._infer_config_path) != inference_entries[0]["sha256"]:
            raise Stage0PlanError(
                f"--infer-config sha256 does not match the config pinned by run {record['run_id']}; "
                "regenerate with the recorded official inference config"
            )
        return checkpoint

    @staticmethod
    def file_sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()


class Stage0SamplePlanBuilder:
    """Builds the L2-compatible four-anchor entries and the L1-side pair records.

    One real anchor per modality per case; each anchor round conditions the
    other three modalities (12 ordered src->tgt pairs per case, spec #51
    decision 8 / CONTEXT.md 仪器使用协议). Every carried path marker keeps the
    ``stage0-baseline`` variant explicit so the manifests cannot masquerade as
    a trained candidate's evidence.
    """

    def __init__(self, run_id, dm_checkpoint_sha256, side, config):
        self._run_id = run_id
        self._dm_sha = dm_checkpoint_sha256
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
                    "variant": STAGE0_VARIANT,
                    "run_id": self._run_id,
                    "side": self._side,
                    "dm_checkpoint_sha256": self._dm_sha,
                    "anchors": anchors,
                }
            )
        return entries

    def pairs(self, cohort, real_of, generated_root):
        """The flat L1-side records: stage-0 baseline + real reference per ordered pair."""
        records = []
        for item in cohort:
            challenge, case = item["sub"], item["case"]
            for src in MODALITIES:
                for tgt in MODALITIES:
                    if src == tgt:
                        continue
                    records.append(
                        {
                            "challenge": challenge,
                            "case": case,
                            "src_modality": src,
                            "target_modality": tgt,
                            "seed": self._config.seed_of(case, src, tgt),
                            "baseline": str(self.generated_path(generated_root, challenge, case, src, tgt)),
                            "reference": str(real_of(challenge, case, tgt)),
                        }
                    )
        return {
            "schema": PAIRS_SCHEMA,
            "run_id": self._run_id,
            "variant": STAGE0_VARIANT,
            "side": self._side,
            "dm_checkpoint_sha256": self._dm_sha,
            "records": records,
        }

    def ordered_pairs(self, case):
        """The 12 ordered src!=tgt modality pairs this plan must cover per case."""
        return [(src, tgt) for src in MODALITIES for tgt in MODALITIES if src != tgt]


class Stage0PlanSelfTest:
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
        except Stage0PlanError:
            return
        self.failures.append(f"expected rejection but succeeded: {label}")

    def _config_payload(self, **overrides):
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

    def run(self):
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._check_config_validation()
        self._check_plan_coverage()
        self._check_run_guard()
        return self.failures

    def _check_config_validation(self):
        Stage0InferenceConfig(self._config_payload())  # positive path
        for label, overrides in (
            ("wrong schema", {"schema": "brats-p3-stage0-infer/2"}),
            ("ddpm scheduler", {"scheduler": "DDPMScheduler"}),
            ("zero steps", {"num_inference_steps": 0}),
            ("negative cfg", {"cfg_guidance_scale": -1.0}),
            ("strength 1.0 erases src", {"strength": 1.0}),
            ("strength 0 copies src", {"strength": 0.0}),
            ("missing modality token", {"modality_tokens": {"t1n": 29, "t2w": 30, "t2f": 31}}),
            ("string modality token", {"modality_tokens": {"t1n": 29, "t2w": 30, "t2f": 31, "t1c": "34"}}),
            ("grid not three dims", {"grid": [256, 256]}),
            ("grid not positive ints", {"grid": [256, 256, -128]}),
            ("empty seed rule", {"seed_rule": "  "}),
        ):
            self.expect_reject(lambda overrides=overrides: Stage0InferenceConfig(self._config_payload(**overrides)), label)

    def _check_plan_coverage(self):
        config = Stage0InferenceConfig(self._config_payload())
        builder = Stage0SamplePlanBuilder("p3-stage0-fixture", "d" * 64, "holdout", config)
        generated_root = self._workdir / "generated"

        def real_of(challenge, case, modality):
            return Path("/ctrl/raw") / challenge / case / f"{case}-{modality}.nii.gz"

        entries = builder.entries(self.CASES, real_of, generated_root)
        pairs_doc = builder.pairs(self.CASES, real_of, generated_root)
        if len(entries) != len(self.CASES):
            self.failures.append(f"expected one entry per case, got {len(entries)}")
        for entry in entries:
            if entry["phase"] != "P3" or entry["variant"] != STAGE0_VARIANT:
                self.failures.append("every entry must carry the explicit P3 stage0-baseline mislabel guards")
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
        records = pairs_doc["records"]
        if pairs_doc.get("schema") != PAIRS_SCHEMA or pairs_doc.get("variant") != STAGE0_VARIANT:
            self.failures.append("pairs document must declare its schema and the stage0-baseline variant")
        if len(records) != 12 * len(self.CASES):
            self.failures.append(f"pairs must hold 12 ordered pairs per case, got {len(records)}")
        pair_keys = {(record["challenge"], record["case"], record["src_modality"], record["target_modality"]) for record in records}
        if len(pair_keys) != len(records):
            self.failures.append("pair records must be unique per (case, src, tgt)")
        baseline_paths = {record["baseline"] for record in records}
        anchor_paths = {sample["path"] for entry in entries for info in entry["anchors"].values() for sample in info["generated"].values()}
        if baseline_paths != anchor_paths:
            self.failures.append("pairs and anchors must reference the identical generated volume set")
        for record in records:
            if record["reference"].endswith(f"-{record['target_modality']}.nii.gz") is False:
                self.failures.append("pair reference must point at the real target-modality volume")
        if config.seed_of("FIXGLI-0200-000", "t1n", "t1c") == config.seed_of("FIXGLI-0200-000", "t1c", "t1n"):
            self.failures.append("seeds must be direction-sensitive (src->tgt differs from tgt->src)")

    def _check_run_guard(self):
        root = self._workdir / "run-guard"
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = root / "p1-dm.pt"
        checkpoint.write_bytes(b"frozen-p1-dm-fixture")
        infer_path = root / "infer.json"
        infer_path.write_text(json.dumps(self._config_payload()))
        dm_sha = Stage0RunGuard.file_sha256(checkpoint)
        infer_sha = Stage0RunGuard.file_sha256(infer_path)

        def record(**overrides):
            payload = {
                "run_id": "p3-stage0-fixture",
                "phase": "P3",
                "variant": STAGE0_VARIANT,
                "status": "frozen",
                "selection": {"checkpoint": {"path": str(checkpoint), "sha256": dm_sha}},
                "upstream": {"checkpoint": {"path": str(checkpoint), "sha256": dm_sha}},
                "configs": [{"role": "inference", "path": str(infer_path), "sha256": infer_sha}],
            }
            payload.update(overrides)
            return payload

        if Stage0RunGuard(record(), infer_path).check() != checkpoint:
            self.failures.append("guard positive path must return the pinned P1-DM checkpoint")
        for label, mutated, infer_override in (
            ("controlnet candidate run", record(variant="controlnet-candidate"), infer_path),
            ("open (unfrozen) run", record(status="open"), infer_path),
            ("selection not pinning the upstream DM", record(selection={"checkpoint": {"path": str(checkpoint), "sha256": "0" * 64}}), infer_path),
            ("pinned checkpoint changed on disk", record(upstream={"checkpoint": {"path": str(checkpoint), "sha256": "0" * 64}}), infer_path),
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
            self.expect_reject(lambda mutated=mutated, infer_override=infer_override: Stage0RunGuard(mutated, infer_override).check(), label)
        drifted = root / "drifted.json"
        drifted.write_text(json.dumps(self._config_payload(cfg_guidance_scale=0.0)))
        self.expect_reject(lambda: Stage0RunGuard(record(), drifted).check(), "infer config drifting from the pinned provenance")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["selftest"])
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args(argv)
    failures = Stage0PlanSelfTest(args.workdir).run()
    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        return 1
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
