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

"""Dev selection-point monitor sampling arm (issue #253, parent #247).

Produces ONLY the sampling products of the dev ET/WT monitor:

1. the stratified dev sample (``DevMonitorCohort``: the pinned monitoring
   quotas filled in ``DevCohortBuilder``'s sha256 order -- deterministic,
   dev-population-only; the holdout 530 is never an input of this module);
2. the four pseudo-quad modalities per case with the candidate checkpoint
   under the frozen sidecar recipe (composition over ``CandidateSampler`` --
   model loading, the denoising loop, the VAE decode, the int16 x1000
   convention and the per-(case, modality) seed rule all reused verbatim, as
   ``TokenSwapSampler`` did for job D);
3. the two-sided instrument plan (``DevMonitorPlanBuilder``):
   schema-compatible with the terminal acceptance's
   ``l2-final-acceptance-plan/1`` so the frozen-instrument execution side
   (``final_acceptance predict`` / ``measurement_run assemble-execute`` /
   ``measure``) runs verbatim, read-only. The generated side carries the
   written sample volumes; the real side resolves each case's four modalities
   from the dev list against the raw root (native BraTS pass-through, the
   same treatment the terminal acceptance gives its real side).

Zero holdout contact: the dev list is the only population input, the plan
records ``population: "dev"``, and a cohort shortfall fails loudly (a silent
shrink would change the observation line's resolution). Everything downstream
is statistics and lives in ``ctmr.application.acceptance.distribution``
(``dev_monitor``, ``variant=diagnostic``, never an acceptance verdict); the
sugon host recipe that chains sample -> instrument -> report is
``deploy/jobs/run_dev_monitor_etwt.sh``.

Usage (sugon, one DCU; VAE path comes from the env json's
trained_autoencoder_path, resolvable from the working directory):
    python -m ctmr.application.generation.modality_label.dev_monitor_sampling \
        --dev-list ... --raw-root ... --emb-root ... --ckpt <candidate .pt> \
        -e env.json -c model.json -t network.json \
        --samples-dir DIR --output-dir DIR [--plan-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import nibabel as nib
import torch

from ctmr.application.acceptance.distribution.diagnostic_support import DiagnosticError
from ctmr.application.acceptance.distribution.measurement_table import CHANNEL_SUFFIXES
from ctmr.application.generation.modality_label.monitor import CandidateSampler, CohortSpacingSource
from ctmr.application.shell import MODALITY_TOKENS
from ctmr.domain.dm_output_grid import V1_DM_OUTPUT_GRID
from ctmr.wiring.generate import modality_label_engine

# The frozen monitoring protocol (issue #253): per-challenge dev sample quotas.
# Roughly proportional to the dev composition with the small challenges
# over-weighted (the 16-case sidecar precedent) so each per-challenge median
# has enough support and the METS rate rule keeps ~4% granularity (24 cases:
# 21/24 = 0.875 fires, 22/24 = 0.917 clears). 130 cases x 4 modalities = 520
# samples per candidate measurement point. NOT knobs: the T8 comparison reads
# this baseline, so the table is pinned here and registered in the experiment
# record; changing it is a new protocol decision, never a CLI flag.
MONITOR_QUOTAS = {"GLI": 50, "MEN": 40, "METS": 24, "PED": 10, "SSA": 6}

# The frozen sidecar sampling recipe (cfg=10, 30 RF steps) -- the values the
# candidate's dev evidence was generated under.
MONITOR_CFG = 10.0
MONITOR_STEPS = 30

# The modality word per dev-list modality spelling (trend.py's mapping).
DEV_MODALITY_TO_WORD = {
    "mri_t1_skull_stripped": "t1n",
    "mri_t1c_skull_stripped": "t1c",
    "mri_t2_skull_stripped": "t2w",
    "mri_flair_skull_stripped": "t2f",
}


class DevMonitorCohort:
    """The monitor's stratified dev sample: pinned quotas in sha256 order.

    ``DevCohortBuilder``'s selection rule generalized to the monitoring quota
    table: within each challenge the cases are ordered by
    ``sha256(f"{challenge}/{case}")`` and the first quota-many enter the
    sample. A dev list that cannot fill a quota raises -- the flag rule's
    resolution is protocol, not whatever the list happens to have that day.
    """

    def __init__(self, dev_list_path, quotas=None):
        self._path = Path(dev_list_path)
        self._quotas = dict(quotas) if quotas is not None else dict(MONITOR_QUOTAS)

    def build(self):
        cases_by_challenge = {}
        for entry in json.loads(self._path.read_text())["training"]:
            cases_by_challenge.setdefault(entry["sub"], set()).add(entry["case"])
        cohort = []
        for challenge in sorted(self._quotas):
            ordered = sorted(
                cases_by_challenge.get(challenge, set()),
                key=lambda case: hashlib.sha256(f"{challenge}/{case}".encode()).hexdigest(),
            )
            selected = ordered[: self._quotas[challenge]]
            if len(selected) < self._quotas[challenge]:
                raise DiagnosticError(f"dev list has {len(selected)} {challenge} cases but the monitoring quota needs {self._quotas[challenge]}")
            for case in selected:
                cohort.append({"sub": challenge, "case": case})
        return cohort

    def write(self, out_path):
        cohort = self.build()
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "population": "dev",
                    "quotas": self._quotas,
                    "selection_rule": "sha256(challenge/case) order, first quota-many per challenge",
                    "cohort": cohort,
                },
                indent=1,
            )
            + "\n"
        )
        return cohort


class DevMonitorPlanBuilder:
    """Turns the cohort + written samples + dev-list real paths into the instrument plan.

    The plan is the terminal acceptance's ``l2-final-acceptance-plan/1`` shape
    (the execution side's schema gate) with the monitoring provenance rides
    (``population: "dev"``, the quota table, the sampling rule) -- extra keys
    the execution side ignores. Observations per case: one four-channel gen
    observation over the written sample volumes and one real observation over
    the dev list's native BraTS files.
    """

    def __init__(self, dev_list_path, raw_root, samples_dir, quotas=None, run_id=None):
        self._dev_list_path = Path(dev_list_path)
        self._raw_root = Path(raw_root)
        self._samples_dir = Path(samples_dir)
        self._quotas = dict(quotas) if quotas is not None else dict(MONITOR_QUOTAS)
        self._run_id = run_id
        self._real_images = self._index_dev_list()

    def _index_dev_list(self):
        images: dict[tuple, dict] = {}
        for entry in json.loads(self._dev_list_path.read_text())["training"]:
            images.setdefault((entry["sub"], entry["case"]), {})[entry["modality"]] = str(self._raw_root / entry["image"])
        return images

    def real_channels(self, challenge, case):
        """The four native real paths of one dev case, keyed by instrument channel.

        Every path must exist: the real side is a pass-through copy in
        assemble-execute, so a missing file is a loud preflight failure, not a
        mid-chain crash.
        """
        by_modality = self._real_images.get((challenge, case), {})
        words = {DEV_MODALITY_TO_WORD[modality]: path for modality, path in by_modality.items()}
        missing = [word for word in DEV_MODALITY_TO_WORD.values() if word not in words]
        if missing:
            raise DiagnosticError(f"dev list real paths missing modalities {missing} for ({challenge}, {case})")
        channels = {CHANNEL_SUFFIXES[word]: words[word] for word in sorted(words)}
        absent = [path for path in channels.values() if not Path(path).is_file()]
        if absent:
            raise DiagnosticError(f"real reference images not found under {self._raw_root}: {absent}")
        return channels

    def gen_channels(self, case):
        """The four written sample volumes of one case (missing file -> loud failure)."""
        channels = {}
        for word in sorted(DEV_MODALITY_TO_WORD.values()):
            seed = CandidateSampler.seed_of(case, word)
            path = self._samples_dir / f"{case}_{word}_seed{seed}.nii.gz"
            if not path.is_file():
                raise DiagnosticError(f"sample volume missing: {path} (sample the cohort first)")
            channels[CHANNEL_SUFFIXES[word]] = str(path)
        return channels

    def build(self, cohort):
        observations = []
        case_counts: dict = {}
        for item in cohort:
            challenge, case = item["sub"], item["case"]
            observations.append(
                {
                    "obs_id": f"{case}__real",
                    "challenge": challenge,
                    "case": case,
                    "side": "real",
                    "anchor": None,
                    "channels": self.real_channels(challenge, case),
                    "condition_mask": None,
                }
            )
            observations.append(
                {
                    "obs_id": f"{case}__gen",
                    "challenge": challenge,
                    "case": case,
                    "side": "gen",
                    "anchor": None,
                    "channels": self.gen_channels(case),
                    "condition_mask": None,
                }
            )
            case_counts.setdefault(challenge, set()).add(case)
        obs_ids = [obs["obs_id"] for obs in observations]
        if len(set(obs_ids)) != len(obs_ids):
            raise DiagnosticError("duplicate obs_id in the monitor plan")
        return {
            "schema": "l2-final-acceptance-plan/1",
            "phase": "P1",
            "population": "dev",
            "run_id": self._run_id,
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "monitor_quotas": self._quotas,
            "sampling_rule": f"sha256(case|modality) per-modality seed, cfg={MONITOR_CFG:g}, {MONITOR_STEPS} steps (RFlowScheduler)",
            "challenges": {
                challenge: {
                    "n_cases": len(cases),
                    "quota": self._quotas[challenge],
                    "provisional": len(cases) < self._quotas[challenge],
                }
                for challenge, cases in sorted(case_counts.items())
            },
            "observations": observations,
        }


class DevMonitorSampler:
    """Generates the four-modal monitoring sample with the sidecar's sampler.

    Composition, not re-decision: the model loading, denoising loop, VAE
    decode and int16 x1000 convention all come from ``CandidateSampler``; this
    arm adds only the cohort loop and the monitor filename family. Existing
    files are skipped, so the arm is re-entrant.
    """

    def __init__(self, args, device, engine):
        self._sampler = CandidateSampler(args, device, None, engine)
        self._device = device

    def sample_cohort(self, checkpoint_path, cohort, spacings, out_dir) -> int:
        model, recon = self._sampler.load_models(checkpoint_path)
        written = 0
        for item in cohort:
            case = item["case"]
            spacing = spacings.spacing_of(case)
            for modality in sorted(DEV_MODALITY_TO_WORD.values()):
                seed = CandidateSampler.seed_of(case, modality)
                out = Path(out_dir) / f"{case}_{modality}_seed{seed}.nii.gz"
                if out.is_file():
                    continue
                data = self._sampler.sample_one(model, recon, MODALITY_TOKENS[modality], spacing, seed)
                out.parent.mkdir(parents=True, exist_ok=True)
                # Ruling #6: declare the v1 DM's real sampling spacing -- the
                # same post-#249 write protocol as the sidecar.
                nib.save(nib.Nifti1Image(data, affine=V1_DM_OUTPUT_GRID.affine()), out)
                written += 1
                print(f"[sample] {case} {modality} (seed {seed}) -> {out.name}", flush=True)
        return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev-list", required=True, help="the P1 dev list json (the ONLY population input -- never the holdout)")
    parser.add_argument("--raw-root", default=None, help="raw BraTS root for the real-side pass-through paths (dev-list relative; plan build only)")
    parser.add_argument("--emb-root", default=None, help="embedding companion root for per-case spacings (t1n entry; sampling only)")
    parser.add_argument("--ckpt", default=None, help="the frozen candidate checkpoint, loaded read-only")
    parser.add_argument("-e", "--env_config_path", default=None)
    parser.add_argument("-c", "--model_config_path", default=None)
    parser.add_argument("-t", "--model_def_path", default=None)
    parser.add_argument("--samples-dir", required=True, help="artifact directory for the per-case volumes (never git)")
    parser.add_argument("--output-dir", required=True, help="monitor work root for cohort.json + plan.json")
    parser.add_argument("--plan-only", action="store_true", help="skip sampling: (re)build cohort + plan from an existing samples dir")
    parser.add_argument(
        "--sampling-only", action="store_true", help="skip the plan: sample the cohort + write cohort.json only (the plan needs the real-side root)"
    )
    parser.add_argument("--run-id", default=None, help="the candidate's run id, recorded into the plan")
    args = parser.parse_args(argv)
    if args.plan_only and args.sampling_only:
        parser.error("--plan-only and --sampling-only are mutually exclusive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = Path(args.samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    cohort = DevMonitorCohort(args.dev_list).write(output_dir / "cohort.json")
    if not args.plan_only:
        for required, flag_name in (
            (args.ckpt, "--ckpt"),
            (args.env_config_path, "-e"),
            (args.model_config_path, "-c"),
            (args.model_def_path, "-t"),
            (args.emb_root, "--emb-root"),
        ):
            if not required:
                parser.error(f"{flag_name} is required for sampling (omit only with --plan-only)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[dev-monitor] device={device}; variant=diagnostic -- dev 选择面采样,零 holdout 接触,checkpoint 只读", flush=True)
        engine = modality_label_engine()
        merged = engine.load_config(args.env_config_path, args.model_config_path, args.model_def_path)
        merged.cfg_guidance_scale = MONITOR_CFG
        merged.diffusion_unet_inference = (
            merged.diffusion_unet_inference if hasattr(merged, "diffusion_unet_inference") else {"num_inference_steps": MONITOR_STEPS}
        )
        spacings = CohortSpacingSource(args.dev_list, args.emb_root)
        written = DevMonitorSampler(merged, device, engine).sample_cohort(args.ckpt, cohort, spacings, samples_dir)
        print(f"[dev-monitor] {len(cohort)} cases x 4 modalities; written {written} new volumes -> {samples_dir}", flush=True)

    if args.sampling_only:
        print("[dev-monitor] --sampling-only: plan deferred until the real-side root is available", flush=True)
        return 0
    if not args.raw_root:
        parser.error("--raw-root is required for the plan build (omit only with --sampling-only)")

    plan = DevMonitorPlanBuilder(args.dev_list, args.raw_root, samples_dir, run_id=args.run_id).build(cohort)
    plan_path = output_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"[dev-monitor] plan -> {plan_path} ({len(plan['observations'])} observations, population={plan['population']})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
