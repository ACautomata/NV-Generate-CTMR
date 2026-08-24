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

"""P2 final-holdout mask-conditioned sample generation (issue #59).

Generates the four target-modality samples for final-holdout cases with the
frozen P2 ControlNet candidate hung off the frozen P1-DM, using exactly the
dev sidecar sampling convention (``P2CandidateSampler``: RFlowScheduler,
cfg=10, 30 steps, fp16, autoencoder decode, ``x1000`` int16 MR scale, seed =
sha256(case|modality)). The condition is the case's ``-combined.nii.gz``
(brain=22 union + 1/2/3 -> 129/130/131 tumour remap) built by ``brats_phase_prep
labels --sides holdout``. The per-case post-resize spacing is replicated from
the raw t1n header with the issue #52 companion formula (``spacing_i =
pixdim_i * shape_i / GRID_i``, GRID = 256x256x128), because holdout cases
carry no embedding companion. The driver writes the L2 assembly-samples
manifest (``phase/challenge/case_id/condition_mask/samples{path,seed}/
real_paths``) over the holdout side.

Sharding: ``--shard i --num-shards n`` takes every n-th case of the
deterministic cohort order; each shard writes ``samples_shard_<i>.json`` and
shares the idempotent ``generated/`` tree, so shards run concurrently on
separate GPUs and their manifests concatenate into the final samples.json.

Usage::

    python -m scripts.brats_p2_holdout_generate \
        --run runs/p2-xxx/run.json \
        --manifest /ctrl/phase/phase_manifest.json \
        --out-root /ctrl/p2/holdout_generated \
        --raw-root /ctrl/phase/raw \
        --label-root /ctrl/phase \
        -e environment_brats_p2_train.json \
        -c configs/config_brats_p2_train.json \
        -t configs/config_network_rflow.json \
        [--shard 0 --num-shards 8] [--limit N] [--challenge GLI] [--only-cases ...]

The merged ``samples.json`` is structured for ``nnunet_l2_final_acceptance
assemble --phase P2 --samples`` (one condition mask, four modalities per case).
"""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .brats_p1_dev_eval import MODALITY_TOKENS, TARGET_MODALITIES
from .brats_p2_dev_eval import P2CandidateSampler
from .diff_model_setting import load_config

GRID = (256, 256, 128)
IDENTITY_AFFINE = np.diag([1.0, 1.0, 1.0, 1.0])


class HoldoutSpacingSource:
    """Replicates the #52 post-resize spacing from the raw t1n header (no embedding companion)."""

    def __init__(self, raw_root, manifest):
        self._raw_root = Path(raw_root)
        self._dirs = {}
        for challenge, info in manifest["challenges"].items():
            source_dir = info.get("source_dir", "")
            rel = Path(source_dir).relative_to(self._raw_root) if source_dir.startswith(str(self._raw_root)) else Path(challenge)
            for case in info["cases"]["holdout"]:
                self._dirs[case] = rel / case

    def directory_of(self, case):
        """The raw-relative case directory (holds t1n/t1c/t2w/t2f/seg)."""
        return self._dirs[case]

    def spacing_of(self, case):
        path = self._raw_root / self._dirs[case] / f"{case}-t1n.nii.gz"
        if not path.is_file():
            raise FileNotFoundError(f"holdout t1n missing: {path}")
        image = nib.load(path)
        zoom = [float(z) for z in image.header.get_zooms()[:3]]
        shape = [float(s) for s in image.shape[:3]]
        return [zoom[i] * shape[i] / GRID[i] for i in range(3)]


class HoldoutMaskSource:
    """Per-case P2 condition mask (``-combined.nii.gz``) under the phase label root."""

    def __init__(self, label_root, manifest):
        self._label_root = Path(label_root)
        self._challenge_of = {
            case: challenge
            for challenge, info in manifest["challenges"].items()
            for case in info["cases"]["holdout"]
        }

    def path_of(self, case):
        challenge = self._challenge_of[case]
        return self._label_root / "labels" / challenge / case / f"{case}-combined.nii.gz"


class HoldoutCohortBuilder:
    """The holdout cohort entries (sub, case) from the pinned phase manifest, sharded deterministically."""

    def __init__(self, manifest, shard=0, num_shards=1, limit=None, only_challenge=None, only_cases=None):
        self._manifest = manifest
        self._shard = shard
        self._num_shards = num_shards
        self._limit = limit
        self._only_challenge = only_challenge
        self._only_cases = set(only_cases or [])

    def build(self):
        cohort = []
        for challenge, info in self._manifest["challenges"].items():
            if self._only_challenge is not None and challenge != self._only_challenge:
                continue
            cases = [case for case in info["cases"]["holdout"] if not self._only_cases or case in self._only_cases]
            for case in cases[: self._limit] if self._limit is not None else cases:
                cohort.append({"sub": challenge, "case": case})
        if self._num_shards > 1:
            cohort = cohort[self._shard :: self._num_shards]
        return cohort


class P2HoldoutSampleWriter:
    """Runs the P2 candidate sampler over the holdout cohort and writes the L2 samples manifest."""

    def __init__(self, merged, run_record, raw_root, out_root, device, logger):
        self._merged = merged
        self._run_record = run_record
        self._raw_root = Path(raw_root)
        self._out_root = Path(out_root)
        self._device = device
        self._logger = logger

    def write(self, cohort, spacings, masks):
        checkpoint_path = self._run_record["selection"]["checkpoint"]["path"]
        sampler = P2CandidateSampler(self._merged, self._device, self._logger)
        autoencoder, unet, controlnet, scale = sampler.load_models(checkpoint_path)
        self._logger(f"[gen] candidate checkpoint: {checkpoint_path} (epoch {self._run_record['selection']['checkpoint'].get('epoch')})")
        entries = []
        generated_dir = self._out_root / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        try:
            for item in cohort:
                challenge, case = item["sub"], item["case"]
                condition = P2CandidateSampler.load_condition_mask(masks, case, self._device)
                case_dir = generated_dir / challenge / case
                case_dir.mkdir(parents=True, exist_ok=True)
                spacing = spacings.spacing_of(case)
                sample_paths = {}
                for modality in TARGET_MODALITIES:
                    seed = P2CandidateSampler.seed_of(case, modality)
                    out = case_dir / f"{case}_{modality}_seed{seed}.nii.gz"
                    if not out.is_file():
                        data = sampler.sample_one(
                            autoencoder, unet, controlnet, scale,
                            MODALITY_TOKENS[modality], spacing, seed, condition,
                        )
                        self._logger(f"[gen] {challenge}/{case}/{modality} seed={seed}")
                        nib.save(nib.Nifti1Image(data, affine=IDENTITY_AFFINE), out)
                    sample_paths[modality] = {"path": str(out.resolve()), "seed": seed}
                entries.append(
                    {
                        "case_id": case,
                        "challenge": challenge,
                        "phase": "P2",
                        "condition_mask": str(masks.path_of(case).resolve()),
                        "samples": sample_paths,
                        "real_paths": {
                            m: str((self._raw_root / spacings.directory_of(case) / f"{case}-{m}.nii.gz").resolve())
                            for m in TARGET_MODALITIES
                        },
                    }
                )
        finally:
            del autoencoder, unet, controlnet
            torch.cuda.empty_cache()
        return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="P2 brats-phase-run record with a recorded selection")
    parser.add_argument("--manifest", required=True, help="pinned phase phase_manifest.json")
    parser.add_argument("--out-root", required=True, help="controlled output root")
    parser.add_argument("--raw-root", required=True, help="phase raw root (holdout images land here)")
    parser.add_argument("--label-root", required=True, help="phase root holding labels/<CH>/<case>/")
    parser.add_argument("-e", "--env_config_path", required=True)
    parser.add_argument("-c", "--model_config_path", required=True)
    parser.add_argument("-t", "--model_def_path", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="max holdout cases per challenge")
    parser.add_argument("--challenge", default=None, help="restrict to one challenge")
    parser.add_argument("--only-cases", nargs="*", default=None)
    args = parser.parse_args(argv)

    run_record = json.loads(Path(args.run).read_text())
    if not run_record.get("selection"):
        print(f"run {run_record.get('run_id')} has no selection; record the dev-side selection first", file=sys.stderr)
        return 1
    manifest = json.loads(Path(args.manifest).read_text())
    merged = load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    merged.diffusion_unet_inference = (
        merged.diffusion_unet_inference if hasattr(merged, "diffusion_unet_inference") else {"num_inference_steps": 30}
    )
    merged.cfg_guidance_scale = 10.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cohort = HoldoutCohortBuilder(manifest, args.shard, args.num_shards, args.limit, args.challenge, args.only_cases).build()
    if not cohort:
        print("empty cohort after sharding/filters; nothing to generate", file=sys.stderr)
        return 1
    spacings = HoldoutSpacingSource(args.raw_root, manifest)
    masks = HoldoutMaskSource(args.label_root, manifest)
    writer = P2HoldoutSampleWriter(merged, run_record, args.raw_root, args.out_root, device, print)
    entries = writer.write(cohort, spacings, masks)
    suffix = f"_shard_{args.shard}" if args.num_shards > 1 else ""
    manifest_path = Path(args.out_root) / f"samples{suffix}.json"
    manifest_path.write_text(json.dumps(entries, indent=1) + "\n")
    print(f"wrote {len(entries)} entries -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
