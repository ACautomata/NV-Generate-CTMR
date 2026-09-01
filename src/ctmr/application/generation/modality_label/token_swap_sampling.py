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

"""Diagnostic job D sampling arm (issue #209, parent #205): same-seed token-swap
generation for the modality-label dilution discrimination.

This module ONLY produces the sampling products: for every case of the fixed
16-case dev cohort (``DevCohortBuilder``, the same fixed cohort the dev FID
trend rides on) it samples five arms -- t1n 29 / t1c 34 / t2w 30 / t2f 31 plus
the pan-MR control 8 -- with the frozen P1 candidate checkpoint under the
frozen sampling recipe (cfg=10, 30 steps, RFlowScheduler). The per-case seed
is the frozen sampling rule's own hash pinned at the discriminated channel
(``token_dilution.SeedAnchor`` = sha256(case|t1c)) and is shared by all five
arms, so within one case the initial noise is bit-identical and every output
difference attributes to the token condition alone. Artifacts land as
``<samples_dir>/<case>_<arm>_seed<seed>.nii.gz`` (holdout filename family,
generalized to the diagnostic arms); existing files are skipped, so the arm
is re-entrant.

Everything downstream is statistics, and lives in
``ctmr.application.acceptance.distribution.token_dilution`` (variant=
diagnostic, never an acceptance verdict); the sugon host recipe that chains
sample → report is ``deploy/jobs/run_token_dilution_d.sh``. Per ADR-0016 the
denoising loop runs on the domain ``DiffusionModel`` through the sidecar's
``CandidateSampler`` (composition -- the sampler's model loading and
per-sample rules are reused verbatim, no recipe value re-decided here); per
ADR-0019 §2-§3 (#272) the engine face rides the port the composition root
assembles (``ctmr.wiring.generate``).

Usage (sugon, one DCU; VAE path comes from the env json's
trained_autoencoder_path, resolvable from the working directory):
    python -m ctmr.application.generation.modality_label.token_swap_sampling \
        --dev-list ... --emb-root ... --ckpt <frozen candidate .pt> \
        -e env.json -c model.json -t network.json --samples-dir DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib

from ctmr.application.acceptance.distribution.token_dilution import (
    ARM_ORDER,
    TOKEN_ARMS,
    SeedAnchor,
)
from ctmr.application.generation.devices import add_device_flag, resolve_device
from ctmr.application.generation.modality_label.monitor import CandidateSampler, CohortSpacingSource
from ctmr.application.generation.trend import DevCohortBuilder
from ctmr.domain.dm_output_grid import V1_DM_OUTPUT_GRID
from ctmr.wiring.generate import modality_label_engine

# The frozen sampling recipe values (dev-sidecar convention): cfg=10 with 30
# RF steps. Not knobs -- pinned literals matching what the candidate's
# holdout/dev evidence was generated under.
TOKEN_SWAP_CFG = 10.0
TOKEN_SWAP_STEPS = 30


class TokenSwapSampler:
    """Generates the five-arm per-case products with the sidecar's sampler.

    Composition, not re-decision: model loading, the denoising loop, the VAE
    decode and the int16 ×1000 output convention all come from
    ``CandidateSampler`` (the same code path the dev evidence used); this arm
    adds only the token-swap loop and the diagnostic filename family.
    """

    def __init__(self, args, device, engine):
        self._sampler = CandidateSampler(args, device, None, engine)
        self._device = device

    def sample_cohort(self, checkpoint_path, cohort, spacings, out_dir) -> int:
        model, recon = self._sampler.load_models(checkpoint_path)
        written = 0
        for item in cohort:
            case = item["case"]
            seed = SeedAnchor.of(case)
            spacing = spacings.spacing_of(case)
            for arm in ARM_ORDER:
                out = Path(out_dir) / f"{case}_{arm}_seed{seed}.nii.gz"
                if out.is_file():
                    continue
                data = self._sampler.sample_one(model, recon, TOKEN_ARMS[arm], spacing, seed)
                out.parent.mkdir(parents=True, exist_ok=True)
                # Ruling #6: the diagnostic products declare the v1 DM's real sampling
                # spacing too -- the same write protocol as the sidecar (issue #249).
                nib.save(nib.Nifti1Image(data, affine=V1_DM_OUTPUT_GRID.affine()), out)
                written += 1
                print(f"[sample] {case} {arm} (token {TOKEN_ARMS[arm]}, seed {seed}) -> {out.name}", flush=True)
        return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev-list", required=True, help="the P1 dev list json (fixed dev cohort source)")
    parser.add_argument("--emb-root", required=True, help="embedding companion root for per-case spacings (t1n entry)")
    parser.add_argument("--ckpt", required=True, help="the frozen candidate checkpoint (epoch_20.pt), loaded read-only")
    parser.add_argument("-e", "--env_config_path", required=True)
    parser.add_argument("-c", "--model_config_path", required=True)
    parser.add_argument("-t", "--model_def_path", required=True)
    parser.add_argument("--samples-dir", required=True, help="artifact directory for the five per-case volumes (never git)")
    add_device_flag(parser)
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    print(f"[job-d] device={device}; variant=diagnostic -- 冻结 checkpoint 只读,不产生任何验收判定", flush=True)

    engine = modality_label_engine()  # the composition root's engine assembly (ADR-0019 §2)
    merged = engine.load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    merged.cfg_guidance_scale = TOKEN_SWAP_CFG
    merged.diffusion_unet_inference = (
        merged.diffusion_unet_inference if hasattr(merged, "diffusion_unet_inference") else {"num_inference_steps": TOKEN_SWAP_STEPS}
    )

    cohort = DevCohortBuilder(args.dev_list).build()
    samples_dir = Path(args.samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)
    (samples_dir / "cohort.json").write_text(json.dumps({"cohort": cohort, "seed_anchor": "sha256(case|t1c), 五臂共用"}, indent=1) + "\n")
    spacings = CohortSpacingSource(args.dev_list, args.emb_root)
    written = TokenSwapSampler(merged, device, engine).sample_cohort(args.ckpt, cohort, spacings, samples_dir)
    print(f"[job-d] {len(cohort)} cases x {len(ARM_ORDER)} arms; written {written} new volumes -> {samples_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
