---
name: download-models
description: How to download pretrained checkpoints (CT, MR, MR-Brain variants) and optional auxiliary data (mask database, anatomy-size-conditions JSON) for NV-Generate-CTMR inference. Trigger when the user asks "where are the checkpoints", "how do I download the model weights", "what does download_model_data do", or hits a missing-checkpoint error.
---

# Downloading pretrained models + auxiliary data

This skill covers the model-download assembly. The former Python assembly
(`ctmr.infrastructure.dataio.downloads`, the `download_model_data` script)
**retired to git history** (issue #230 / ADR-0018 — the `ctmr data` command
family never landed); weights are downloaded directly from HuggingFace with
the `hf` CLI. The variant → file mapping below is unchanged.

## TL;DR

```bash
hf download nvidia/NV-Generate-CT models/autoencoder_v1.pt --local-dir .   # every CT/MR-Brain variant needs the image AE
```

then add the per-variant files listed below (`hf download <repo> <file> --local-dir .`). Files land under `./models/` (weights) and `./datasets/` (optional auxiliary data).

## What gets downloaded per variant

Source: HuggingFace Hub. The retired assembly also pinged each repo's `config.json` once so HuggingFace's download counter ticks — `hf download` does not.

### `rflow-ct` (CT, Rectified Flow — recommended for CT)

Always downloaded (`models/`):

- `autoencoder_v1.pt` (image AE) — from `nvidia/NV-Generate-CT`
- `mask_generation_autoencoder.pt` — from `nvidia/NV-Generate-CT`
- `mask_generation_diffusion_unet.pt` — from `nvidia/NV-Generate-CT`
- `diff_unet_3d_rflow-ct.pt` (image DM) — from `nvidia/NV-Generate-CT`
- `controlnet_3d_rflow-ct.pt` — from `nvidia/NV-Generate-CT`

For the full paired-inference pipeline, also download (`datasets/`, from `nvidia/NV-Generate-CT`):

- `all_anatomy_size_conditions.json` — anatomy-size database for `prepare_anatomy_size_condition` (Path A in mask-image paired inference)
- `all_masks_flexible_size_and_spacing_4000.zip` — training-mask database for `find_masks` (Path B)
- `candidate_masks_flexible_size_and_spacing_4000.json` — index for the mask DB

### `ddpm-ct` (CT, DDPM — slower but supports body_region input)

Same as `rflow-ct` but swaps:

- `diff_unet_3d_ddpm-ct.pt` + `controlnet_3d_ddpm-ct.pt` (instead of the rflow variants)

And uses `candidate_masks_flexible_size_and_spacing_3000.json` (smaller mask index).

### `rflow-mr-brain` (Brain MRI, Rectified Flow)

Only the image-DM stack (no mask DM, no ControlNet):

- `autoencoder_v1.pt` — from `nvidia/NV-Generate-CT` (yes, MR-Brain reuses the CT image AE)
- `diff_unet_3d_rflow-mr-brain_v1.pt` — from `nvidia/NV-Generate-MR-Brain`

### `rflow-mr` (Other MRI, Rectified Flow)

- `autoencoder_v2.pt` — from `nvidia/NV-Generate-MR`
- `diff_unet_3d_rflow-mr.pt` — from `nvidia/NV-Generate-MR`

## Output layout

```text
./
├── models/
│   ├── autoencoder_v1.pt                      # image AE (CT + MR-Brain)
│   ├── autoencoder_v2.pt                      # image AE (MR)
│   ├── mask_generation_autoencoder.pt         # mask AE (CT only)
│   ├── mask_generation_diffusion_unet.pt      # mask DM (CT only)
│   ├── diff_unet_3d_<variant>.pt              # image DM
│   └── controlnet_3d_<variant>.pt             # ControlNet (CT only)
└── datasets/
    ├── all_anatomy_size_conditions.json       # CT infer_mask-only database
    ├── all_masks_flexible_size_and_spacing_4000.zip
    └── candidate_masks_flexible_size_and_spacing_4000.json
```

The paths above are exactly what the `environment_<variant>.json` configs expect, so as long as you run `hf download ... --local-dir .` from the repo root, no path edits are needed.

## Which files to skip

- **Image-only use**: skip the mask AE / mask DM (`mask_generation_*.pt`) and the `datasets/` auxiliary files — the image stack alone (AE + image DM, plus ControlNet for the mask-conditioned chains) is enough.
- **`controllable_anatomy_size` (Path A, diffusion-generated masks)**: still skip the `datasets/` mask database — only the model weights are needed.
- **Full paired-inference pipeline (Path B real-mask retrieval)**: download everything, including `all_masks_flexible_size_and_spacing_4000.zip` and the mask index JSON.

## License gating

Some HuggingFace repos require you to accept their license terms before downloading. If you hit a 403, visit the repo page and accept terms:

- [nvidia/NV-Generate-CT](https://huggingface.co/nvidia/NV-Generate-CT) — NVIDIA Open Model License
- [nvidia/NV-Generate-MR](https://huggingface.co/nvidia/NV-Generate-MR) — NVIDIA Non-Commercial
- [nvidia/NV-Generate-MR-Brain](https://huggingface.co/nvidia/NV-Generate-MR-Brain) — NVIDIA Open Model License

After accepting, set `HF_TOKEN` in the environment (the `huggingface_hub` client picks it up).

## Failure modes and retries

| Symptom | Cause | Fix |
|---|---|---|
| `huggingface_hub.errors.GatedRepoError` | License not accepted | Visit repo page, accept terms, retry |
| `requests.exceptions.ConnectionError` | Network drop | Just re-run — `hf_hub_download` resumes from cache |
| Partial file (size mismatch) | Interrupted download | Delete the partial file in `./models/` or `./datasets/` and re-run |
| Wrong checkpoint shape at inference | Stale cached checkpoint after a model update | Delete the stale local file, then re-run |

## Related entries

| Entry | Role |
|---|---|
| `git history` | The retired `ctmr.infrastructure.dataio.downloads` assembly (`download_model_data`, `fetch_to_hf_path_cmd`) lives only in git history (issue #230 / ADR-0018); the tables above carry its knowledge forward. |

## Related skills

- [`infer_image-only`](../infer_image-only/SKILL.md) — uses the image DM only (no ControlNet, no mask DM). Skip the mask stack.
- [`infer_mask-image-paired`](../infer_mask-image-paired/SKILL.md) — needs the full set (mask AE + mask DM + image DM + ControlNet). Download everything if you'll use Path B.
- [`infer_mask-only`](../infer_mask-only/SKILL.md) — mask-stage details.
- [`infer_image-from-mask`](../infer_image-from-mask/SKILL.md) — image-stage details.
