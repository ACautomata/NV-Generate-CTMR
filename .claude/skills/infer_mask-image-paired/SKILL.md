---
name: infer_mask-image-paired
description: Reference guide to the retired paired mask + image pipeline's configuration, mask-path dispatch, and output contract. No canonical live runner currently exists; use this to assess or prepare a future runner, not to launch one. Trigger when the user asks "how does LDMSampler work", "how does the paired pipeline chain the stages", or needs the retired README §2.3 CT Paired Image/Mask configuration contract.
---

# Mask + image paired reference

This reference records the **paired generation** contract: a mask stage followed by a ControlNet-conditioned image stage. The one-command orchestrator (which instantiated `LDMSampler` and called `sample_multiple_images`) retired with the scripts layer (issue #143; git history is its reproduction anchor).

> **Entry status — no live runner.** The package retains lower-level stage components but no canonical `ctmr` command owns the former orchestration (model assembly, Path A/B dispatch, retry policy, NIfTI I/O, and output publication). This document is not an invocation guide, and the component symbols below must not be treated as a drop-in command replacement. Use git history only to replay a historical run; a future dedicated ticket must restore a live runner.

For a **live** no-mask workflow, use [`infer_image-only`](../infer_image-only/SKILL.md). The image-from-mask configuration reference is [`infer_image-from-mask`](../infer_image-from-mask/SKILL.md).

## Retired entry contract

The former orchestrator took these fixed inputs:

```text
-t configs/config_network_<network>.json
-i configs/config_infer.json
-e configs/environment_<variant>.json
--random-seed 0
--version <variant>
```

> ⚠️ **`ddpm-ct` requires `num_inference_steps = 1000`** (vs 30 for `rflow-ct`). The retired orchestrator auto-applied this when `generate_version == "ddpm-ct"`; a future replacement must preserve that behavior. This is 33× slower than `rflow-ct` but produces equivalent quality.

Three configs fed the run:

- `-t` network architecture (`config_network_rflow.json` or `config_network_ddpm.json`).
- `-i` inference parameters (`config_infer.json` — `body_region`, `anatomy_list`, `output_size`, `spacing`, `controllable_anatomy_size`, etc.).
- `-e` environment paths (`environment_rflow-ct.json` or `environment_ddpm-ct.json` — checkpoint paths, label dicts, mask database).

### Reference configuration: paired chest CT (Path B)

A former 24 GB / 512×512×128 Path-B configuration used `config_infer_24g_512x512x128.json` with:

```json
"body_region": ["chest"],
"anatomy_list": ["liver", "spleen", "lung"],
"controllable_anatomy_size": [],
"num_output_samples": 1
```

The retired orchestrator published a CT / label NIfTI pair beneath the environment `output_dir`. For **Path A** (organ/tumor-size control), `controllable_anatomy_size` was a non-empty list of `(organ_name, size)` tuples, e.g. `[["pancreas", 0.5], ["hepatic tumor", 0.3]]`, and `body_region` stayed empty. The dispatch flowchart below records the behavior a future runner must retain.

## How the (retired) orchestrator chose the mask path

```text
controllable_anatomy_size non-empty?
            │
   ┌────────┴─────────┐
  YES                 NO
   │                   │
   ▼                   ▼
prepare_anatomy_size_  find_masks(body_region, anatomy_list, ...)
condition()            (look up real training masks; resample if needed)
   │                   │
   ▼                   ▼
sample_one_mask()      read_mask_information(mask_file)
(diffusion-generated)  (no diffusion, just load + transform)
   │                   │
   └────────┬──────────┘
            ▼
   prepare_one_mask_and_meta_info()  (assign 1.5mm iso affine, derive
                                      top/bottom_region_index)
            │
            ▼
   sample_one_pair()  (ControlNet + image DM — see infer_image-from-mask skill)
            │
            ▼
   quality_check_ct(image, mask)
            │
        passed?
            │
   ┌────────┴────────┐
  YES               NO
   │                 │
save image+label   re-generate (up to the retired orchestrator's max_try_time=2 retries)
```

### Two paths to obtain a mask

Which path runs is driven by `controllable_anatomy_size` in `config_infer.json`:

- **Path A — diffusion from scratch** (`controllable_anatomy_size` non-empty): the user provides `(organ, size)` tuples; the mask DM samples a new mask conditioned on the resulting `anatomy_size` 10-d vector. Use this when you want to *control* organ/tumor presence and size.
- **Path B — training-mask database lookup** (`controllable_anatomy_size` empty): a real training mask matching `body_region` + `anatomy_list` + `spacing` + `output_size` is retrieved and lightly augmented so the output isn't a verbatim copy. No diffusion runs in the mask stage. Use this when you only need a plausible mask of the right anatomy and don't care about controlling specific organ sizes.

Both paths produce a MAISI-vocabulary mask that then feeds the image stage. For the per-path knobs and the `anatomy_size` slot table, see [`infer_mask-only`](../infer_mask-only/SKILL.md). The image stage that consumes the mask is documented in [`infer_image-from-mask`](../infer_image-from-mask/SKILL.md).

## `dim` and `spacing` — same FOV rules as image-only

> ⚠️ **FOV (= `dim × spacing`) is the #1 quality knob.** See the **"Why FOV matters"** section at the top of [`infer_image-only`](../infer_image-only/SKILL.md) — same warning applies here. Out-of-distribution FOVs produce unusable output even when the validator accepts the inputs.

The mask + image pipeline uses **the same** `output_size` and `spacing` constraints as image-only inference — see the `infer_image-only` skill for the table of recommended `(dim, spacing)` per anatomical target and the hard constraints from `check_input_ct` / `check_input_mr`.

Additional FOV considerations specific to the paired pipeline:

- The **mask DM** was pretrained at **256³ × 1.5 mm iso** (= 384 mm cube FOV). Generating a mask at significantly different shape forces the `ensure_output_size_and_spacing` resampling, which degrades label boundaries. Stay at or near 256³ × 1.5mm for Path A.
- For Path B (mask DB lookup), the candidate masks are themselves drawn from a training-FOV distribution — `find_closest_masks` picks the closest matches, but the closer your requested FOV is to a mode of that distribution, the less reshaping is needed.

## How to configure a run

### 1. `modality` → driven by your anatomy

Pick the modality code matching what you want to generate (full list in `configs/modality_mapping.json`). This mask-image paired pipeline is **CT-only** (the mask DM and ControlNet are CT-only — no MR ControlNet exists), so `modality = 1`. For MR generation use [`infer_image-only`](../infer_image-only/SKILL.md). For recommended FOVs per anatomy, see `docs/inference.md#recommended-spacing-for-ct`.

### 2. `autoencoder_sliding_window_infer_size`, `autoencoder_sliding_window_infer_overlap`, `autoencoder_tp_num_splits` → from GPU memory + `output_size`

Validated presets (drawn from `configs/config_infer_<XXg>_<dim>.json`):

| GPU mem | `output_size` | `autoencoder_sliding_window_infer_size` | `autoencoder_sliding_window_infer_overlap` | `autoencoder_tp_num_splits` |
|---|---|---|---|---|
| 16 GB | 256×256×128 | [96, 96, 96] | 0.25 | 2 |
| 16 GB | 256×256×256 | [48, 48, 64] | 0.6666 | 4 |
| 16 GB | 512×512×128 | [64, 64, 32] | 0.5 | 2 |
| 24 GB | 256×256×256 | [64, 64, 64] | 0.25 | 4 |
| 24 GB | 512×512×128 | [80, 80, 32] | 0.4 | 2 |
| 24 GB | 512×512×512 | [64, 64, 48] | 0.4 | 2 |
| 32 GB | 512×512×512 | [80, 80, 48] | 0.4 | 4 |
| 80 GB | 512×512×512 | [80, 80, 80] | 0.4 | 4 |
| 80 GB | 512×512×768 | [80, 80, 96] | 0.4 | 4 |

Tuning rules if no preset matches:

- **OOM** → shrink `autoencoder_sliding_window_infer_size` (must be divisible by 16), or raise `autoencoder_tp_num_splits` to the next value in `{2, 4, 8, 16}`.
- **Seam artifacts** → raise `autoencoder_sliding_window_infer_overlap` toward `0.6667`.
- **Speed** → lower the overlap toward `0.25`, then enlarge the sliding-window size if VRAM permits.

### 3. `spacing` → from FOV and `output_size`

```text
spacing[i] = FOV[i] / output_size[i]
```

Pick FOV from the anatomy-recommended table (step 1), pick `output_size` from the GPU preset (step 2), compute `spacing`.

### 4. Modality-CFG → not used in this pipeline

This pipeline is CT-only and modality is fixed at `CT=1`, so modality-CFG has nothing to amplify. The modality-CFG version of `cfg_guidance_scale` lives in `config_maisi_diff_model_*.json` and is read by the image-only entry `ctmr.infrastructure.maisi_engine.diff_model_infer` ([`infer_image-only`](../infer_image-only/SKILL.md)), where it is required for MR — see that skill.

### 5. `cfg_guidance_scale` (tumor-CFG in this pipeline)

Classifier-free guidance (CFG) scale on tumor presence. CFG runs the model twice per step (mask as-is vs mask with `remove_tumors()`) and amplifies the difference, strengthening tumor signal in the synthesized image. CT-only. `0` (default) = off. `1..5` = stronger tumor enforcement, growing artifact risk above 5. Doubles per-step compute when `> 0`. Same key name as the modality-CFG (step 4) — semantics depend on which entry reads the config: tumor here, modality in the image-only `ctmr.infrastructure.maisi_engine.diff_model_infer`.

### 6. `num_inference_steps`

Driven by the scheduler the variant uses, not by GPU memory:

- `rflow-ct` → **30** (RFlow scheduler).
- `ddpm-ct` → **1000** (DDPM scheduler). Lower values emit a warning and degrade quality — not optional.
- `mask_generation_num_inference_steps` → always **1000**: the mask DM is DDPM regardless of which image-DM variant you pick.

## Configuration knobs

Live in the three configs:

- `config_network_*.json` — fixed network architecture; not usually edited.
- `config_infer.json` — user intent (see below).
- `environment_*.json` — paths.

Key `config_infer.json` knobs:

| Key | Effect |
|---|---|
| `body_region` | List of regions present in the requested mask: any of `["head", "chest", "thorax", "abdomen", "pelvis", "lower"]`. Used by Path B only (`find_masks` filter). |
| `anatomy_list` | List of organ names from `configs/label_dict.json` that must be present. Used by `find_masks` (Path B) AND as the post-process filter (`filter_mask_with_organs`) for both paths. |
| `controllable_anatomy_size` | Empty list → Path B. Non-empty list of `(organ_name, size)` tuples → Path A (diffusion-generated mask). At most 10 entries; at most 1 tumor. |
| `output_size` | Target volume shape. Hard constraints apply (see `infer_image-only` skill). |
| `spacing` | Target voxel spacing (mm). Hard constraints apply. |
| `modality` | Modality code (1=CT, 8..32=MR variants). |
| `num_inference_steps` | RFlow → 30, **DDPM → 1000**. ⚠️ For `ddpm-ct` you must set this to 1000; the retired orchestrator auto-applied this override — recompose the call with it set explicitly. |
| `mask_generation_num_inference_steps` | **1000** — the mask DM always uses DDPM regardless of which image-DM variant you pick. Setting this lower silently degrades mask quality. |
| `cfg_guidance_scale` | Strengthens **tumor** signal (this pipeline is CT-only). `0` (default) = off; `1..5` = stronger tumor enforcement, more artifact risk. The same key name in `config_maisi_diff_model_*.json` is the modality-CFG used by MR image-only inference — see [`infer_image-only`](../infer_image-only/SKILL.md). |

## Output

For each successful generation, two files are saved to `output_dir`:

- `sample_<timestamp>_image.nii.gz` — synthetic CT/MR
- `sample_<timestamp>_label.nii.gz` — paired mask (filtered to `anatomy_list`)

## Related entries

| Entry | Role |
|---|---|
| former `inference.py` (scripts layer, git history) | Paired-pipeline entry point for this skill. |
| former `sample.py` (scripts layer, git history) — `LDMSampler` | Orchestrator: dispatched the mask stage and the image stage, applied the QC retry loop. |
| former `ctmr.infrastructure.dataio.sample_mask` (git history, issue #230 / ADR-0018) | Path A mask DM (`ldm_conditional_sample_one_mask`). |
| former `ctmr.infrastructure.dataio.find_masks` (git history, issue #230 / ADR-0018) | Path B exact-match DB lookup (`find_masks`). |
| `ctmr.application.generation.mask.inference` | Former image-from-mask pipeline function (the orchestrator's image stage; its wrapper kernel was deleted with issue #175 — git history; only the conditioning helpers survive). |
| `hf download` (CLI) — see the download-models skill | Downloads mask DM + image DM + ControlNet weights (the retired `download_model_data` assembly, git history per ADR-0018). |

## Related skills

- [`infer_mask-only`](../infer_mask-only/SKILL.md) — mask-stage details.
- [`infer_image-from-mask`](../infer_image-from-mask/SKILL.md) — image-stage details.
- [`infer_image-only`](../infer_image-only/SKILL.md) — image-only path (no mask, including MR); covers the FOV / `dim` / `spacing` recommendations.
- [`download-models`](../download-models/SKILL.md) — fetch checkpoints first.
