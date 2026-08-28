---
name: infer_image-from-mask
description: Reference guide to the CT image-from-mask pipeline's input mask, configuration, output contract, and package homes. The former end-to-end entry is retired and no canonical live runner currently exists; use this to assess or prepare a future runner, not to launch one. Trigger when the user asks "what mask format does NV-Generate-CTMR expect", "how do I add the body envelope", or needs the retired image-from-mask workflow's configuration contract. CT-only (no MR ControlNet exists in this repo).
---

# Image-from-mask reference (NV-Generate-CTMR)

This reference records the **image-from-mask** pipeline contract: a user-supplied mask conditions the image LDM through a ControlNet branch. **CT-only** — the released ControlNet checkpoints were trained on CT masks; no MR ControlNet exists.

> **Entry status — no live runner.** The one-command image-from-mask entry retired with the scripts layer (issue #143), and its full responsibility (model assembly, NIfTI I/O, validation, sampling, and output publication) has not yet landed behind a canonical `ctmr` command. The package function `ctmr.application.generation.mask.inference.ldm_conditional_sample_one_image_from_mask` is a lower-level kernel, **not** a user entry point. Do not call it from this page as an incomplete replacement. Use git history only to reproduce a historical run; a future dedicated ticket must restore a live runner.

For a no-mask **live** workflow (including MR), use [`infer_image-only`](../infer_image-only/SKILL.md). The paired-mask reference is [`infer_mask-image-paired`](../infer_mask-image-paired/SKILL.md).

## Retired entry contract

The former runner accepted three configuration layers plus a mask and seed:

- `-t` — network architecture (`config_network_rflow.json` or `config_network_ddpm.json`).
- `-i` — inference hyper-parameters (`config_infer.json`, or one of the `config_infer_<XXg>_<dim>.json` presets — see [Configuration](#configuration) below).
- `-e` — environment paths (`environment_rflow-ct.json` or `environment_ddpm-ct.json`).
- `--mask` — a mask NIfTI satisfying [Input: the mask](#input-the-mask).
- `--random-seed` — optional integer for deterministic sampling.

### Reference configuration: MAISI-vocabulary mask

Assume `~/your_mask.nii.gz` contains MAISI 132-class labels with `body=200` (see [Input: the mask](#input-the-mask)). A former 24 GB / 512×512×128 configuration used `config_infer_24g_512x512x128.json` with:

```json
"output_size": [512, 512, 128],
"spacing": [0.75, 0.75, 4.0],
"modality": 1,
"cfg_guidance_scale": 0.0,
"num_inference_steps": 30,
"autoencoder_sliding_window_infer_size": [80, 80, 32],
"autoencoder_sliding_window_infer_overlap": 0.4,
"autoencoder_tp_num_splits": 2
```

The retired runner published `sample_<timestamp>_image.nii.gz` (CT in HU `[-1000, 1000]`) and `sample_<timestamp>_label.nii.gz` under the environment `output_dir`.

## Input: the mask

### Format

A **1-channel integer NIfTI** (`.nii` or `.nii.gz`) whose voxel values are drawn from the **MAISI 132-class label vocabulary** plus the body envelope:

| Value | Meaning |
|---|---|
| `0` | background |
| `1..132` (with gaps — see [`configs/label_dict.json`](../configs/label_dict.json)) | organ / structure labels (e.g. `1`=liver, `3`=spleen, `4`=pancreas, `5`=right kidney, `14`=left kidney, `28..32`=lung lobes, `33..57`=vertebrae) |
| `200` | **body envelope** — every body voxel not labeled with a specific organ |

`200` is the critical outer-body label. The CT ControlNet expects it; `nv-segment-ct` does not produce it. **You must add it yourself** during preprocessing.

### Producing a valid mask from a CT image

#### Option A (recommended): `nv-segment-ct` + add body envelope

1. Run [`nv-segment`](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR) (the NV-Segment-CTMR bundle, modality `CT_BODY`) on your CT image. It outputs a 1-channel NIfTI **already in the MAISI vocabulary** — no label remapping needed.
2. Add the body envelope (label `200`) with the retired `add_body_envelope` utility (git history: the former `utils.py`; its ~20-line body labels every non-organ body voxel `200`).
3. Save as a 1-channel integer NIfTI.

#### Option B: another segmenter + remap + add body envelope

1. Run your segmenter (e.g. TotalSegmentator) on the CT image.
2. **Remap label IDs** to the MAISI 132-class space defined in [`configs/label_dict.json`](../configs/label_dict.json) by matching anatomical structure names. Structures not present in MAISI must be set to `0`. If you skip this step or get the mapping wrong, the generated CT will be unusable.
3. Add the body envelope (label `200`) as in Option A step 2.
4. Save as a 1-channel integer NIfTI.

### Common pitfall: the 0..124 AE-channel space is NOT the right space

The mask AE inside this repo internally produces a 125-channel softmax that gets `argmax`'d to integer labels in `[0, 124]`. Those values are remapped to the MAISI 132-class vocabulary via [`configs/label_dict_124_to_132.json`](../configs/label_dict_124_to_132.json) before the CT ControlNet sees them. So:

- ✅ Correct user-mask input: MAISI 132-class labels with `body=200`.
- ❌ Incorrect: feeding `0..124` AE-channel-space labels.

If you have a mask in `0..124` space (e.g. from intermediate steps of a custom mask-DM pipeline), apply `remap_labels(mask, configs/label_dict_124_to_132.json)` first before passing it here.

### Validation in the CLI

The retired entry's `validate_user_mask` step (git history) did the following — replicate its checks when recomposing:

- Confirm the mask is 1-channel integer NIfTI.
- Warn (not error) if any voxel value is outside the MAISI 132-class vocabulary (`{0..132} ∪ {200}`).
- Auto-resample shape/spacing to a valid `(output_size, spacing)` target (with a warning) if needed.

If many voxel values fall outside the vocabulary you almost certainly forgot a remap step.

## Configuration

The config the retired entry took as `-i` is `config_infer.json` (or one of the `config_infer_<XXg>_<dim>.json` presets). For the **per-knob walkthrough** — `modality`, `output_size`, `spacing = FOV / output_size`, AE sliding-window knobs (`_size` / `_overlap` / `_tp_num_splits`) keyed to GPU memory, `cfg_guidance_scale`, `num_inference_steps` — see the **"How to configure a run"** section in [`infer_mask-image-paired`](../infer_mask-image-paired/SKILL.md). The same presets apply here — the recomposed call consumes the same `config_infer.json`.

Quick reminder of the CT-specific knob most relevant to this skill:

- `cfg_guidance_scale` (tumor-CFG in this pipeline) — classifier-free guidance scale on tumor presence. CFG runs the model twice per step (mask as-is vs mask with `remove_tumors()`) and amplifies the difference, strengthening tumor signal in the synthesized image. `0` (default) = off, correct whenever the mask has no tumors or you want unsteered output. `1..5` = stronger tumor enforcement (more artifact risk above 5). Doubles per-step compute when `> 0`. The same key name also lives in `config_maisi_diff_model_*.json` as the modality-CFG for MR image-only inference — see [`infer_image-only`](../infer_image-only/SKILL.md).

### Hard constraints on `output_size` + `spacing`

Validated by `check_input_ct` / `check_input_mr` (in `ctmr.infrastructure.maisi_engine.inference_primitives`):

- `output_size[0] == output_size[1]`
- `output_size[0] ∈ {256, 384, 512}`
- `output_size[2] ∈ {128, 256, 384, 512, 640, 768}`
- `spacing[0] == spacing[1]`
- `spacing[0] ∈ [0.5, 3.0]` mm, `spacing[2] ∈ [0.5, 5.0]` mm
- FOV_xy ≥ 256 mm for head, ≥ 384 mm for abdomen / body

See [`infer_image-only`](infer_image-only.md) for the recommended `(dim, spacing)` per anatomical target.

## Output

Two files in the directory given by `output_dir` (from the environment config):

- `sample_<timestamp>_image.nii.gz` — synthetic CT volume.
- `sample_<timestamp>_label.nii.gz` — the (resampled) input mask returned for downstream filtering.

Image intensity ranges:

| Modality | Output type | Voxel value range |
|---|---|---|
| CT (modality code `1..7`) | int16 NIfTI | HU, clipped to `[-1000, 1000]` |
| MR (codes `8..32`) — n/a for this pipeline; provided for context | int16 NIfTI | `[0, +∞)` |

Background voxels (where the mask is `0`) are set to `a_min` (CT: `-1000` HU) via `crop_img_body_mask`, preserving the body silhouette.

## Related entries

| Entry | Role |
|---|---|
| `ctmr.application.generation.mask.inference` | Home of this workflow's pipeline: `ldm_conditional_sample_one_image_from_mask` loads models, `binarize_labels` conditions, `crop_img_body_mask` regularizes. |
| former `sample.py` (scripts layer, git history) — `LDMSampler.sample_one_pair` | Paired-pipeline wrapper around the same core sampler. |
| `ctmr.infrastructure.dataio.downloads` (`download_model_data`) | Downloads ControlNet + image AE + image DM weights. Run once before inference. |
| former `utils.py` (`add_body_envelope`, `remap_labels`) — retired (git history) | Mask preprocessing utilities — needed for Options A/B above and for converting 0..124 → 132-class. (`binarize_labels` moved into `ctmr.application.generation.mask.inference`.) |
