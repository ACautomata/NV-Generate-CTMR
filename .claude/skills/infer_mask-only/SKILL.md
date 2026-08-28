---
name: infer_mask-only
description: Reference guide to the retired mask-generation stage: Path A/B dispatch through config_infer.json, anatomy_size conditioning, and output-mask contract. It has no standalone live entry; use it to assess a future paired runner, not to launch one. Trigger when the user asks "what does controllable_anatomy_size do" or "how does Path A / Path B differ".
---

# Mask-generation reference (NV-Generate-CTMR)

This reference records the **mask-generation stage** formerly run inside the paired pipeline. Its one-command orchestrator (the `LDMSampler` class) retired with the scripts layer (issue #143), and it was never a standalone CLI. The Path A kernel remains in `ctmr.infrastructure.dataio.sample_mask`, but it is **not** a user entry point: model assembly, Path-B dispatch, I/O, and output publication have no canonical live runner. See [`infer_mask-image-paired`](../infer_mask-image-paired/SKILL.md) for the full retired contract.

The mask stage produced a 3D MAISI-labeled volume to condition the image LDM. **CT-only** — the mask DM was trained on CT masks, and there is no MR equivalent.

## Workflow

```text
[anatomy_size  ──┐
 (10-d vector)]   │  cross-attention conditioning
                  ▼
[random noise]──▶[Mask Diffusion UNet]──▶[mask latent (4-ch)]
                        DDPM loop
                                              │
                                              ▼ sliding-window AE decode
                                  [125-channel softmax]
                                              │ argmax
                                              ▼
                                       [labels 0..124]
                                              │ remap_labels via label_dict_124_to_132.json
                                              ▼
                            [MAISI 132-class label NIfTI (with body=200)]
                                              │ tumor-aware + general post-process
                                              ▼
                                          [final mask]
```

## Configuration: Path A vs Path B

The mask stage has two paths, dispatched by `controllable_anatomy_size` in `config_infer.json`:

| Path | Trigger | What happens |
|---|---|---|
| **Path A** — diffusion from scratch | `controllable_anatomy_size` non-empty | Mask DM samples a new mask conditioned on the anatomy_size vector. |
| **Path B** — training-mask DB lookup | `controllable_anatomy_size` empty | Look up a real training mask from `configs/all_mask_files_*.json` matching `body_region` + `anatomy_list` + `spacing` + `output_size`; apply light augmentation so the output isn't a verbatim copy. |

Knobs that drive these:

| Knob | Path | Effect |
|---|---|---|
| `controllable_anatomy_size` | A vs B switch | Non-empty list of `(organ_name, size)` tuples — at most 10 entries, at most 1 tumor — triggers Path A. Empty triggers Path B. |
| `body_region` | B | Filters the mask DB. Any subset of `["head", "chest", "thorax", "abdomen", "pelvis", "lower"]`. |
| `anatomy_list` | A and B | Required organs. Used by Path B's `find_masks` filter; also used by both paths as the post-process `filter_mask_with_organs` (only listed organs survive in the output). |
| `output_size`, `spacing` | A and B | Target shape and voxel spacing — see [`infer_mask-image-paired`](infer_mask-image-paired.md) for the GPU-memory presets table. |
| `mask_generation_num_inference_steps` | A | Always **1000** — the mask DM is DDPM regardless of the image-DM variant; lowering it silently degrades mask quality. |

## Input: the `anatomy_size` slot vector (Path A only)

When Path A runs, the user-specified `(organ_name, size)` tuples are turned into a 10-d vector with fixed slots:

| Index | Organ | Index | Tumor |
|---|---|---|---|
| 0 | gallbladder | 5 | lung tumor |
| 1 | liver | 6 | pancreatic tumor |
| 2 | stomach | 7 | hepatic tumor |
| 3 | pancreas | 8 | colon cancer primaries |
| 4 | colon | 9 | bone lesion |

Each slot value is either:

- A float in `[0, 1]` — desired size on a normalized scale, **or**
- `-1.0` — "no preference / don't care".

The pipeline snaps the user-specified vector to the closest entry in `configs/all_anatomy_size_conditions.json` (a database of size vectors from real training cases), then **overwrites** the user-specified slots with the user's exact values. This keeps the conditioning vector near the training distribution while honouring user intent.

## Output

A 3D integer NIfTI of MAISI labels with shape `(H, W, D)`. Contains MAISI organ labels (1..132 with gaps) and the body envelope `200`. Saved by the paired pipeline as `sample_<timestamp>_label.nii.gz` alongside the paired image.

## Output-size and spacing constraints

The pretrained mask DM was trained at **256×256×256 × 1.5 mm isotropic** (Path A). Resampling to your requested `output_size` and `spacing` happens automatically; major upsampling degrades label boundaries, so stay close to 256³ × 1.5 mm when feasible. For Path B, mask candidates are drawn from a training-FOV distribution — the closer your requested FOV is to a mode of that distribution, the less reshaping is needed.

## Related entries

| Entry | Role |
|---|---|
| `ctmr.infrastructure.dataio.sample_mask` | Path A core sampler: `ldm_conditional_sample_one_mask` (DDPM → softmax/argmax → label remap → post-process). |
| `ctmr.infrastructure.dataio.find_masks` | Path B exact-match DB lookup: `find_masks(body_region, anatomy_list, spacing, output_size, ...)`. |
| former `sample.py` (scripts layer, git history) — `LDMSampler` | Orchestrator: chose Path A or B based on `controllable_anatomy_size`, then chained the image stage. Hosted `LDMSampler.find_closest_masks` for Path B's closest-match fallback. |
| former `inference.py` (scripts layer, git history) | Paired-pipeline entry point (mask stage + image stage together). |
| `ctmr.infrastructure.dataio.mask_postprocess` | Label utilities: `remap_labels`, `general_mask_generation_post_process`. |

## Related skills

- [`infer_mask-image-paired`](../infer_mask-image-paired/SKILL.md) — the pipeline that drives this stage end-to-end.
- [`infer_image-from-mask`](../infer_image-from-mask/SKILL.md) — what happens to the mask after this stage.
- [`infer_image-only`](../infer_image-only/SKILL.md) — image-only generation (no mask DM involved).
