---
name: finetune_image-from-mask_data-prep
description: Reference guide to the CT ControlNet finetuning data contract: VAE embeddings, pseudo labels, combined labels, label remapping, JSON lists, and fold/weighted-loss semantics. The complete generic preparation pipeline has no canonical live entry because body-envelope assembly retired without a replacement; use this to assess or prepare a future runner, not as an end-to-end launch guide. CT-only.
---

# ControlNet finetuning data reference

This reference describes the data contract for CT ControlNet finetuning. The generic preparation pipeline is **not currently end-to-end runnable**: image embedding has a vendored engine entry, but the required body-envelope assembly retired without a replacement. Do not treat the individual steps below as a complete launch recipe; use git history only to reproduce a historical run, and restore the missing boundary in a dedicated future ticket.

The reference applies when each case contains:

- an **original image** (`*.nii.gz`), and
- an **original label mask** (`*.nii.gz`) with one or more classes.

It records the three derived files the ControlNet training loop consumes and the MAISI label-remapping rules. **CT-only.** The released ControlNet checkpoints were trained on CT masks; there is no MR ControlNet in this repo. The reference walkthrough is [data/README.md §4.3](../../../data/README.md#43-example-finetuning-on-a-new-dataset) (the C4KC-KiTS Kidney-Tumor example, which maps a new class to index `129`).

## What you have → what you need

```text
            |-image.nii.gz                # original image          ← you have this
KiTS-000* --|-mask.nii.gz                 # original mask           ← you have this
            |-image_emb.nii.gz            # VAE-encoded embedding         (Step 1)
            |-mask_pseudo_label.nii.gz    # NV-Segment labels + body (200) (Step 2)
            |-mask_combined_label.nii.gz  # pseudo labels + your remapped mask  (Steps 3–4)
```

The training loop reads only `*_emb.nii.gz` (as `image`) and `*_combined_label.nii.gz` (as `label`). Steps 1–4 below produce them.

---

## Step 1 — Image embedding (`*_emb.nii.gz`)

VAE-encode each original image with the vendored engine entry `ctmr.infrastructure.maisi_engine.create_training_data` (run as `python -m …`). It resamples every image to the nearest multiple of 128 per axis, runs the autoencoder encoder (sliding-window, AMP), and writes `<image>_emb.nii.gz`.

Point an `environment_*` config at your data and use **`autoencoder_v1.pt`** (the CT ControlNet's autoencoder):

```json
{
    "trained_autoencoder_path": "./models/autoencoder_v1.pt",
    "data_base_dir": "./datasets/my_dataset",
    "embedding_base_dir": "./datasets/my_dataset",
    "json_data_list": "./datasets/my_dataset.json"
}
```

```bash
python -m ctmr.infrastructure.maisi_engine.create_training_data \
    -t ./configs/config_network_rflow.json \
    -c ./configs/config_maisi_diff_model_rflow-ct.json \
    -e ./configs/<your_env>.json -g 1
```

> The data list for **this step** must carry a `modality` field per entry (e.g. `"ct"`) — it drives intensity normalization. See [modality_mapping.json](../configs/modality_mapping.json) for valid values. Encoding up front (instead of inside the training loop) is what keeps GPU memory low during finetuning.

## Step 2 — Whole-body labels + body envelope (`mask_pseudo_label*.nii.gz`)

**Blocked in the live package.** A valid pseudo label requires a MAISI-vocabulary whole-body segmentation with the body envelope (`200`). NV-Segment (`CT_BODY`) can supply organ labels, but the required body-envelope assembly utility retired with the scripts layer and has no canonical package replacement. Do not claim that generic data preparation is complete until a dedicated runner restores this operation. For historical reproduction only, its implementation is available in git history.

## Step 3 — Remap your mask, then combine (`mask_combined_label*.nii.gz`)

This is the key step. The combined mask = the Step-2 pseudo label (organs + body envelope `200`) with **your mask written on top** in MAISI indices. Two sub-steps:

### 3a. Remap your label values to MAISI indices

For every class in your original mask, decide its MAISI index:

- **Class already exists in MAISI** (liver, kidney, spleen, …) → use its existing index from [label_dict.json](../configs/label_dict.json) (e.g. `liver=1`, `left kidney=14`, `right kidney=5`).
- **Class is new / unseen** (your tumor, lesion, device — not in the vocabulary) → assign it to **any unclaimed integer below 256**.

**The rule for a new class: pick any integer in `0–255` that isn't already used and isn't reserved.** ControlNet supports up to 256 labels (`0–255`). Don't reuse an index a real anatomical class already owns, don't use `0` (background), and don't use `200` (body envelope). Anything else is fair game — there is nothing special about the `dummy` names below; they're just **pre-named convenience slots**.

**Free indices that collide with nothing** (no existing label uses them): **`133–199`** and **`201–255`** — 123 values, wide open. Pick from here if you want zero risk of clobbering an existing class.

**Pre-named `dummy` placeholders** in `label_dict.json` (handy because they already have an entry you can rename — see [§ below](#training-config-touch-ups)):

| Placeholder | Index | |
|---|---|---|
| `dummy6` | **129** | the slot the C4KC Kidney-Tumor example uses |
| `dummy7` | 130 | |
| `dummy8` | 131 | |
| `dummy1`–`dummy5` | 2, 16, 18, 20, 21 | lower indices, interspersed among real organs |

> Whatever index you pick — a `dummy` slot or a fresh integer like `150` — **add a named entry for it in `label_dict.json`**. (Optionally also list it in `weighted_loss_label` to emphasize it during training — see below.) The only hard constraints are: integer, `0–255`, not already claimed, not `0`, not `200`.

### 3b. Combine: write your remapped mask on top

Overlay your remapped mask onto the Step-2 pseudo label (organs + body envelope `200`). The repo provides the remap building block; the overlay is a small step you assemble:

```python
import torch
from ctmr.infrastructure.dataio.augmentation import remap_labels   # remap_labels(tensor, {old_value: new_index})

# your_mask:  integer label tensor from mask.nii.gz
# pseudo:     Step-2 pseudo label — MAISI organ labels + body envelope (200)

# 3a: liver(1) and right-kidney(2) exist in MAISI; my new lesion(3) is unseen -> dummy6 (129)
remapped = remap_labels(your_mask, {1: 1, 2: 5, 3: 129})

# 3b: write your foreground classes on top, leave the organ/body context intact
combined = pseudo.clone()
combined[remapped > 0] = remapped[remapped > 0]
# save `combined` as mask_combined_label*.nii.gz
```

(The same remap-from-JSON pattern existed in the former `ctmr.infrastructure.dataio.mask_postprocess.remap_labels` — retired to git history, issue #230 / ADR-0018. The inline remap above is the live recipe.)

---

## Step 4 — Put the combined label on the encoded-image grid

**The combined label's spatial size must be 4× the latent per axis** (the training loop does not auto-resample, so a mismatch errors out). Resample it there with **nearest-neighbor** (never linear/bspline — they invent fractional class IDs):

```python
import torch.nn.functional as F
# image_size = 4× the latent, e.g. [512, 512, 128] for a 128×128×32 latent (NOT the latent size)
combined = F.interpolate(combined.float()[None, None], size=image_size, mode="nearest")[0, 0].long()
```

---

## Build the JSON data list

One JSON pairs each embedding with its combined label. Paths are **relative to `data_base_dir`**:

```python
{
    "training": [
        {
            "image": "KiTS-000/image_emb.nii.gz",        # from Step 1
            "label": "KiTS-000/mask_combined_label.nii.gz",  # from Step 3
            "dim": [512, 512, 512],                        # resampled volume size — informational (but the label's actual voxel grid must be 4× the latent; see Step 4)
            "spacing": [1.0, 1.0, 1.0],                    # voxel spacing
            "top_region_index": [0, 1, 0, 0],              # ddpm-ct ONLY (omit for rflow-ct)
            "bottom_region_index": [0, 0, 0, 1],           # ddpm-ct ONLY (omit for rflow-ct)
            "modality": "ct",                              # required by Step 1's embedding script
            "fold": 0
        }
        // ...
    ]
}
```

> **`top_region_index` / `bottom_region_index` are only needed for `ddpm-ct`.** That variant's network (`config_network_ddpm.json`) sets `include_body_region: true`, so the loader feeds these body-region one-hots. `rflow-ct` (`config_network_rflow.json`) sets `include_body_region: false` and **ignores them** — you can leave them out entirely. The loader otherwise requires only `image`, `label`, and `spacing`; `dim` is informational (though the label file's actual voxel grid must be 4× the latent — see [Step 4](#step-4--put-the-combined-label-on-the-encoded-image-grid)).

<!-- -->

> **Fold split (read carefully — easy to get backwards):** an item is held out for **validation** when its `"fold"` **equals** `fold` in `config_maisi_controlnet_train*.json` (default `0`), and used for **training** otherwise. So if *every* item is `fold: 0` with the default config, your **training set is empty**. Spread items across folds (`0`, `1`, `2`, …) so the held-out fold gives a non-empty validation set and the rest train.

## Training-config touch-ups

**Emphasize a region of interest (optional).** `weighted_loss_label` lists the label indices whose voxels get an up-weighted L1 loss (by `weighted_loss`, only active when `weighted_loss > 1.0`) — use it to make the model focus on a small/hard ROI such as a tumor. It works for **any** label, new or existing; it is *not* tied to new classes. In `configs/config_maisi_controlnet_train*.json`:

```json
"weighted_loss_label": [129],   // any label index(es) you want emphasized, e.g. a tumor
"weighted_loss": 100            // weight multiplier; set to 1 to disable weighting entirely
```

**Name a new class (new-class case only).** If you assigned a new class to a `dummy` index in Step 3a, rename that entry in `configs/label_dict.json` so the vocabulary is self-documenting:

```diff
-    "dummy6": 129,
+    "kidney tumor": 129,
```

## Next: train the ControlNet

Once the files and JSON exist and the configs point at them, continue with the [`train_controlnet_image-from-mask`](../train_controlnet_image-from-mask/SKILL.md) skill — it covers the training configs, knobs (folds, `weighted_loss`, region-contrastive loss), launch, and outputs. (See also [docs/training.md → 3D ControlNet Training](../../../docs/training.md#3d-controlnet-training).) The generic trainer is retired; its former input shape was `-t config_network`, `-c config_maisi_controlnet_train`, `-e environment_maisi_controlnet_train`, `-g <GPU count>`.

## Gotchas checklist

- [ ] Embeddings made with **`autoencoder_v1.pt`** (not v2) for the CT ControlNet.
- [ ] **Body envelope (`200`) added** (retired `add_body_envelope` utility, git history) — NV-Segment never produces it, and the ControlNet needs it on every non-organ body voxel.
- [ ] **Combined label on the encoded-image grid** — resampled (nearest-neighbor) to the Step-1 resampled size = 4× the latent per axis, or training errors out on a shape mismatch.
- [ ] New classes remapped to **any unclaimed integer in `0–255`** (free ranges `133–199` / `201–255`, or a `dummy` slot like `129`); existing organs remapped to their real MAISI indices. Never reuse a claimed index, `0`, or `200`.
- [ ] `label_dict.json` has a named entry for any new index. (Optional: `weighted_loss_label` set if you want to emphasize an ROI such as a tumor.)
- [ ] Items spread across **multiple folds** so the held-out (validation) fold isn't the whole dataset.
- [ ] `modality` field present (needed by the Step-1 embedding script).
