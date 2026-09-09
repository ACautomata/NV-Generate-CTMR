---
name: publish_experiment
description: After any experiment finishes (VAE / diffusion / ControlNet training, inference, evaluation), archive its outputs to a GitHub Release instead of the repo. Covers collecting the artifact bundle (checkpoints, TensorBoard events, logs, metrics, generated volumes, one-off scripts), culling whole output sets down to a few typical example images, writing the release-notes Markdown, and `gh release create` with a draft-verify-publish loop. Trigger when an experiment has just finished, when the user asks to publish / release / share results, asks "where do the checkpoints / logs / sample images go", or asks how to showcase generated images.
---

# Publish an experiment to a GitHub Release

The release is the archive; the repo holds code. Experiment outputs — checkpoints, logs, metrics, generated volumes, scripts written only for the run — never enter git. One experiment → one release:

- **Release body** — the showcase Markdown: summary, config, metrics, typical examples.
- **Release assets** — checkpoints, raw records, experiment scripts, example PNGs.

Shared code is the exception: fixes to `scripts/*.py` are code changes and get committed. Scripts written *for this experiment* (notebooks, one-off preprocessing, plotting) ship in a tarball instead.

## 1. Collect the artifact bundle

Enumerate everything this experiment wrote. Map from the env config (`-e`) the run used:

| Env-config key | Holds |
|---|---|
| `model_dir` | checkpoints — `{exp_name}_best.pt`, `{exp_name}_current.pt` |
| `tfevent_path` | TensorBoard event files |
| `output_dir` | generated volumes (`*.nii.gz`) |
| `output_prefix` | filename prefix of the generated volumes |

Then sweep for artifacts that land outside those keys: training logs, eval outputs (`scripts/compute_fid_2-5d_ct.py`, `scripts/quality_check.py`), and every script written for this run.

**Done when**: a manifest lists every artifact with its path and size — nothing the experiment produced is unlisted.

## 2. Cull to typical examples

Stage the bundle **outside the worktree** (e.g. `../release-<tag>/`), so nothing is one `git add` away from a commit.

Pick 3–6 volumes that span the run's range — different modality / anatomy / condition / seed, and include a weak case if one exists. Per volume, render 2–3 orthogonal slices to PNG (`scripts/utils_plot.py::get_xyz_plot` builds an x/y/z montage), side by side with ground truth when one exists.

Whole output sets never go into a release: the notes show *typical*, the assets carry the data.

**Done when**: the staging folder holds one PNG per example, each traceable to a named source volume.

## 3. Write the release notes

One Markdown file in the staging folder:

1. What the experiment was — model, `generate_version` (`rflow-ct` / `rflow-mr` / `ddpm-ct` / `rflow-mr-brain`), launch command.
2. Config snapshot — the `-c` / `-e` values that matter (epochs, lr, fold, data list).
3. Metrics table — numbers from the raw records, nothing unbacked.
4. Typical examples — each PNG embedded by its **deterministic asset URL**, valid before upload:
   `https://github.com/ACautomata/NV-Generate-CTMR/releases/download/<tag>/<file>`
5. Asset manifest — what each uploaded file is (from the Step 1 manifest).

**Done when**: every image resolves to a release-asset URL and every claim traces to a record in the bundle.

## 4. Publish

```bash
tag="exp-<exp_name>-$(date +%Y%m%d)"
staging="../release-${tag}"
gh release create "$tag" --draft --title "<Title>" --notes-file "$staging/RELEASE_NOTES.md" \
    "$staging/<exp_name>_best.pt" \
    "$staging/records-${tag}.tar.gz" \
    "$staging/scripts-${tag}.tar.gz" \
    "$staging"/example-*.png
```

- Raw records and experiment scripts ship as tarballs: `records-<tag>.tar.gz` (logs, TB events, metrics), `scripts-<tag>.tar.gz`.
- **2 GiB per asset** — a checkpoint over the limit must be split (`split -b 1G`) and the parts attached; say so in the asset manifest.
- Draft first, then verify and publish:

```bash
gh release view "$tag"          # asset list complete, notes as intended
gh release edit "$tag" --draft=false
curl -fsIL "https://github.com/ACautomata/NV-Generate-CTMR/releases/download/${tag}/<one-example>.png"   # → HTTP 200
```

Draft asset URLs 404 even for maintainers — the URL check only works after publishing, which is why the asset list is verified while still a draft.

**Done when**: the release is public, `gh release view "$tag"` shows every asset in the manifest, and one embedded image URL returns 200.

## 5. Leave the repo clean

- The repo ships **no `.gitignore`** — create or extend one so the run's output dirs (`models/`, `outputs/`, `output/`, `datasets/`, `embeddings/`, plus any scratch dirs this run used) are ignored.
- `git status --porcelain` must list nothing from the experiment. Only intended code/doc changes get committed.

**Done when**: `git status --porcelain` is clean apart from intentional changes.

## Gotchas

- Generated (synthetic) volumes are always safe to show. Images derived from real patient data go in only when the dataset's license and anonymization allow it.
- Upload the assets and the notes in the same `gh release create` call — embedded image URLs resolve only once the release exists.
- `gh` infers the repo from `git remote -v`, same as for issues (see [issue-tracker](../docs/agents/issue-tracker.md)).
