# Training Guide

## Prerequisites

- GPU with sufficient VRAM (see [GPU memory requirements](#training-gpu-memory-usage) below)
- Training data prepared according to [data preparation](data.md)
- Model weights downloaded (see [setup guide](setup.md))

Training data preparation can be found in [../data/README.md](../data/README.md).

## 3D Autoencoder (VAE) Training

The VAE training loop lives in [`ctmr.application.vae_train`](../src/ctmr/application/vae_train.py) -- the single in-repo implementation of the MAISI autoencoder GAN loop (alternating generator/discriminator updates, three-phase LR warmup, AMP GradScaler pairs).

The information for the training hyperparameters and data processing parameters, like learning rate and patch size, are stored in [../configs/config_maisi_vae_train.json](../configs/config_maisi_vae_train.json). The provided configuration works for 16G V100 GPU. Please feel free to tune the parameters for your datasets and device.

### Retired VAE runner contract

The end-to-end VAE caller retired to git history with the scripts layer (issue #143), pending the `ctmr data` command family (ADR-0015 §3); the loop itself lives on in [`ctmr.application.vae_train`](../src/ctmr/application/vae_train.py) and its invocation shape is pinned by its tests. The retired caller wired: three config layers (network / train / environment) → data pipeline (`VAE_Transform` + `CacheDataset`) → the per-epoch loop → checkpoint publication + periodic validation.

Its input shape, for reference when the caller is rebuilt:

```text
network=rflow
-t configs/config_network_<network>.json
-c configs/config_maisi_vae_train.json
-e configs/environment_maisi_vae_train.json
--train-list train_ct.json --train-list train_mri.json
--val-list val_ct.json --val-list val_mri.json
```

Each `--train-list` / `--val-list` file is a JSON array of `{"image": <path>, "class": "ct"|"mri"}` entries (one file per modality). Every epoch publishes `autoencoder.pt` / `discriminator.pt` under `model_dir`, and the validation pass (every `val_interval` epochs) additionally saves the best-score epoch as `autoencoder_epoch<N>.pt`. TensorBoard/plot observers of the deleted tutorial notebook are not rebuilt -- epoch summaries go to stdout.

### Dataset Preprocessing Parameters

- `"random_aug"`: bool, whether to add random data augmentation for training data.
- `"spacing_type"`: choose from `"original"` (no resampling involved), `"fixed"` (all images resampled to same voxel size), and `"rand_zoom"` (images randomly zoomed, valid when `"random_aug"` is True).
- `"spacing"`: None or list of three floats. If `"spacing_type"` is `"fixed"`, all the images will be interpolated to the voxel size of `"spacing"`.
- `"select_channel"`: int, if multi-channel MRI, which channel it will select.

### Training Configuration Parameters

- `"batch_size"`: training batch size. Please consider increasing it if GPU memory is larger than 16G.
- `"patch_size"`: training patch size. For the released model, we first trained the autoencoder with small patch size [64,64,64], then continued training with patch size of [128,128,128].
- `"val_patch_size"`: Size of validation patches. If None, will use the whole volume for validation. If given, will central crop a patch for validation.
- `"val_sliding_window_patch_size"`: if the validation patch is too large, will use sliding window inference. Please consider increasing it if GPU memory is larger than 16G.
- `"val_batch_size"`: validation batch size.
- `"perceptual_weight"`: perceptual loss weight.
- `"kl_weight"`: KL loss weight, important hyper-parameter. If too large, decoder cannot recon good results from latent space. If too small, latent space will not be regularized enough for the diffusion model.
- `"adv_weight"`: adversarial loss weight.
- `"recon_loss"`: choose from 'l1' and 'l2'.
- `"val_interval"`: int, do validation every `"val_interval"` epochs.
- `"cache"`: float between 0 and 1, dataloader cache, choose small value if CPU memory is small.
- `"n_epochs"`: int, number of epochs to train. Please adjust it based on the size of your datasets. We used 280 epochs for the released model on 58k data.

## 3D Latent Diffusion Training

```bash
export NUM_GPUS_PER_NODE=8
network="rflow"
generate_version="rflow-ct"
torchrun \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --nnodes=1 \
    --master_addr=localhost --master_port=1234 \
    -m ctmr.infrastructure.maisi_engine.diff_model_train -t ./configs/config_network_${network}.json -c ./configs/config_maisi_diff_model_${generate_version}.json -e ./configs/environment_maisi_diff_model_${generate_version}.json -g ${NUM_GPUS_PER_NODE}
```

To run the diffusion model training script with MAISI Rectified flow for MRI, please run the code above with:

```bash
network="rflow"
generate_version="rflow-mr"
```

To run the diffusion model training script with MAISI DDPM for CT, please run the code above with:

```bash
network="ddpm"
generate_version="ddpm-ct"
```

## 3D ControlNet Training

We provide a [training config](../configs/config_maisi_controlnet_train.json) executing finetuning for pretrained ControlNet with a new class (i.e., Kidney Tumor).
When finetuning with other new class names, please update the `weighted_loss_label` in training config
and [label_dict.json](../configs/label_dict.json) accordingly. There are 8 dummy labels as deletable placeholders in default `label_dict.json` that can be used for finetuning. Users may apply any placeholder labels for fine-tuning purpose. If there are more than 8 new labels needed in finetuning, users can freely define numeric label indices less than 256. The current ControlNet implementation can support up to 256 labels (0~255).
Preprocessed dataset for ControlNet training and more details about data preparation can be found in the [README](../data/README.md).

### Training Configuration

The training was performed with the following:

- GPU: at least 60GB GPU memory for 512 x 512 x 512 volume
- Actual Model Input (the size of 3D image feature in latent space) for the latent diffusion model: 128 x 128 x 128 for 512 x 512 x 512 volume
- AMP: True

### Retired generic ControlNet trainer contract

> ⚠️ The upstream ControlNet trainer entry retired with the scripts layer (issue #143); git history is its reproduction anchor. The project's own ControlNet training is the mask-conditioned candidate chain below (`ctmr generate mask train`). Its former config shape was `-t config_network`, `-c config_maisi_controlnet_train`, `-e environment_maisi_controlnet_train`, `-g <GPU count>`.

To run the ControlNet model training script with MAISI Rectified flow for MRI, please run the code above with:

```bash
network="rflow"
generate_version="rflow-mr"
```

To run the ControlNet model training script with MAISI DDPM for CT, please run the code above with:

```bash
network="ddpm"
generate_version="ddpm-ct"
```

The retired generic training entry previously supported multi-GPU DDP. The project candidate chains derive `torchrun` themselves, so invoke `ctmr generate modality-label train` or `ctmr generate mask train` with the desired `-g` count rather than hand-writing a launcher command.

## Training GPU Memory Usage

The VAE is trained on patches and can be trained using a 16G GPU if the patch size is set to a small value, such as [64, 64, 64]. Users can adjust the patch size to fit the available GPU memory. For the released model, we initially trained the autoencoder on 16G V100 GPUs with a small patch size of [64, 64, 64], and then continued training on 32G V100 GPUs with a larger patch size of [128, 128, 128].

The DM and ControlNet are trained on whole images rather than patches. The GPU memory usage during training depends on the size of the input images. There is no big difference on memory usage between `maisi3d-ddpm` and `maisi3d-rflow`.

| image size | latent size | Peak Memory |
|---|:---|:-:|
| 256x256x128 | 4x64x64x32 | 5G |
| 256x256x256 | 4x64x64x64 | 8G |
| 512x512x128 | 4x128x128x32 | 12G |
| 512x512x256 | 4x128x128x64 | 21G |
| 512x512x512 | 4x128x128x128 | 39G |
| 512x512x768 | 4x128x128x192 | 58G |

## BraTS2023 P1 Fine-tuning (Project Recipe)

The project recipe (spec [issue #51](https://github.com/ACautomata/NV-Generate-CTMR/issues/51), execution decisions in [ADR-0005](../adr/0005-p1-candidate-training-execution.md)) continues the frozen `rflow-mr-brain v1` DM on BraTS2023 with a 1:1 MR-RATE replay mix. It does not use the upstream trainer directly; the pinned deltas live in `ctmr.application.generation.modality_label.train` (run via `ctmr generate modality-label train`):

- **Full-parameter DM continuation, VAE frozen**, `scale_factor` **reused from the base checkpoint** (the recomputed `1/std(z)` of the first batch is logged and asserted as a sanity check only);
- **Hyperparameters are frozen** in `configs/config_brats_p1_train.json`: `lr=2e-6`, `batch=1`, `cache_rate=0`, `n_epochs<=100`, L1 loss, Rectified Flow uniform timestep sampling (scale 1.4, `config_network_rflow.json`), PolynomialLR power 2.0, `augment_modality_label prob=0.1` (t1c token 34 included);
- **1:1 replay** — the training list is the concatenation of the #52 BraTS `p1_image_only.json` (7404 entries) and the MR-RATE replay cohort (prep tooling retired to git history in #143; cohort rules in ADR-0005); replay entries keep the original whole-brain tokens `mri_t1/mri_t2/mri_flair`;
- **bf16 autocast by default** (DCU), fp32 fallback via `--no_amp`, DDP via `torchrun` (RCCL);
- **Per-epoch checkpoints** `epoch_<N>.pt` (upstream key layout) feed the dev-eval sidecar and the phase-run contract selection.

### Dev light acceptance and early stopping

`ctmr generate modality-label dev-eval` (`ctmr.application.generation.modality_label.monitor`) runs beside the trainer on a reserved GPU. Every 5 epochs it generates the fixed 16-case dev cohort (4 modalities x fixed seeds, cfg=10, 30 steps), records the per-modality 2.5D RadImageNet FID trend against the dev real bank, and runs the frozen L2 instruments on the generated pseudo-four-modality volumes (WT/TC/ET volume medians + failure counts; trend only). The pre-recorded early-stop rule (patience 3 evals, min epoch 30, cap 100) halts the trainer through `<ckpt_dir>/.early_stop`; the candidate is the `argmin` mean-FID epoch. See ADR-0005 for the exact rule text.

### Launch (sugon DCU, P1)

```bash
ctmr generate modality-label train -e run/environment.json -c configs/config_brats_p1_train.json \
    -t configs/config_network_rflow.json --replay-list run/lists/p1_mrrate_replay.json -g 7
# torchrun spawn is derived by the launcher; the dev-eval sidecar runs separately:
ctmr generate modality-label dev-eval ...
```

Prerequisites (controlled storage only): the #52 phase lists/embeddings, the MR-RATE replay cohort (its prep/encode tooling retired to git history in #143; cohort rules in ADR-0005), the v1 base checkpoint, and the dev real feature bank (`ctmr generate modality-label dev-eval reference`).

## BraTS2023 P2 Mask→Image Candidate (ControlNet-only bypass)

The project recipe (spec [issue #51](https://github.com/ACautomata/NV-Generate-CTMR/issues/51) decision 7, execution decisions in [ADR-0007](../adr/0007-p2-mask-conditioned-candidate-training-execution.md)) trains a mask-conditioned ControlNet bypass hung off the **frozen P1-DM** (`dm_source.json` registered candidate, ADR-0006). It does not use the upstream `train_controlnet.py` loop; the pinned deltas live in the `ctmr.application.generation.mask.train` entry (`ctmr generate mask train`):

- **ControlNet-only** — DM and VAE untouched; the ControlNet is initialized from the frozen P1-DM encoder/mid (`copy_model_state`); only `controlnet.requires_grad=True`;
- **Hyperparameters are frozen** in `configs/config_brats_p2_train.json`: `lr=1e-5`, `batch=1`, `cache_rate=0`, `n_epochs<=100`, L1 loss, Rectified Flow uniform timestep sampling (scale 1.4), PolynomialLR power 2.0, `weighted_loss=100` on `[129,130,131]`, `use_region_contrasive_loss=off` (ADR-0007 guard raises on any deviation);
- **No MR-RATE replay** — pure BraTS; the #52 `p2_mask_cond.json` list (train fold=1, dev fold=0, one entry per (case, modality)) is split by `fold=0` so the trainer trains on the train side and the val split is *discarded* — never used to pick a checkpoint (spec §decision 7 forbids the old select-by-train-loss behaviour);
- **Condition vocabulary** — the `combined` mask (brain=22 union ∪ 1/2/3→129/130/131) is binarized to the 8-bit ControlNet condition; `weighted_loss_label=[129,130,131]` weights the tumor subregions;
- **bf16 autocast by default** (DCU), fp32 fallback via `--no_amp`, DDP via `torchrun` (RCCL);
- **Per-epoch checkpoints** `epoch_<N>.pt` (`controlnet_state_dict` + `scale_factor`) feed the dev-eval sidecar and the phase-run contract selection.

### Dev light acceptance, round-trip Dice and early stopping

`ctmr generate mask dev-eval watch` (`ctmr.application.generation.mask.monitor`) runs beside the trainer on a reserved GPU. Every 5 epochs it generates the fixed 16-case dev cohort (4 modalities × fixed seed, cfg=10, 30 steps) **from the case's combined condition mask**, records the per-modality 2.5D RadImageNet FID trend against the dev real bank, runs the frozen L2 instruments on the generated four-modality volumes (WT/TC/ET volume medians + failure counts; trend only), and computes the **P2 condition round-trip Dice trend** (instrument-predicted mask vs the combined condition, nearest-neighbour aligned + remapped 0/1/2/3).
The pre-recorded early-stop rule (patience 3 evals, min epoch 30, cap 100) halts the trainer through `<ckpt_dir>/.early_stop`; the candidate is the `argmin` mean-FID epoch. See ADR-0007 for the rule text and the round-trip Dice semantics.

### Launch (sugon DCU, P2)

```bash
ctmr generate mask train -e run/environment.json -c configs/config_brats_p2_train.json \
    -t configs/config_network_rflow.json -g 7
# torchrun spawn is derived by the launcher; the dev-eval sidecar runs separately:
ctmr generate mask dev-eval watch ...
```

### Run-contract wiring (P2)

The phase-run contract (`ctmr accept contract`, module `ctmr.application.acceptance.contract`) already carries P2 init/select/attach/verify/conclude and the `undecided` handling. Two prerequisites gate a P2 run on the cluster:

1. **DM source registered (hard gate)** — P2 `init` requires `--upstream-run` pointing at a *frozen and registered* P1 candidate (`DmSourceLedger.check_upstream` matches run_id + checkpoint sha256). Until the P1 run has its L1/L2/L3 reports attached and passes `conclude` (#58), `dm_source.json` does not exist and P2 `init` is rejected with `"no P1 candidate has passed final acceptance yet; ... conclude a passing P1 run first"`. This is not P2 code; it is the #58 conclusion execution that must complete first.
2. **Init invocation** — `init --phase P2 --record-root DIR --manifest phase_manifest.json --config train=configs/config_brats_p2_train.json --config network=configs/config_network_rflow.json --data-list train=lists/p2_mask_cond.json --upstream-run records/runs/<p1-run-id>/run.json --platform-json run/environment_brats_p2_train.json` (no `--base-ckpt`; a replay list is rejected). Note `environment_brats_p2_train.json` (paths) embeds as `platform`; `p2_mask_cond.json` is the single train+dev list, split internally by `fold=1/fold=0` (the contract's holdout guard passes since #52 includes no holdout entries).

`select` (dev-only evidence), `attach` (l1/l2/l3; L2 report must carry P2 round-trip + `undecided` allowed), then `conclude` (non-compensatory L1∧L2∧L3) follow the frozen-candidate protocol identically to P1.
