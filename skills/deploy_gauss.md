---
name: deploy_gauss
description: Deploy and run NV-Generate-CTMR experiments on the shared "gauss" GPU server — a centralized run root on the large scratch disk, a UV-managed environment, and a free-GPU sweep that claims every idle GPU for training or validation while leaving GPUs other people are using untouched. Covers connecting via the `gauss` SSH alias (host, user, and jump route live in the local SSH config, never in this repo), first-time setup (uv, venv, code, checkpoints, datasets), the run-directory layout, and launching long jobs under named tmux sessions with CUDA_VISIBLE_DEVICES locked to the swept GPUs. Trigger when the user mentions the gauss server, asks to deploy / set up / migrate the project to a GPU server, asks where runs, environments, checkpoints, or datasets should live there, wants training or validation launched on the server, or asks which GPUs are free and how to avoid disturbing other users' jobs.
---

# Deploy experiments on the gauss server

gauss is a **shared** GPU server: other people run jobs on it around the clock, and two resources are shared — GPUs and disk. Everything in this skill serves two invariants:

- **One run root** — every artifact this project produces on gauss (code, environment, checkpoints, datasets, run outputs) lives under one directory on the big scratch disk. `$HOME` stays empty beyond uv itself.
- **Free-GPU sweep** — every launch starts by sweeping for idle GPUs and claiming all of them. GPUs somebody else is using are never touched.

This skill owns the *where* and the *with which GPUs*; what to run is owned by the training and inference skills (see Related skills).

## Server facts

| Fact | Value |
| --- | --- |
| Connect | `ssh gauss` — the alias resolves host, user, and jump route from the local `~/.ssh/config`. Never copy those details into repo files. |
| GPUs | 4× RTX A6000 (48 GB), shared with other users |
| CPU / RAM | 80 cores / 503 GB — a generous MONAI `cache_rate` is affordable |
| Driver / CUDA | 575.57.08 / CUDA 12.9 — cu12x torch wheels |
| System python | 3.8 only — install a fresh interpreter with `uv` |
| Outbound net | pypi, GitHub, HuggingFace, astral.sh all reachable — clone, install, and download directly on the server |
| Disks | `/data72` (73 T) hosts the run root. `$HOME` is small and chronically near-full — a venv, model, or cache in `$HOME` fills it for everyone. |
| Dataset pool | `/data72/dataset` — shared, pre-populated with many datasets; the first place to look before downloading anything. |

## Run root layout

```text
RUN_ROOT=/data72/$USER/nv-ctmr
├── code/NV-Generate-CTMR/         # the clone; .venv lives here
├── models/                        # pretrained checkpoints — shared across runs
├── datasets/                      # training / inference data — mostly symlinks into the shared pool
├── cache/                         # HF_HOME and other tool caches
└── runs/<exp_name>-<yyyymmdd>/    # everything one experiment writes
    ├── configs/                   # snapshot of the -t/-c/-e JSONs actually used
    ├── logs/
    └── outputs/                   # tfevents, checkpoints, generated volumes
```

`models/` and `datasets/` are read-mostly assets every run points into. Each `runs/` folder is self-contained: the env-config copy inside redirects every output key (`model_dir`, `tfevent_path`, `output_dir`) into the run folder, so experiments never overwrite each other and [publish_experiment](publish_experiment.md) collects from one place.

## 1. Free-GPU sweep

Run on gauss before every launch — first setup and every later job alike:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
  | awk -F', ' '$2 < 1000 {print $1}' | paste -sd, -
```

A GPU is **free** when its memory used is under ~1 GiB — a job still warming up already holds memory, so any held memory means occupied. The sweep yields the **claim list**, e.g. `GPUS=1,2`.

- **Claim all free GPUs** — that is the point of the sweep; `-g` / `--nproc_per_node` equals the count.
- **Zero free GPUs** → report to the user and wait (re-sweep later); squeezing onto an occupied GPU is the one way this workflow must never fail.

**Done when**: a claim list exists, e.g. `GPUS=1,2`.

## 2. One-time setup

```bash
ssh gauss
RUN_ROOT=/data72/$USER/nv-ctmr
mkdir -p $RUN_ROOT/{code,models,datasets,cache,runs}
export HF_HOME=$RUN_ROOT/cache/huggingface    # every shell that downloads or runs needs this

# code
git clone https://github.com/ACautomata/NV-Generate-CTMR.git $RUN_ROOT/code/NV-Generate-CTMR

# uv (lands in ~/.local/bin — follow the installer's PATH hint)
curl -LsSf https://astral.sh/uv/install.sh | sh

# environment — run from the repo, uv finds .venv on its own
cd $RUN_ROOT/code/NV-Generate-CTMR
uv venv --python 3.11
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt
```

If a step fails on the network, fall back to fetching locally and `rsync`-ing up (the uv binary, the repo, or wheels) — direct access is the norm on gauss, so treat this as the exception.

**Done when**: `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` prints a cu12x version and `True`.

## 3. Checkpoints & datasets

- Checkpoints — follow [download-models](download-models.md) on gauss: `uv run python -m scripts.download_model_data --version <VARIANT> --root_dir $RUN_ROOT` lands weights in `$RUN_ROOT/models/` and auxiliary data in `$RUN_ROOT/datasets/`, exactly the paths the env configs expect. Gated repos need a token or prior license acceptance (see that skill).
- Datasets — **local-first: search `/data72/dataset` before downloading anything**; the shared pool already holds most public sets. Found → symlink it into `$RUN_ROOT/datasets/` (or point the env config straight at it). Missing → download into `$RUN_ROOT/datasets/`. Either way, the JSON data list and `data_base_dir` in the run's env-config copy (Step 4) point at the resolved location.

**Done when**: every path an env config expects — weights under `models/`, data under `datasets/` — exists on gauss.

## 4. Launch a run (training or validation)

Training (`scripts.diff_model_train`, `scripts.train_controlnet`) and validation/inference (`scripts.diff_model_infer`, `scripts.infer_image_from_mask_batch`) share one shape: `torchrun` + a `-g`/`--num_gpus` flag, work sharded across ranks. So one claim-and-launch recipe covers everything — swap the module and flags per the training/inference skills.

1. **New run folder + config snapshot** — copy the `-t`/`-c`/`-e` JSONs into `runs/<exp_name>-<yyyymmdd>/configs/`; in the env copy point every output key (`model_dir`, `tfevent_path`, `output_dir`) into the run folder and the `trained_*` / data keys at the shared `models/` / `datasets/`.
2. **Smoke first** — before committing every claimed GPU to a long DDP run, launch once on a single GPU with a throwaway config copy (tiny epoch/sample count) to prove imports, paths, and data resolve. A crash caught in smoke costs one GPU-minute; in a full-width launch it wastes everybody's GPUs.
3. **Re-sweep, then launch** — a GPU can flip state between the sweep and the launch, so re-run the sweep and use the fresh list. Pick a random high master port — a fixed port collides with other users' torchrun sessions.

```bash
GPUS=1,2                                     # fresh free-GPU sweep result
N=$(echo "$GPUS" | tr ',' '\n' | wc -l)
PORT=$(( 20000 + RANDOM % 20000 ))
RUN=runs/<exp_name>-<yyyymmdd>               # relative to RUN_ROOT

tmux new -s nv-<exp_name> -d
tmux send-keys -t nv-<exp_name> "cd $RUN_ROOT/code/NV-Generate-CTMR && \
  source .venv/bin/activate && export HF_HOME=$RUN_ROOT/cache/huggingface && \
  CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$N --nnodes=1 \
  --master_addr=localhost --master_port=$PORT \
  -m scripts.train_controlnet \
  -t ./configs/config_network_rflow.json \
  -c $RUN/configs/config_maisi_controlnet_train_rflow-ct.json \
  -e $RUN/configs/environment_maisi_controlnet_train_rflow-ct.json \
  -g $N 2>&1 | tee $RUN/logs/train.log" Enter
```

tmux keeps the job alive across disconnects; the session name `nv-<exp_name>` tells other users whose it is.

**Done when**: `tmux ls` shows the session, the log shows the DDP world size and the first step (or the first generated volume), and `nvidia-smi` shows this job's processes on exactly the claimed GPUs.

Validation/inference swaps the module (`scripts.diff_model_infer`, `scripts.infer_image_from_mask_batch`) and its flags per the inference skills — the sweep, tmux, random port, and log shape stay identical. On the 48 GB A6000, pick inference configs sized ≤ 32 GB (`config_infer_32g_*`).

## 5. Monitor and finish

- Watch: `tmux attach -t nv-<exp_name>` or `tail -f $RUN_ROOT/runs/<...>/logs/train.log`.
- When the run finishes, archive with [publish_experiment](publish_experiment.md) — the self-contained run folder is the bundle — then delete bulky intermediates so the next run has room.
- Manage only your own sessions: `tmux kill-session -t nv-<exp_name>`. Other users' processes and sessions are off-limits even when their GPUs look underused.

**Done when**: outputs are archived via publish_experiment and the run folder holds nothing bulky.

## Gotchas

- `$HOME` is shared and nearly full — venvs, models, datasets, caches, and outputs all live under the run root; only uv itself stays in `$HOME`.
- `HF_HOME` must point into the run root before any download — HuggingFace's default cache writes gigabytes into `$HOME`.
- Random high `--master_port` per launch; a fixed port collides with other users' torchrun jobs.
- Never `pip install --user`, and never write outside the run root — other users' `/data72/<user>/` trees are theirs.
- The sweep threshold errs toward caution: a GPU that looks idle but holds memory is somebody's job.
- First launch in a fresh environment is always a smoke launch (Step 4.2) — full-width DDP comes after the pipeline has proven itself once.
