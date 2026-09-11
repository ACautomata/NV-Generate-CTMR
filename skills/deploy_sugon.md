---
name: deploy_sugon
description: Deploy NV-Generate-CTMR to the 中科曙光/SothisAI DCU cluster (Sugon/曙光) and run training there — rsync the code up, install dependencies without touching the preinstalled DCU torch (PyPI wheels are CUDA builds), put datasets on the team-shared group_data disk and checkpoints/run outputs on the private persistent disk, sweep hy-smi for clean cards, smoke-test, then launch torchrun under tmux. Builds on the user-level sugon-bootstrap skill, which owns cluster access, the dual-source environment, and filesystem persistence — connection and DCU setup questions go there, not here. Trigger when the user wants this project deployed to 曙光/Sugon/SothisAI/DCU, asks to launch, monitor, or wrap up a training run on the Sugon cluster, or asks where code, checkpoints, datasets, or run outputs should live on Sugon.
---

# Deploy training on the Sugon (曙光) DCU cluster

This skill owns the project's *what* and *where* on Sugon: how the code gets there, how dependencies are laid down without breaking the DCU stack, where every artifact lives, and how a training run is launched and wrapped up. The cluster layer — SSH connection, the 双 source environment, filesystem persistence, DCU operator wheels — is owned by the user-level **sugon-bootstrap** skill: this document assumes it has already been run, and never repeats it. What to run (data formats, config choice, flag semantics) is owned by the training skills (see Related skills).

Two invariants serve everything below:

- **Disk per asset** — every artifact lands on the disk that matches its nature: the volatile system disk holds only what rsync restores (code, pip packages); the team-shared `group_data` holds the public, shareable bulk (datasets); the private `private_data` holds everything experimental (checkpoints, logs, run outputs). An instance reset then costs one rsync and one dependency reinstall — never data.
- **DCU-first** — every software decision assumes DCU, not CUDA: the preinstalled DCU torch is never replaced by a PyPI wheel, monitoring is `hy-smi`, and card binding goes through `CUDA_VISIBLE_DEVICES` (the DCU torch's HIP layer honors it).

## Prerequisite — bootstrap ready

One line proves the cluster layer sugon-bootstrap owns is all green:

```bash
ssh sugon 'echo "proxy=${https_proxy:-UNSET}"; command -v hy-smi >/dev/null && echo hy-smi-OK || echo hy-smi-MISSING; [ -f /opt/dtk/env.sh ] && echo DTK_NODE || echo NO_DTK; [ -w /root/private_data ] && echo PRIVATE_OK || echo PRIVATE_MISSING; [ -w /root/group_data ] && echo GROUP_OK || echo GROUP_MISSING'
```

Expect a live proxy URL, `hy-smi-OK`, `DTK_NODE`, `PRIVATE_OK`, `GROUP_OK`. Anything else → run sugon-bootstrap first; do not patch the cluster layer from here.

**Done when**: all four checks above are green.

## Server facts

| Fact | Value |
| --- | --- |
| Connect | `ssh sugon` — alias owned by sugon-bootstrap; the port changes whenever the instance is reassigned |
| Accelerators | Hygon DCUs on the DTK stack — count and occupancy come from `hy-smi` at launch time, never cached here |
| Preinstalled stack | System python `/usr/local/bin/python` (3.11) ships the DCU torch (`2.9.0+das…dtk2604`) — no conda, no uv, and no torch install of any kind |
| Outbound net | Proxy-only — pip/curl/git work only with the platform proxy live (the 双 source contract) |
| Disks | `/` (system): volatile, wiped on instance reset · `/root/private_data`: persistent, private · `/root/group_data`: persistent, team-shared, the roomy one — see sugon-bootstrap §3 for the full mount table |
| Shared pools | `/root/public_data` (platform, **read-only**, pretrained DL assets) — search before downloading; `/root/group_data` is writable and holds this project's datasets |

## Run-root layout

```text
CODE_ROOT=/root/nv-ctmr                     # system disk — volatile, rsync-restorable
GROUP_ROOT=/root/group_data/nv-ctmr         # persistent, team-shared, roomy — the public bulk
└── datasets/                               # training data, visible and reusable by the whole team
RUN_ROOT=/root/private_data/nv-ctmr         # persistent, private — everything experimental
├── models/                                 # pretrained checkpoints
├── cache/                                  # HF_HOME and other tool caches
└── runs/<exp_name>-<yyyymmdd>/             # everything one experiment writes
    ├── configs/                            # snapshot of the -t/-c/-e JSONs, outputs redirected here
    ├── logs/
    └── outputs/                            # checkpoints, tfevents, generated volumes
```

Code on the volatile disk is deliberate: code is always one rsync away, and keeping it off the persistent disks saves quota. Datasets are the public bulk — big and shareable, so they live in the team's roomy `group_data`. Checkpoints, logs, and run outputs are the experiment's private record and never leave `private_data`. The post-reset recovery order is in Gotchas.

## 1. Code up

```bash
rsync -az --delete --exclude .git --exclude .venv --exclude '__pycache__' \
  ./ sugon:/root/nv-ctmr/
```

Direct ssh transfer — no proxy involved. A git clone through the platform proxy is the fallback when rsync is unavailable, not the norm.

**Done when**: `ssh sugon 'ls /root/nv-ctmr/scripts/train_controlnet.py'` prints the path.

## 2. Dependencies against the DCU torch

The system python already carries the DCU torch; the entire point of this step is to lay the project's requirements down **without disturbing it**:

```bash
ssh sugon 'cd /root/nv-ctmr && pip install $(grep -viE "^torch" requirements.txt)'
```

The grep is the step: `requirements.txt` opens with `torch>=2.1.0`, and an unfiltered install has pip pull the CUDA wheel from PyPI over the DCU one — the stack breaks on the spot.

Self-check immediately, and again after **every** later pip operation:

```bash
ssh sugon 'python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"'
```

- torch prints `2.9.0+das…dtk…` and numpy is **1.x** → ready.
- numpy reads **2.x** (common — new wheels demand numpy≥2) → `pip install numpy==1.26.4`, then re-check. Mechanism: sugon-bootstrap's `references/dcu-pitfalls.md`.
- torch is anything but a `das` build → it got clobbered; reinstall the DCU torch per sugon-bootstrap's troubleshooting table.

Training operators (flash_attn / triton) are checked with sugon-bootstrap §4's `scripts/ensure_dcu_ops.sh` — the system disk is volatile, so an instance reset can take them away.

**Done when**: the self-check prints a DCU torch version with numpy 1.x, and `ssh sugon 'python -c "import monai"'` exits clean.

## 3. Checkpoints & datasets → the persistent disk

- **Pretrained weights** — search `/root/public_data` first; the platform pool already carries many pretrained DL assets (MAISI VAE among them). Found → point the env config straight at it (read-only is fine) or symlink/copy into `$RUN_ROOT/models/`. Missing → follow [download-models](download-models.md) on Sugon with `HF_HOME=$RUN_ROOT/cache/huggingface` set first — otherwise the HF cache lands on the volatile disk — or download locally and rsync up.
- **Datasets** — the public, shareable bulk: rsync them into `$GROUP_ROOT/datasets/`, where the team's roomy storage makes them reusable across members and instances.

**Done when**: every path the training env config expects — weights under `models/` (private), data under `datasets/` (group) — exists on its persistent disk.

## 4. Launch training

Three sub-steps: sweep → smoke → re-sweep + launch. Which module to run and with which flags is owned by the training skills; this step owns *on which cards, in what manner*.

**4.1 Clean-card sweep.** The instance is an exclusive root instance — nobody else's jobs appear, so the sweep checks for leftovers, not neighbors:

```bash
ssh sugon 'hy-smi'
```

A card with ~0 memory used is clean; any held memory → `ps -ef | grep python` to identify the leftover (a crashed run's tail), clear it only once confirmed to be yours. The sweep yields the **claim list**, e.g. `CARDS=0,1`.

**4.2 Smoke first.** Before committing every card to a long DDP run, launch once on a single card with a throwaway config copy (tiny epoch/sample count) to prove imports, paths, and data resolve. This environment is hand-assembled — first-shot failures are expected, and a crash the smoke catches costs a minute while a full-width one wastes every card.

**4.3 Re-sweep, then launch.** Card state can shift between the sweep and the launch (your own smoke tail, a leftover), so re-run 4.1 and use the fresh list.

1. **Run folder + config snapshot** — create `$RUN_ROOT/runs/<exp_name>-<yyyymmdd>/{configs,logs,outputs}`; copy the `-t`/`-c`/`-e` JSONs into `configs/`; in the env copy, point the three output keys (`model_dir`, `tfevent_path`, `output_dir`) into the run folder on the private disk, the `trained_*` keys at the shared `models/`, and the data keys at `$GROUP_ROOT/datasets/`.
2. **tmux + torchrun**:

```bash
CARDS=0,1                                     # fresh clean-card sweep result
N=$(echo "$CARDS" | tr ',' '\n' | wc -l)
PORT=$(( 20000 + RANDOM % 20000 ))            # random per launch — fixed ports collide with TIME_WAIT after a crashed torchrun
RUN=$RUN_ROOT/runs/<exp_name>-<yyyymmdd>

tmux new -s nv-<exp_name> -d
tmux send-keys -t nv-<exp_name> "cd /root/nv-ctmr && \
  CUDA_VISIBLE_DEVICES=$CARDS torchrun --nproc_per_node=$N --nnodes=1 \
  --master_addr=localhost --master_port=$PORT \
  -m scripts.train_controlnet \
  -t ./configs/config_network_rflow.json \
  -c $RUN/configs/config_maisi_controlnet_train_rflow-ct.json \
  -e $RUN/configs/environment_maisi_controlnet_train_rflow-ct.json \
  -g $N 2>&1 | tee $RUN/logs/train.log" Enter
```

tmux keeps the job alive across disconnects; the session name `nv-<exp_name>` reads at a glance whose it is.

**Done when**: `tmux ls` shows the session, the log prints the DDP world size and the first step, and `hy-smi` shows this job's processes on exactly the claimed cards.

## 5. Monitor and finish

- Watch: `tmux attach -t nv-<exp_name>` or `tail -f $RUN_ROOT/runs/<...>/logs/train.log`; card occupancy via `hy-smi`.
- When the run finishes, **rsync the run folder back to the local machine** (`rsync -az sugon:$RUN_ROOT/runs/<...> ./local_runs/`), then archive with [publish_experiment](publish_experiment.md) locally — the Sugon instance holds no repo push credentials, so publishing happens at home.
- Once archived, clear the bulky files on the server (persistent storage has quota) and keep only the small config/log remnants.
- Manage only your own sessions: `tmux kill-session -t nv-<exp_name>`.

**Done when**: outputs are archived via publish_experiment and the server-side run folder holds nothing bulky.

## Gotchas

- **Output keys on the volatile disk mean checkpoints that never come back.** In the env-config copy, `model_dir` / `tfevent_path` / `output_dir` always point into `$RUN_ROOT/runs/...` on the persistent disk.
- Set `HF_HOME=$RUN_ROOT/cache/huggingface` before any HuggingFace download — the default cache writes into `$HOME` on the volatile disk.
- **Post-reset recovery order**: ① the instance port may have changed — update the `sugon` ssh alias per sugon-bootstrap; ② re-run the bashrc dual-source injection (the bashrc lives on the volatile disk); ③ rsync the code; ④ reinstall dependencies + self-check (Step 2); ⑤ datasets (group disk) and checkpoints/logs/run outputs (private disk) are persistent — untouched.
- Saving an image freezes the environment (code + deps), but persistent-disk assets never rely on it — a saved image is convenience, not backup.
- Random `--master_port` per launch — a fixed port collides with TIME_WAIT after any crashed torchrun.

## Related skills

- [sugon-bootstrap](../../../.claude/skills/sugon-bootstrap/SKILL.md) (user-level) — the cluster layer: connection, 双 source, mount persistence, DCU operator wheels, troubleshooting table.
- [train_controlnet_image-from-mask](train_controlnet_image-from-mask.md) — the training *what*: data format, config selection, flag semantics.
- [download-models](download-models.md) — how pretrained weights are fetched.
- [publish_experiment](publish_experiment.md) — post-run archiving.
- [deploy_gauss](deploy_gauss.md) — the parallel flow on the NVIDIA server; the sweep/tmux/Done-when shape matches, the software stack and disk constraints do not.
