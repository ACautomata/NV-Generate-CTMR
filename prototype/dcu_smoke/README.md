# 曙光 DCU 实机训练冒烟（wayfinder #15）

> 一次性冒烟资产。目标：实证 torch-dcu 环境、依赖、bf16、DDP(RCCL) 在**本仓真实训练管线**上可用——单卡 bf16 端到端跑通 1 epoch，并逐项过 #6 的 11 项实机验证清单。
> 走的是仓内脚本 `diff_model_create_training_data.py`（VAE 编码）→ `diff_model_train.py`（DM 训练），非自写 loop。环境/运行时验证，非配方调优。

## 集群布局（持久盘 `/root/private_data`）

| 路径 | 内容 |
|---|---|
| `nv-dcu-smoke/NV-Generate-CTMR/` | 本仓代码（含 bf16 patch） |
| `nv-dcu-smoke/NV-Generate-CTMR/models/` | `autoencoder_v1.pt` + `diff_unet_3d_rflow-mr-brain_v1.pt`（经 `HF_ENDPOINT=https://hf-mirror.com` 下载；huggingface.co 被代理挡） |
| `datasets/ASNR-MICCAI-BraTS2023/.../TrainingData` | 集群已有全量 GLI（1251 例），冒烟取前 N 例 |
| `nv-dcu-smoke/embeddings/`、`output/`、`models_trained/` | 冒烟产物（VAE embedding、推理输出、ckpt） |

数据只用 **t1n/t2w/t2f** 三模态（skull-stripped 索引 29/30/31，v1 DM 训练过的合法类嵌入行）；**跳过 t1c**（其 skull-stripped 索引 34 是 P1 才新增的、v1 未训）。

## bf16 patch（`scripts/diff_model_train.py`，本冒烟唯一源码改动）

`autocast("cuda")` 原本未指定 dtype → 默认 fp16 + GradScaler。patch 新增 `--amp_dtype {fp16,bf16}`（默认 `fp16`，行为不变）：

- `bf16`：`autocast("cuda", dtype=torch.bfloat16)`，且**跳过 GradScaler**(bf16 动态范围够，无需梯度缩放）——#10 锁定的 DCU 路径。
- `fp16`：维持原 fp16 + GradScaler，用于对照清单第 4 项（ROCm 上 fp16+GradScaler 有 NaN 报告）。

## 跑法

```bash
# 集群上（bashrc 已持久化双 source；脚本内亦幂等 source 一次）
bash prototype/dcu_smoke/run_smoke.sh 6 bf16     # 6 例 × 3 模态，bf16
bash prototype/dcu_smoke/run_smoke.sh 2 fp16     # 对照：fp16 + GradScaler
```

或分步：

```bash
python prototype/dcu_smoke/dcu_env_check.py            # 环境项 gate
python prototype/dcu_smoke/make_smoke_datalist.py \
    --data-base-dir /root/private_data/datasets/ASNR-MICCAI-BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
    --out prototype/dcu_smoke/dataset_dcu_smoke.json --n-cases 6
python -m scripts.diff_model_create_training_data -e prototype/dcu_smoke/environment_dcu_smoke.json \
    -c prototype/dcu_smoke/config_dcu_smoke.json -t configs/config_network_rflow.json -g 1
python prototype/dcu_smoke/write_emb_metadata.py --data-list prototype/dcu_smoke/dataset_dcu_smoke.json \
    --embedding-base-dir /root/private_data/nv-dcu-smoke/embeddings
python -m scripts.diff_model_train -e prototype/dcu_smoke/environment_dcu_smoke.json \
    -c prototype/dcu_smoke/config_dcu_smoke.json -t configs/config_network_rflow.json -g 1 --amp_dtype bf16
```

> **冒烟中发现的上游缺口**:`diff_model_train.py` 对每个 embedding 读一个伴侣 `<emb>.json`(键 `spacing`/`modality`，见 `prepare_data` 的 `_load_data_from_file`)，但本仓与上游 NVIDIA 仓的 `diff_model_create_training_data.py` 都**只写 `_emb.nii.gz`、不写该 json**。冒烟用 `write_emb_metadata.py` 补齐（spacing 读回 emb 自身 affine 的 pixdim;modality 取自数据 list)。spec 的全量数据管线需含这一步。

## 11 项实机验证清单 ↔ 覆盖方式

| # | 验证项 | 由什么覆盖 |
|---|---|---|
| 1 | device 命名 `"cuda"` | `dcu_env_check.py` |
| 2 | `backend="nccl"`→RCCL/HCCL | `dcu_env_check.py`（单卡）；**2 卡 all_reduce 补验** |
| 3 | SDPA 后端正确性 | `dcu_env_check.py`（有限输出即保底） |
| 4 | AMP：fp16+GradScaler 是否 NaN vs bf16 免 scaler | `run_smoke.sh` 两次（`bf16` / `fp16` 对照），看 loss 是否有限 |
| 5 | SyncBatchNorm 多卡 kernel | **2 卡 `diff_model_train.py` 跑几步**（8 卡可用则补） |
| 6 | 算子覆盖（3D conv/GN/SDPA/LogisticNormal） | 单卡端到端 1 epoch 跑通即证 |
| 7 | numpy 版本（防 2.x） | `dcu_env_check.py`（装后 gate） |
| 8 | 设备可见性变量 | `dcu_env_check.py`（报告 CUDA/HIP_VISIBLE_DEVICES） |
| 9 | CacheDataset RAM | 单卡跑通 + `cache_rate=0` 起步；观察 RAM |
| 10 | 显存容量 | 单卡跑通 + `hy-smi` 观察 VRAM |
| 11 | monai ≥1.5.2 | `dcu_env_check.py` |

完成标志：逐项有结论；单卡 bf16 端到端 1 epoch 跑通，或定位到具体阻塞点。

## 冒烟结果（2026-08-18，DTK26.04 / torch 2.9.0+das / 8×BW DCU，11/11 全绿）

| # | 验证项 | 结论 |
|---|---|---|
| 1 | device 命名 | ✅ `cuda.is_available()=True`,count=8,dev0="BW",hip=6.3.26093 |
| 2 | nccl→RCCL | ✅ `is_nccl_available=True`,默认后端 nccl;**2 卡 torchrun 跑通 1 epoch**,epoch 末 loss all_reduce 过 RCCL 正常 |
| 3 | SDPA | ✅ 前向有限;实测走 DCU flash-attn(cutlassfa) 接口;mem-efficient 未编译(math 保底,不影响) |
| 4 | AMP | ✅ **bf16(免 GradScaler)18/18 有限,均值 0.9742**;fp16+GradScaler 对照也 18/18 有限(均值 0.9664,冒烟尺度无 NaN);#10 锁定 bf16 依旧更稳 |
| 5 | SyncBN | ✅ 模型全 GroupNorm(0 处 BatchNorm),`convert_sync_batchnorm` 空操作;多卡路径由 2 卡跑通实证 |
| 6 | 算子覆盖 | ✅ 端到端(3D conv/GN/SDPA/LogisticNormal/RFlow)1 epoch 无错 |
| 7 | numpy | ✅ 1.26.4,补装 scikit-image/fire 及其纯 python 依赖(--no-deps)后 torch 完好 |
| 8 | 可见性变量 | ✅ CUDA/HIP_VISIBLE_DEVICES 均未设,count=8(全卡可见) |
| 9 | CacheDataset | ✅ cache_rate=0 跑通;全量数据缓存 RAM 需实机再评估(spec 注意项) |
| 10 | VRAM | ✅ 单卡峰值 **6134 MiB / 65520 MiB**(≈9.4%,256×256×128 batch1 bf16;512³ 估 39G 也能进 64G 卡) |
| 11 | monai | ✅ 1.6.0 + RFlowScheduler 可导入 |

**两个上游发现(与 DCU 无关，NVIDIA 卡同样存在)**：

1. **伴侣 json 缺口**：训练要读 `<emb>.json`(spacing/modality),但本仓与上游 create_training_data 都不写 → 冒烟用 `write_emb_metadata.py` 补;spec 全量数据管线需含此步。
2. **多卡 DDP 属性 bug**：DDP 包装后读 `include_top_region_index_input`/`num_class_embeds` 崩 → 已按 `.module` 既有模式修复(本分支);spec 若用多卡需带上。
