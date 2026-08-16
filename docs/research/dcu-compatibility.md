# 曙光 DCU 兼容性调研（issue #6）

> 隶属地图票 [#2](https://github.com/ACautomata/NV-Generate-CTMR/issues/2)。
> 目标：判断本仓库训练管线（P1 微调 DM / P2 微调 ControlNet / 数据预备 VAE 编码）在**中科曙光 DCU（ROCm 系 DTK 生态）**上的兼容性风险与所需补丁点。
> 方法：仓库代码静态分析 + 公开文档调研。**本票不做实机操作**；不确定项归入「需实机验证」清单（§6），实机冒烟属后续执行票。
> 证据标注：`文件:行号` 为本仓库代码；`URL` 为外部公开来源；`[skill]` 为本机 `sugon-bootstrap` skill 的集群实测经验（一手，可信）；`[推断]` 为基于 ROCm 类比的推论，需实机确认。

---

## 0. 结论速览

**一句话**：本仓库训练管线对 DCU 的「源码级」兼容性很好——因为 torch-dcu（ROCm 系）保留 `"cuda"` 设备命名空间，绝大多数 `torch.cuda.*` / `autocast("cuda")` / `GradScaler("cuda")` 硬编码**无需改源码即可运行**；真正的风险集中在三处「环境/运行时」而非「代码」：

1. **依赖安装机制**（最高优先级、最容易翻车）：torch/torchvision 必须装 **sourcefind 的 DCU 构建**（不能 PyPI），且所有 `pip install` 必须 `--no-deps` + **numpy 钉 1.x（1.26.4）**——否则 numpy 被升 2.x 会当场顶坏按 numpy 1.x 编的 DCU torch。`[skill]`
2. **AMP 在 DCU 的可用性**：`torch.amp.GradScaler("cuda")` + `autocast("cuda")` 在 ROCm 系 torch 上支持但历史有坑；**P2 ControlNet 训练无条件全程用 AMP**（`train_controlnet.py:394, 477`），无 `amp` 开关，故 GradScaler 必须在 DCU 上可用，否则需打补丁。**需实机验证**。
3. **分布式后端**：`dist.init_process_group(backend="nccl")` 在 DCU 上**保持写 `"nccl"` 即可**——PyTorch 默认把 cuda 设备映射到 nccl 后端，底层自动落到海光集合通信库（ROCm 生态名 RCCL / 海光品牌名 HCCL，本质同一库）。**不要改成 `backend="hccl"`**（那是华为昇腾 torch_npu / Intel Gaudi 的写法，不适用海光）。仅需实机一行确认 `dist.is_nccl_available()` 为真。**需实机验证**（低风险）。

**无需源码补丁即可跑通的可能性高**；需要的「补丁」更多是**环境配置 + 可选的稳妥化开关**（如给 ControlNet 加 `amp` 开关、必要时把 flash attention 回退到 math 后端）。

---

## 1. 核心判断：torch-dcu 保留 `"cuda"` 设备命名空间

这是整份调研的地基。PyTorch 的 ROCm 构建（torch-rocm）为保持源码兼容，**沿用 `"cuda"` 作为设备名**：`torch.device("cuda")`、`torch.cuda.is_available()`、`torch.cuda.set_device()`、`tensor.cuda()` 等 API 在 ROCm 上全部有效，内部映射到 HIP 设备。曙光 DCU 的 torch（torch-dcu，sourcefind 上的 `torch 2.9.0+das.opt1.dtk2604`）是基于同一 HIP/ROCm 软件栈的 fork，遵循同一约定 `[skill][推断]`。

**推论**：本仓库遍布各脚本的 `"cuda"` 硬编码（见 §2）在 DCU 上**语法层面全部可直接运行**，不需要把 `"cuda"` 改成 `"dcu"` 之类的补丁。需要改的只是**运行环境**（装对 DCU 版 torch）与**个别运行时行为**（AMP、后端、flash kernel）的确认。

**公开文档佐证（ROCm 侧，高置信）**：Kubeflow 的 PyTorch-on-ROCm 指南与 AMD 官方博客均直接用 `torch.cuda.set_device(local_rank)`/`torch.cuda.device_count()` 操作 AMD GPU；ROCm 仓库的 feature request [ROCm/ROCm#4231](https://github.com/ROCm/ROCm/issues/4231) 正因为「ROCm 版 torch 覆盖了 cuda 命名、没有独立的 `torch.device("rocm")`」而请求显式 rocm API——**反证现状就是沿用 `"cuda"` 命名**。可用 `torch.version.hip` 区分真 CUDA 与 ROCm（ROCm 构建该值非 None）。

> ✅ 置信度：ROCm 保留 `"cuda"` 命名是 PyTorch 官方长期行为（高置信，上述公开来源证实）；**海光 DCU 侧亦有多个实操指南一致确认**——torch-dcu 走 HIP 翻译层，代码写 `.to("cuda")`/`torch.cuda.is_available()` 即可，底层自动调度到 DCU，且 `torch.version.hip` 非空（可据此判别 ROCm 系而非 CUDA 系）。故「DCU 沿用 `"cuda"`」由「推断」升级为「公开文档证实（海光指南）」。仍列入实机验证清单第 1 条做终验。

---

## 2. CUDA 依赖点逐项盘点（训练管线）

下表逐项列出 issue #6 点名的依赖点及盘点中发现的其余点。**判断列**：✅=可直接用；⚠️=语法可用但运行时行为需实机确认；🟡=可用但建议小改/注意；❌=需补丁。

### 2.1 主训练脚本 `scripts/diff_model_train.py`（P1 微调 DM 用）

| 行号 | 依赖点 | DCU 判断 | 补丁建议 |
|---|---|---|---|
| `diff_model_train.py:29` | `from torch.amp import GradScaler, autocast` | ✅ 纯 import | — |
| `diff_model_train.py:295` | `with autocast("cuda", enabled=amp):` | ⚠️ autocast 在 ROCm/DCU 支持（沿用 `"cuda"` 名），但未指定 `dtype` → 默认 **fp16**；ROCm 上 fp16 不如 bf16 稳 | 建议评估 `dtype=torch.bfloat16`（见 §2.3 注） |
| `diff_model_train.py:477` | `scaler = GradScaler("cuda")` | ⚠️ ROCm 支持 GradScaler（device-agnostic），但有 fp16 NaN 报告 [ROCm/ROCm#6351](https://github.com/ROCm/ROCm/issues/6351)；**本脚本有 `amp` 开关，amp=False 时 scaler 创建但不被调用**（`346-348` 被 `if amp:` 门控） | 若 fp16+GradScaler 在 DCU 出 NaN：改 bf16（免 scaler）或置 `amp=False` 走 fp32 兜底 |
| `diff_model_train.py:479` | `torch.set_float32_matmul_precision("highest")` | ✅ 设备无关设置；`"highest"`=纯 fp32（禁 TF32），最安全。注：ROCm 本就不支持 `matmul.allow_tf32`，`"highest"` 正好规避 | — |
| `diff_model_train.py:155` | `torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)` | ⚠️ SyncBatchNorm 在 ROCm 支持；DCU kernel 需确认 | 单卡可去掉；多卡需确认 SyncBN 前向/反向 kernel |
| `diff_model_train.py:158` | `DistributedDataParallel(unet, device_ids=[device], find_unused_parameters=True)` | ✅ DDP 本体设备无关；后端保持 `"nccl"`（§2.2） | — |
| `diff_model_train.py:502` | `if torch.cuda.device_count() == 1 or local_rank == 0:` | ✅ `device_count()` 在 ROCm/DCU 返回 HIP 设备数 | — |
| `diff_model_train.py:137` | `monai.data.CacheDataset(..., num_workers=num_workers)` | ✅ 纯 python，CPU 侧 RAM 缓存，设备无关 | — |
| `diff_model_train.py:459` | `partition_dataset(...)` | ✅ 纯 python MONAI | — |
| `diff_model_train.py:327-329` | rectified flow 目标 `model_gt = images - noise` | ✅ 纯 tensor 运算 | — |

> 注：`diff_model_train.py` 的 `amp` 是命令行参数（默认 True），同时门控 `autocast(enabled=amp)` 与 `scaler.*` 三段（`295, 346-348`）。**这给 P1 留了 fp32 兜底开关**，是 ControlNet 没有的。

### 2.2 环境/启动封装 `scripts/diff_model_setting.py`

| 行号 | 依赖点 | DCU 判断 | 补丁建议 |
|---|---|---|---|
| `diff_model_setting.py:86` | `if torch.cuda.is_available() and num_gpus > 1:` | ✅ ROCm/DCU 下 `is_available()` 为 True | — |
| `diff_model_setting.py:87` | `dist.init_process_group(backend="nccl", init_method="env://")` | ✅ **保持 `"nccl"` 即可**：PyTorch 默认 cuda→nccl 映射，底层自动落到海光 RCCL/HCCL；海光 DCU 示例仓库（FlyAIBox/dcu-in-action）分布式即标准 `torchrun`+`"nccl"` | 无需改；**勿改 `"hccl"`**（昇腾/Gaudi 写法）；实机 `dist.is_nccl_available()` 终验（§6-2） |
| `diff_model_setting.py:93-94` | `torch.device("cuda", local_rank)`; `torch.cuda.set_device(device)` | ✅ `"cuda"` 命名空间沿用 | — |
| `diff_model_setting.py:98-119` | `run_torchrun()` 用 `subprocess` 起 `torchrun`，设 `OMP_NUM_THREADS=1` | ✅ torchrun 是 torch 自带 launcher，DCU 同样可用 | — |

### 2.3 ControlNet 训练 `scripts/train_controlnet.py`（P2 用）

| 行号 | 依赖点 | DCU 判断 | 补丁建议 |
|---|---|---|---|
| `train_controlnet.py:28` | `from torch.amp import GradScaler, autocast` | ✅ | — |
| `train_controlnet.py:359` | `scaler = GradScaler("cuda")` | ⚠️ 同 §2.1，但**此处无 `amp` 开关** | **若 DCU 上 GradScaler 不可用，需打补丁**（见下） |
| `train_controlnet.py:394` | `with autocast("cuda", enabled=True):` | ⚠️ **`enabled=True` 写死**，P2 全程 AMP | 同上 |
| `train_controlnet.py:477,483-485` | `scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()` | ⚠️ **无条件执行**（不像 P1 有 `if amp:` 门控） | 同上 |
| `train_controlnet.py:247-248` | `rank=int(os.environ["LOCAL_RANK"])`; `setup_ddp(rank, world_size)` | ✅ 标准 env 约定 | — |
| `train_controlnet.py:255-256` | `torch.cuda.set_device(device)`; `torch.cuda.device_count()` | ✅ | — |
| `train_controlnet.py:323` | `DDP(controlnet, device_ids=[device], output_device=rank, find_unused_parameters=True)` | ✅ DDP 本体设备无关；后端保持 `"nccl"`（§2.2） | — |
| `train_controlnet.py:535` | `torch.cuda.empty_cache()` | ✅ ROCm 支持 | — |
| `train_controlnet.py:32` | `from .augmentation import remove_tumors` | ✅ `remove_tumors` 本体**不含** `.cuda()`（见 §2.6） | — |

> **P2 的 AMP 补丁点（唯一可能需要动训练源码的地方）**：`train_controlnet.py` 全程 AMP 且无开关。若实机发现 GradScaler 在 DCU 上行为异常，最小补丁是仿照 `diff_model_train.py` 给 ControlNet 加一个 `amp` 开关，把 `394` 的 `enabled=True` 与 `477/483-485` 的 scaler 调用改为按 `amp` 门控，提供 fp32 兜底。**仅当实机验证 AMP 不可用时才需要。**
<!-- -->
> **AMP 数据类型注意（公开证据）**：本仓库 `autocast("cuda")` 未指定 `dtype`，默认 **fp16**，且配 `GradScaler`。ROCm 上有报告 [ROCm/ROCm#6351](https://github.com/ROCm/ROCm/issues/6351) 显示 GradScaler 在 ROCm 上训练几步后可能产出 **NaN loss**（fp16）；AMD 官方建议 **bf16 更稳且 bf16 不需要 GradScaler**（bf16 动态范围够大，无梯度下溢）。→ DCU 上更稳的 AMP 路径可能是 `autocast("cuda", dtype=torch.bfloat16)` 并停用 GradScaler；这是一个**可选的稳妥化改造**，列入实机验证（§6-4）一并评估。

### 2.4 分布式工具 `scripts/utils.py` + 评估 `scripts/compute_fid_2-5d_ct.py`

| 行号 | 依赖点 | DCU 判断 | 补丁建议 |
|---|---|---|---|
| `utils.py:202` | `dist.init_process_group(backend="nccl", ..., timeout=36000s)` | ✅ 同 §2.2-87，保持 `"nccl"` | — |
| `utils.py:204` | `device = torch.device(f"cuda:{rank}")` | ✅ | — |
| `compute_fid_2-5d_ct.py:461` | `dist.init_process_group(backend="nccl", ..., timeout=7200s)` | ⚠️ 评估侧同 | — |
| `compute_fid_2-5d_ct.py:465-466` | `torch.device("cuda", local_rank)`; `torch.cuda.set_device` | ✅ | — |
| `compute_fid_2-5d_ct.py:503` | `torchvision.models.squeezenet1_1(pretrained=True)` | 🟡 **torchvision 不在 requirements.txt**；需装 sourcefind 的 DCU 版 torchvision（`0.24.0+das...`）；`pretrained=True` 需联网（集群走代理） | 把 torchvision 列入 DCU 环境清单；仅 FID 评估用，不阻塞训练 |
| `compute_fid_2-5d_ct.py:21` | docstring `export CUDA_VISIBLE_DEVICES=0..6` | 🟡 DCU 上设备可见性环境变量可能是 `HIP_VISIBLE_DEVICES`/DCU 专有变量 | 实机确认可见性变量名（§6） |

### 2.5 数据预备 `scripts/diff_model_create_training_data.py`（VAE 编码，产 `*_emb.nii.gz`）

| 行号 | 依赖点 | DCU 判断 | 补丁建议 |
|---|---|---|---|
| `diff_model_create_training_data.py:169` | `with torch.amp.autocast("cuda"):` | ⚠️ autocast 同 §2.1；**此处无 enabled 开关、恒开** | 若 AMP 在 DCU 有问题，需小改加开关或改 fp32 |
| `diff_model_create_training_data.py:211` | `initialize_distributed(num_gpus)` | ⚠️ 经 §2.2 的 NCCL 后端 | — |
| `diff_model_create_training_data.py:262` | `dist.destroy_process_group()` | ✅ | — |

### 2.6 推理/采样脚本（非训练，顺带盘点）

| 文件:行号 | 依赖点 | DCU 判断 |
|---|---|---|
| `inference.py:235-237` | `torch.cuda.reset_peak_memory_stats()`; `torch.cuda.max_memory_allocated()` | ✅ ROCm 支持显存统计 |
| `utils_infer.py:291,319`、`sample.py:286`、`train_controlnet.py:535` | `torch.cuda.empty_cache()` | ✅ |
| `utils_infer.py:197`、`sample_mask.py:81`、`diff_model_infer.py:172`、`diff_model_create_training_data.py:169` | `torch.amp.autocast("cuda")` | ⚠️ 同 AMP 判断 |
| `augmentation.py:104,152,208,260,314` | `(x).cuda()`（硬 `.cuda()`，不带设备号） | 🟡 DCU 下 `.cuda()` 仍映射到当前 HIP 设备可运行；但这些是**器官特异性肿瘤增强函数**（bone/liver/lung/pancreas/colon，CT 体部），经 `augmentation()`（`augmentation.py:436`）被 `sample.py:41` 推理用，**不在脑 MR 训练路径**；训练用的 `remove_tumors`（`augmentation.py:484`）不含 `.cuda()` |

---

## 3. requirements.txt 依赖在 DCU/DTK 的可安装性

**总原则 `[skill]`**：DCU 上 `torch`/`torchvision` 必须装 **sourcefind 的 DCU 构建**（PyPI 上是 CUDA 版，装上会崩或静默错算）；纯 python 包可走 PyPI，但**一律 `pip install --no-deps`**，且**装完任何带 numpy 依赖的包立刻验证 `import torch`**（防 numpy 被升 2.x 顶坏 DCU torch）。

| 包（requirements 约束） | DCU 可安装性 | 判断 |
|---|---|---|
| `torch>=2.1.0` | ⚠️ **不能 PyPI**；torch-dcu 是海光基于 ROCm fork 的厂商构建（HIP 化 + dispatch 修改），必须装 sourcefind 的 DCU 专用 wheel（也不能用 AMD ROCm 版）。当前实例 `torch 2.9.0+das.opt1.dtk2604`（DTK26.04，py3.11）`[skill]` | 满足 `>=2.1.0`；版本号含 `+das...` 本地后缀，pip 视其为 2.9.0 |
| `monai>=1.5.0` | ✅ 纯 python 为主，PyPI 装（`--no-deps`）。**关键版本坑**：上游 monai 1.5.0/1.5.1 的 setup.py 有 `torch<=2.6` 上限（[MONAI#8507](https://github.com/Project-MONAI/MONAI/issues/8507)），与本栈 torch 2.9 冲突 → **需 monai>=1.5.2（建议 1.6）**；skill 实测 monai 1.6 + numpy 1.26.4 在 DCU 栈可用 | `>=1.5.0` 会被 pip 解析到最新 1.6，满足；**勿钉死 1.5.0/1.5.1**。DCU 也可选 sourcefind 的 DCU 版 monai |
| `numpy>=1.24.0` | ⚠️ **必须钉 1.x**：`pip install "numpy==1.26.4"`（DCU torch 按 numpy 1.x 编，2.x 当场 `import torch` 崩）`[skill]` | 1.26.4 满足 `>=1.24.0` |
| `scipy>=1.10.0` | ✅ PyPI（`--no-deps`）；注意选与 numpy 1.x 兼容的版本（skill 实测 scipy 1.17 兼容 1.26.4） | 满足 |
| `scikit-image>=0.20.0` | ✅ PyPI（`--no-deps`） | 满足 |
| `nibabel>=5.0.0` | ✅ 纯 python | 满足 |
| `matplotlib>=3.7.0` | ✅ PyPI（`--no-deps`） | 满足 |
| `einops>=0.7.0` | ✅ 纯 python；**MONAI 注意力块（`crossattention.py:22`）与 flash_attn wheel 都隐式依赖它** | 满足 |
| `huggingface_hub>=0.20.0` | ✅ 纯 python；联网走代理 | 满足 |
| `tqdm>=4.65.0` / `fire>=0.5.0` / `PyYAML>=6.0` | ✅ 纯 python | 满足 |
| `tensorboard>=2.14.0` | ✅ 纯 python（`torch.utils.tensorboard`，`train_controlnet.py:30`） | 满足 |
| `torchvision`（**未在 requirements**） | ⚠️ FID 评估需要（`compute_fid_2-5d_ct.py:503`）；装 sourcefind DCU 版 `torchvision 0.24.0+das...` `[skill]` | **训练不需要**，仅评估；建议补入 DCU 环境清单 |

**numpy ABI 是 DCU 集群通病（非本项目特有）`[skill]`**：DCU torch 2.9.0 的 C 扩展按 numpy 1.x 编，而 PyPI 上 monai/scipy/scikit-learn 等新 wheel 要 numpy≥2，resolver 会把 numpy 升 2.x → `import torch` 崩 `A module compiled using NumPy 1.x cannot be run in NumPy 2.x`。**规则：装完任何 ML 依赖立刻 `python -c "import torch"` 验证；崩了 `pip install "numpy==1.26.4"` 修回。**

---

## 4. MONAI 专项

| 项 | 结论 | 证据 |
|---|---|---|
| **RFlowScheduler 是否纯 python** | ✅ **是**。`monai/networks/schedulers/rectified_flow.py` 只 import `numpy`、`torch`、`torch.distributions.LogisticNormal` 及 MONAI 自身 `StrEnum`/`DDPMPredictionType`/`Scheduler`；全部标准 tensor 运算（`torch.rand`/`randn_like`/`from_numpy`/`cat`/算术），**无自定义 CUDA kernel、无编译扩展**。DCU 上只要 torch 能跑它就能跑 | MONAI 源码（见附录 URL） |
| **noise_scheduler 实例化方式** | 经 MONAI `define_instance`（ConfigParser）从 JSON 实例化；`configs/config_network_rflow.json:135` 的 `_target_` = `monai.networks.schedulers.rectified_flow.RFlowScheduler` | `utils.py:208-219`、`diff_model_train.py:431` |
| **CacheDataset** | ✅ 纯 python，CPU 侧 RAM 缓存（`cache_rate`），`num_workers` 并行加载在 CPU 进程；设备无关 | `utils.py:324,333`、`diff_model_train.py:137` |
| **generate_param_groups** | ✅ **本仓库未使用**。issue 把它列为候选，实际代码只读 `optimizer.param_groups[0]["lr"]`（读当前 LR 用）；优化器是裸 `torch.optim.Adam`/`AdamW`，无 MONAI 参数组工具 | `diff_model_train.py:270,278`；全库 grep 无 `generate_param_groups` |
| **flash attention 底层** | `use_flash_attention: true`（`config_network_rflow.json` 的 DiffusionModelUNetMaisi:58 / ControlNetMaisi:84）→ MONAI 调 **`torch.nn.functional.scaled_dot_product_attention`（PyTorch 内置 SDPA）**，**不是 `flash_attn` 包**。故 sourcefind 的 flash_attn wheel 对本仓库注意力路径**非必需** | `crossattention.py:167`、`diffusion_model_unet.py:71` docstring |
| **upcast_attention=true** | 注意力上采样到 fp32 计算，配合 AMP 更稳；DCU 上无特殊问题 `[推断]` | `config_network_rflow.json`（DM unet `upcast_attention: true`） |
| **SDPA 在 DCU 的后端** | ⚠️ SDPA 在 ROCm 有 flash/mem-efficient/math 后端；ROCm 旧文档曾标注 SDPA 的 FlashAttention 后端不支持（会回落 mem-efficient/math，**正确性保底**）。DCU 另有 sourcefind 独立 `flash_attn` 包（`flash_attn-2.6.1+das...`）可作加速备选，但 MONAI 这条注意力路径用的是 torch SDPA、不读该包 | 见 §6-3 |

---

## 5. DCU 训练环境推荐形态

### 5.1 版本匹配（当前实例实测栈 `[skill]`）

| 层 | 版本 | 来源 |
|---|---|---|
| DTK（DCU Toolkit） | 26.04 | 系统 `/opt/dtk/env.sh` |
| Python | 3.11（系统 `/usr/local/bin/python`，**非 conda**） | 实例自带 |
| torch | `2.9.0+das.opt1.dtk2604` | sourcefind DAS1.8 ↔ DTK26.04 ↔ torch2.9 |
| torchvision | `0.24.0+das.opt1.dtk2604` | sourcefind（仅 FID 评估用） |
| numpy | **1.26.4（钉死）** | PyPI |
| monai | **>=1.5.2（建议 1.6）**——勿用 1.5.0/1.5.1（钉 `torch<=2.6`，与 torch2.9 冲突，[MONAI#8507](https://github.com/Project-MONAI/MONAI/issues/8507)） | PyPI `--no-deps`，或 sourcefind DCU 版 |

> 版本匹配要点 `[skill]`：sourcefind 的 wheel 名后缀 `.torch290.cp311.dtk2604` = torch2.9 / py3.11 / DTK26.04 三元组，**装包时认准本栈后缀**。换 DTK 大版本需整套换对应 torch 构建。

**DAS ↔ DTK ↔ torch 公开对照（二手来源交叉印证，非官方全表）**：sourcefind 全表需登录开发者账号，公网无完整对照表；以下为多个一致来源的规律——

| DAS | DTK | torch | wheel 后缀示例 |
|---|---|---|---|
| DAS1.1.1 | 24.04 | 2.1.0 | `+das1.1...dtk2404` |
| DAS1.3 | 24.04.3 | 2.3.0 | `+das.opt2.dtk24043` |
| — | 25.04(.1) | 2.4.1 | `+das.opt1.dtk25041`（配 flash_attn 2.6.1） |
| DAS1.8 | 26.04 | 2.5.1 / **2.9.0**（本实例） | `+das.opt1.dtk2604` |

> 同一 DTK 大版本似对应多个 torch 档位（26.04 下公网可见 2.5.1，本实例为 2.9.0，另有 2.10 说法）。比 26.04 更新的 DTK 公网无信息，26.04 目前是较新版。规律可信，具体 wheel 串以 sourcefind 实际列表为准。

### 5.2 安装形态（容器/镜像）

- **官方基础镜像（推荐起点）**：曙光 sourcefind 提供预装 DTK+torch 的 docker 镜像，registry 为 **`image.sourcefind.cn:5000`**，命名规律 `dcu/admin/base/pytorch:{torch}-{os}-dtk{dtk}-py{py}`（例 `pytorch:2.4.1-ubuntu22.04-dtk25.04-py3.10`）。入口 `sourcefind.cn` / `developer.sourcefind.cn`（完整列表/下载需登录）。优先用官方镜像打底，省去手装 DTK+torch。
- **wheel 包来源** `[skill]`：`download.sourcefind.cn:65024`（AI 生态包 CategoryID=4）。**注意：sourcefind 是 JS 文件浏览器、不是 PEP503 pip 索引**——`pip install --find-links/--index-url` 解析不到会回退 PyPI 拉 CUDA sdist，必须用 `.whl` **直链 + `pip install --no-deps --no-index <url>`**。
- **必须从 sourcefind 装 DCU 版的包**（含 GPU 二进制）：`torch`、`torchvision`、`triton`、`flash_attn`、`deepspeed`、`bitsandbytes`、`xformers`、`monai`（可选）、`vllm`。**纯 python 上层库**（`numpy`/`scipy`/`nibabel`/`einops`/`transformers`/`diffusers`/`accelerate` 等）可直接 PyPI。本仓库训练只用 torch + monai（纯 python 路径），故 torch 必须 sourcefind、monai 可 PyPI（>=1.5.2）。
- **双 source 前置** `[skill]`：任何 torch/GPU 操作前 `source /opt/dtk/env.sh`（DTK 算）；任何 pip/curl/git 联网前 `source .../ai_proxy`（平台 squid 代理）。可一次性幂等写进 `~/.bashrc`（注意系统盘易失，重置后需重注）。
- **安装纪律** `[skill]`：① 永不 `pip install torch`（PyPI CUDA 版会覆盖 DCU torch）；② 所有安装 `--no-deps`；③ 装完立验 `import torch`；④ checkpoint/log/数据**只写持久盘** `/root/private_data`（系统盘 `/` 易失）。

### 5.3 多卡 DDP 在 SothisAI 平台的提交要点

- **本仓库的标准启动就是单机多卡 torchrun**（`docs/training.md`）：

  ```bash
  torchrun --nproc_per_node=${N} --nnodes=1 \
      --master_addr=localhost --master_port=1234 \
      -m scripts.diff_model_train -t <network.json> -c <model.json> -e <env.json> -g ${N}
  ```

  `torchrun` 是 torch 自带 launcher，DCU 栈同样提供；脚本内 `LOCAL_RANK`/`WORLD_SIZE` 从 env 读（`train_controlnet.py:247-248`），`init_method="env://"`（`diff_model_setting.py:87`、`utils.py:202`）与 torchrun 的 env 注入天然匹配。
- **平台形态**：SothisAI 是曙光 AI 任务管理 + 算力调度平台，底层 **Slurm + Kubernetes 双调度、容器化运行**；提交方式有 Web GUI（模型训练→训练任务→创建训练，可选分布式/非分布式，设每 Worker 的 DCU 数/CPU/内存/镜像/超时）与 SSH 登录后 Slurm（`sbatch`/`srun`）两种。对本仓库**单机多卡**场景，容器内 `torchrun --nproc_per_node=N` 即标准做法，无需特殊平台钩子。
- **⚠️ 最长运行时限**：平台训练任务常有最大运行时间限制（常见 24/48/72h），超时强杀——**长训练必须高频存 checkpoint 到持久盘** `/root/private_data`（本仓库 `diff_model_train.py`/`train_controlnet.py` 每 epoch 存 ckpt，且支持 `existing_ckpt_filepath` 续训，天然适配断点续跑）。
- **多节点**：本仓库启动封装 `run_torchrun` 写死 `nnodes=1`（`diff_model_setting.py:99`），当前设计面是单机多卡；若需跨节点需另配 rendezvous/调度，超出本期范围。

---

## 6. 需实机验证清单（本票不做，属后续执行票）

| # | 验证项 | 为什么需要 | 验证方法（建议） |
|---|---|---|---|
| 1 | **device 命名**：torch-dcu 下 `torch.device("cuda")`/`torch.cuda.is_available()` 是否确为 `"cuda"` 命名 | 全仓库兼容性地基（§1） | `python -c "import torch;print(torch.cuda.is_available(),torch.cuda.device_count(),torch.get_device_name(0))"` |
| 2 | **分布式后端**：`backend="nccl"` 在 torch-dcu 是否可用并落到 RCCL/HCCL；2 卡 all_reduce 是否通 | DDP 前提（§2.2/2.4） | `python -c "import torch.distributed as d; print(d.is_nccl_available()); print(d.get_default_backend_for_device('cuda'))"`；再 2 卡 torchrun 跑最小 all_reduce |
| 3 | **SDPA 后端**：`use_flash_attention=true` 下 SDPA 在 DCU 走哪个后端、是否正确/够快；必要时回退 | 影响训练性能与显存（§4） | 单步前向，或 `torch.nn.attention.sdpa_kernel` 强制 math 对比 |
| 4 | **AMP/GradScaler**：`autocast("cuda")`+`GradScaler("cuda")` 在 DCU 是否正确缩放、不 NaN（ROCm 有 fp16 GradScaler NaN 报告 [ROCm/ROCm#6351](https://github.com/ROCm/ROCm/issues/6351)） | **P2 全程 AMP 无开关**（§2.3），最高风险 | 单卡跑几步 P2 训练看 loss 是否有限/下降；并对比 `bf16 autocast（停 GradScaler）` 路径 |
| 5 | **SyncBatchNorm**：`convert_sync_batchnorm` 后多卡前向/反向 kernel 是否可用 | P1 多卡（§2.1-155） | 2 卡跑几步 P1 训练 |
| 6 | **算子覆盖**：rectified flow 路径所有算子（3D conv、group norm、SDPA、LogisticNormal 采样）在 DCU kernel 是否齐全 | 3D 大 tensor 可能有未实现 kernel | 单卡端到端跑通 1 epoch 冒烟 |
| 7 | **numpy 版本**：装完 monai 等依赖后 `import torch` 是否仍好（防 numpy 2.x） | 环境装机必踩（§3） | 装机脚本里固化为检查步骤 |
| 8 | **设备可见性变量**：DCU 用 `CUDA_VISIBLE_DEVICES` 还是 `HIP_VISIBLE_DEVICES`/专有变量 | FID 脚本/多卡选卡（§2.4-21） | `hy-smi` + 设变量后看 `device_count()` |
| 9 | **CacheDataset 大数据 RAM**：`cache_rate` 下 BraTS latent 缓存是否超 CPU 内存 | 数据加载（§4） | 实机按数据集规模调 `cache_rate` |
| 10 | **显存容量**：单 DCU VRAM 是否够目标输出尺寸（512³ 训练需 ≥40GB，`docs/setup.md`） | 决定 batch_size/输出尺寸上限 | 实机 `hy-smi` 看显存，从小尺寸起冒烟 |
| 11 | **monai↔torch 版本**：实装 monai 版本是否 ≥1.5.2（与 torch 2.9 兼容）；`import monai` 后与 torch 无冲突 | 装错 monai 1.5.0/1.5.1 会试图降级 torch（§3） | `python -c "import monai,torch; print(monai.__version__, torch.__version__)"` 并跑 `from monai.networks.schedulers import RFlowScheduler` |

---

## 7. 风险清单汇总

| 风险 | 级别 | 性质 | 缓解 |
|---|---|---|---|
| 依赖安装（PyPI CUDA torch 覆盖 / numpy 2.x / sourcefind 非索引） | **高** | 环境 | `--no-deps` + numpy 钉 1.26.4 + sourcefind 直链 + 装后验 `import torch` `[skill]` |
| monai 版本钉错（1.5.0/1.5.1 钉 `torch<=2.6`，与 torch2.9 冲突） | **中-高** | 环境 | 装 **monai>=1.5.2（建议 1.6）**；`requirements.txt` 的 `>=1.5.0` 会被解析到最新版，天然规避，但勿手动钉 1.5.0/1.5.1 |
| P2 全程 fp16 AMP 无开关，GradScaler 在 DCU 有 NaN 风险 | **中-高** | 运行时 | 实机验证（§6-4）；优先评估 **bf16 autocast（免 GradScaler）**；必要时给 ControlNet 加 `amp` 开关走 fp32 |
| NCCL→HCCL 后端映射 | **低** | 运行时 | 保持 `backend="nccl"`（PyTorch 默认 cuda→nccl 自动落 RCCL/HCCL）；实机 `is_nccl_available()` 终验（§6-2）；**勿改 `"hccl"`** |
| SDPA 高效 kernel 在 DCU 可用性 | **中** | 性能 | math/mem-efficient fallback 保底正确性；必要时 `use_flash_attention=false` 或试 sourcefind flash_attn |
| SyncBatchNorm DCU kernel | 低-中 | 运行时 | 实机验证（§6-5）；单卡可去 |
| torchvision 未入 requirements（FID 评估用） | 低 | 环境 | 补 sourcefind DCU torchvision；不影响训练 |
| 平台最长运行时限（24/48/72h）强杀长训练 | **中（运维）** | 环境 | 高频存 ckpt 到 `/root/private_data`；本仓库每 epoch 存 + 支持续训，天然适配 |
| `augmentation.py` 器官函数硬 `.cuda()` | 低 | 代码规范 | DCU 下可运行；非脑训练路径；可后续改 `.to(device)` |
| 系统盘易失 → checkpoint 丢失风险 | 高（运维） | 环境 | 一律写 `/root/private_data` `[skill]` |

---

## 附录：证据来源

**本仓库代码**（行号见正文）：`scripts/diff_model_train.py`、`scripts/train_controlnet.py`、`scripts/diff_model_setting.py`、`scripts/utils.py`、`scripts/diff_model_create_training_data.py`、`scripts/compute_fid_2-5d_ct.py`、`scripts/augmentation.py`、`configs/config_network_rflow.json`、`requirements.txt`、`docs/setup.md`、`docs/training.md`、`data/README.md`。

**MONAI 源码**（dev 分支，验证 RFlowScheduler 纯 python / flash attention=SDPA）：

- `monai/networks/schedulers/rectified_flow.py`：<https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/monai/networks/schedulers/rectified_flow.py>
- `monai/networks/blocks/crossattention.py`（SDPA 调用在 :167）：<https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/monai/networks/blocks/crossattention.py>
- `monai/networks/nets/diffusion_model_unet.py`（`use_flash_attention` docstring :71）：<https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/monai/networks/nets/diffusion_model_unet.py>

**ROCm 生态公开文档**（佐证 `"cuda"` 命名沿用 / AMP 支持状态，DCU 为其衍生）：

- Kubeflow PyTorch-on-ROCm 指南（用 `torch.cuda.*` 操作 AMD GPU）：<https://trainer.kubeflow.org/en/latest/user-guides/pytorch-rocm.html>
- AMD 官方博客 torch.compile/AMP on ROCm（`torch.cuda.device_count()` 返回 AMD 设备）：<https://rocm.blogs.amd.com/artificial-intelligence/torch_compile/README.html>
- [ROCm/ROCm#4231](https://github.com/ROCm/ROCm/issues/4231)：feature request 求显式 `torch.device("rocm")`——反证 ROCm 现状沿用 `"cuda"` 命名
- [ROCm/ROCm#6351](https://github.com/ROCm/ROCm/issues/6351)：GradScaler 在 ROCm 上 fp16 训练数步后 NaN 的报告；AMD 建议 bf16 更稳
- AMD 官方 AMP 指南（autocast/GradScaler on ROCm，bf16 无需 GradScaler）：<https://rocm.blogs.amd.com/artificial-intelligence/automatic-mixed-precision/README.html>
- AMD 官方 PyTorch 兼容性矩阵（ROCm 版 autocast/GradScaler/TF32 支持范围）：<https://rocm.docs.amd.com/en/docs-6.4.0/compatibility/ml-compatibility/pytorch-compatibility.html>

**曙光 DCU / DTK / SothisAI / sourcefind**（多为二手非官方来源，交叉印证，供参考；权威以 sourcefind 实际列表与平台界面为准）：

- 海光 DCU 分布式示例仓库（标准 `torchrun --nproc_per_node` + `"nccl"`，无特殊 backend 名）：<https://github.com/FlyAIBox/dcu-in-action>
- PyTorch 分布式文档（默认 cuda→nccl / cpu→gloo / xpu→xccl 映射）：<https://docs.pytorch.ac.cn/docs/stable/distributed.html>
- DAS↔DTK↔torch 版本与 DCU wheel 实例（ktransformers-dcu，DTK26.04↔torch2.5.1）：<https://github.com/lutianshu824/ktransformers-dcu>
- sourcefind 镜像/wheel 命名规律（`image.sourcefind.cn:5000`、`download.sourcefind.cn:65024`）、DTK25.04↔torch2.4.1、flash_attn 2.6.1：<https://blog.csdn.net/JulyLi2019/article/details/153266775>、<https://blog.csdn.net/zhangfeng1133/article/details/161574507>、<https://www.psvmc.cn/article/2026-02-03-ai-hygon-gpu.html>
- SothisAI 平台（Slurm+K8s 容器化、训练任务 GUI、最长运行时限）：SothisAI V2.8 用户手册 <https://ai.bnu.edu.cn/docs/2023-08/80e753f9088b43108a4cad980a17c5f8.pdf>、<https://blog.csdn.net/zhangfeng1133/article/details/160934380>
- DCU 上 `torch.cuda.*` 沿用 `"cuda"` 命名、`torch.version.hip` 判别 ROCm 系：<https://www.kuazhi.com/post/716519037.html>

**MONAI 版本兼容**：

- [MONAI#8507](https://github.com/Project-MONAI/MONAI/issues/8507)：monai 1.5.0/1.5.1 setup.py 的 `torch<=2.6` 上限问题（→ torch2.9 需 monai>=1.5.2）
- MONAI PyPI（版本与依赖）：<https://pypi.org/project/monai/>
- AMD ROCm 版 MONAI（amd-monai，面向 Instinct/MI300X，不适用海光 DCU，仅佐证 MONAI 可移植到 ROCm 系）：<https://rocm.docs.amd.com/projects/monai/en/latest/install/installation.html>

**集群环境一手实测** `[skill]`：本机 `sugon-bootstrap` skill（`SKILL.md` + `references/dcu-pitfalls.md`）——DTK26.04/torch2.9.0+das/python3.11 栈、sourcefind 安装机制、numpy ABI 陷阱、双 source、挂载点持久性。
