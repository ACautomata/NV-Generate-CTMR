# 最小训练量微调方法调研（issue #5）

> 隶属地图票 [#2](https://github.com/ACautomata/NV-Generate-CTMR/issues/2)。
> 目标：在本仓库代码事实约束下，给出 P1/P2 各自「相对最小训练量」的微调方法，外加 P3 前置侦察。
> 已定约束（地图钉死）：基础模型 `rflow-mr-brain v1`；数据 = BraTS2023 全部子挑战；**能冻结就不训**（VAE、DM 优先冻结，优先只训 ControlNet）。
> 证据标注：`文件:行号` 为本仓库代码；`URL` 为外部来源。

---

## 0. 结论速览（三路径对比）

| 维度 | 路径 A 全参微调 DM | 路径 B ControlNet-only | 路径 C LoRA/部分层 |
|---|---|---|---|
| 适用阶段 | **P1**（image-only 带肿瘤生成，须改 DM 本体） | **P2**（掩码→影像配对生成） | P1 的省显存替代（非常规） |
| 训练对象 | 整个 DiffusionUNet（冻结 VAE） | 仅 ControlNet 旁路，**DM 全冻结** | DM 内插入的低秩旁路 |
| 仓库现成支持 | ✅ `scripts/diff_model_train.py` 直接支持续训 | ✅ `scripts/train_controlnet.py` 直接支持 | ❌ 无任何现成支持 |
| 改动量 | 仅改 config（数据/路径/超参） | 仅改 config + 新建 brain env config | 需改网络定义 + 训练循环（侵入式） |
| 训练量 | 中（全参反传，但只训 DM） | **最小**（只训旁路，DM 前向冻结） | 小（但工程量大） |
| 灾难性遗忘风险 | **高**（小数据全参）→ 需回放混合 | 无（DM 冻结，原能力天然保留） | 低 |
| 对「能冻结就不训」约束的契合 | 仅 P1 无可避免时采用 | **最契合**，P2 首选 | 契合但不值得 |
| 推荐 | **P1 用**，配 MR-RATE 回放防遗忘 | **P2 用**，即插即用 | **不推荐**本期投入 |

**一句话**：P1 走路径 A（全参微调 DM + MR-RATE 回放混合）；P2 走路径 B（ControlNet-only，DM 冻结，即插即用）；路径 C（LoRA）改造量大、收益不抵工程成本，本期不建议。

---

## 1. 路径 A —— 全参微调 DM（P1）

### 1.1 续训机制（代码事实）

`scripts/diff_model_train.py` 原生支持从 pretrained checkpoint 续训，**无需改任何 Python 代码**，仅需改 config：

- 入口 `load_unet()`：`existing_ckpt_filepath` 非空时，`torch.load` 后仅加载 `unet_state_dict`（`strict=False`），日志确认加载（`diff_model_train.py:160-169`）。**注意：只加载 UNet 权重，不加载 checkpoint 里存的 `scale_factor`**。
- env config `configs/environment_maisi_diff_model_rflow-mr-brain.json:10` 已把 `existing_ckpt_filepath` 指向 `./models/diff_unet_3d_rflow-mr-brain_v1.pt` —— 续训路径开箱即用。
- 优化器：`torch.optim.Adam(params=model.parameters(), lr=lr)`，**全参**反传（`diff_model_train.py:198-209`）。
- 调度器：`PolynomialLR(total_iters=total_steps, power=2.0)`，按 step 衰减到 0（`diff_model_train.py:212-223`，`train_one_epoch` 内 `lr_scheduler.step()` 每 iter 调用，`diff_model_train.py:353`）。
- `total_steps = n_epochs * len(dataset) / batch_size`（`diff_model_train.py:474`）。
- Loss：`L1Loss`，rectified flow 目标 `model_gt = images - noise`（沿线性路径的速度，`diff_model_train.py:327-329,343`）。
- 每 epoch 末 `save_checkpoint` 存 `{epoch, loss, num_train_timesteps, scale_factor, unet_state_dict}`（`diff_model_train.py:369-400`）。

### 1.2 scale_factor 处理（关键微调细节）

- 训练脚本里 `scale_factor` **不是从 ckpt 读，而是用新训练数据首 batch 重算**：`scale_factor = 1 / torch.std(z)`（`diff_model_train.py:174-195`，`471` 行无条件调用 `calculate_scale_factor`）。
- 推理侧 `scale_factor` 从 checkpoint 读（`diff_model_infer.py:73`），而新 checkpoint 存的正是这个**重算值** → 推理自洽，不会崩。
- **但存在分布偏移隐患**：预训练权重是在 MR-RATE 的 scale_factor 下学出的输入分布；若 BraTS latent 的 std 与 MR-RATE 不同，重算的 scale_factor 会让 `images * scale_factor` 后的输入分布相对预训练期望发生偏移，等效于微调初期给模型一个输入尺度冲击。
- **建议**：BraTS 与 MR-RATE 同为脑 MRI、走同一 VAE 编码，latent std 应接近，重算通常可接受；但更稳妥的做法是**复用 ckpt 的 scale_factor**（把 `calculate_scale_factor` 替换为读取 `existing_ckpt_filepath` 里的 `scale_factor`，一行改动）。这是一个可选的稳妥化改造，非必须。

### 1.3 改成微调 BraTS 需动的字段

`configs/environment_maisi_diff_model_rflow-mr-brain.json`：

| 字段 | 现值 | 改为 |
|---|---|---|
| `embedding_base_dir` | `./embeddings` | BraTS 预处理后的 embedding 目录 |
| `json_data_list` | `./dataset.json` | BraTS data list JSON |
| `model_dir` / `model_filename` | `./models` / `diff_unet_3d_rflow-mr-brain_v1.pt` | 输出目录 / 新文件名（避免覆盖基座） |
| `existing_ckpt_filepath` | 已指向基座 | 保持指向 `diff_unet_3d_rflow-mr-brain_v1.pt`（作初始化） |
| `modality_mapping_path` | `./configs/modality_mapping.json` | 保持 |

`configs/config_maisi_diff_model_rflow-mr-brain.json` 的 `diffusion_unet_train`：

| 字段 | 现值 | 微调建议 |
|---|---|---|
| `lr` | `0.00001` (1e-5) | 见 §4 起点（1e-6 ~ 1e-5 量级） |
| `n_epochs` | `1000` | 微调远小于从头训，见 §4 |
| `batch_size` | `1` | 3D whole-volume，保持 1 |
| `cache_rate` | `0` | 视 RAM 调 |

**数据前提**：BraTS 影像需先用 `scripts/diff_model_create_training_data.py` 经 VAE 编码成 `*_emb.nii.gz`（data/README §4.3 描述的 embedding 流程），并在每个 `*_emb.nii.gz.json` 里提供 `spacing` 与 `modality` 字段（`diff_model_train.py:120-134` 读取）。modality 取脑 MRI 对比度对应的整数（`configs/modality_mapping.json`：T1w=9、T2w=10、FLAIR=11、SWI=20、及 skull-stripped 29-32）。

### 1.4 防灾难性遗忘 —— MR-RATE 回放混合

- **回放数据源天然可得**：`rflow-mr-brain v0` 的训练数据就是 [MR-RATE](https://huggingface.co/datasets/Forithmus/MR-RATE)（README.md:40）。用 MR-RATE 作回放（replay）数据，与 BraTS 混合训练，是防遗忘的直接手段——模型在拟合 BraTS 肿瘤分布的同时被锚定在原脑 MRI 分布上。
- **实现**：仓库 `diff_model_train.py` 的 `load_filenames` 只读单个 `json_data_list`（`diff_model_train.py:69-82`），不支持多 list 混合。要混合需在 **数据 list 层**把 MR-RATE 与 BraTS 条目合并进同一个 JSON 的 `training` 数组（可按比例重复/采样），无需改训练代码。
- **混合比例先例**：见 §4 文献。

---

## 2. 路径 B —— ControlNet-only（P2，首选）

### 2.1 机制（代码事实）

`scripts/train_controlnet.py` 完全实现「冻结 DM、只训旁路」：

1. 加载冻结 DM：`trained_diffusion_path` 非空（强制，`train_controlnet.py:292,302`），`load_state_dict(unet_state_dict, strict=False)`，**并从 ckpt 读 `scale_factor`**（`train_controlnet.py:295-300` —— 与路径 A 不同，这里保留原 scale_factor）。
2. 建 ControlNet 并**从 DM encoder/mid 权重初始化**：`copy_model_state(controlnet, unet.state_dict())`（`train_controlnet.py:305-307`）。
3. 可选从 `existing_ckpt_filepath` 暖启动 ControlNet（`train_controlnet.py:309-313`）。
4. **冻结 DM 全部参数**：`for p in unet.parameters(): p.requires_grad = False`（`train_controlnet.py:317-318`）。
5. 只训 ControlNet：`AdamW(controlnet.parameters(), lr)` + `PolynomialLR(power=2.0)`（`train_controlnet.py:351-355`）。

### 2.2 对 rflow-mr-brain checkpoint 的适配点 —— **即插即用（结构兼容）**

把 `trained_diffusion_path` 指向 `diff_unet_3d_rflow-mr-brain_v1.pt` **结构上即插即用**，依据：

- 全仓库只有一个 rflow 网络定义 `configs/config_network_rflow.json`，`rflow-mr` 与 `rflow-mr-brain` 共用（docs/training.md:43-58：`network="rflow"`；README brain 推理也用 `config_network_rflow.json`）。
- `diffusion_unet_def` 与 `controlnet_def` 在 encoder 侧完全对齐：`num_channels=[64,128,256,512]`、`attention_levels=[F,F,T,T]`、`num_head_channels=[0,0,32,32]`、`num_res_blocks=2`、`num_class_embeds=128`、`in_channels=latent_channels=4`（`config_network_rflow.json:39-90`）。`copy_model_state` 按键名匹配拷贝 encoder/mid，结构一致故能拷上。
- brain 模型 `include_body_region=false`（`config_network_rflow.json:5`），与 controlnet rflow 配置一致，无需 top/bottom region 字段。
- `num_class_embeds=128`：脑模态最高索引 32（`mri_swi_skull_stripped`）< 128，class embedding 尺寸兼容。

**注意（须新建 config）**：仓库未附带 brain 专用 ControlNet env config（`ls configs/ | grep controlnet.*brain` 无结果）。须复制 `configs/environment_maisi_controlnet_train_rflow-mr.json`，改两处：

- `trained_diffusion_path` → `./models/diff_unet_3d_rflow-mr-brain_v1.pt`
- `data_base_dir` / `json_data_list` → BraTS 配对数据（`*_emb.nii.gz` + `*_combined_label.nii.gz`）

`configs/config_maisi_controlnet_train_rflow-mr.json` 现有 `controlnet_train`：`batch_size=1, lr=1e-5, n_epochs=100, weighted_loss=100, weighted_loss_label=[129]`（`config_maisi_controlnet_train_rflow-mr.json:1-10`）。

### 2.3 关键障碍：脑肿瘤亚区标签超 255

- ControlNet 条件经 `binarize_labels` 做 **8 位二进制编码 → 8 通道**，上限 **256 类（0–255）**（`utils.py:173-188`；`config_network_rflow.json:85` `conditioning_embedding_in_channels=8`；docs/training.md:74「support up to 256 labels (0~255)」）。
- 但 `configs/label_dict_ctmr.json` 中脑肿瘤亚区标签为 **NCR/NETC=401、ED=402、ET=403**（`label_dict_ctmr.json:347-349`），**超出 255，无法被 8 位编码表示**（会发生混叠）。
- **结论**：P2 数据预处理时须把 BraTS 亚区标签**重映射到 0–255 空段**。仓库预留了空段：`label_dict.json` 的 `dummy1-8`（含 129/130/131，`label_dict.json:130-132`）与 `133–199`、`201–255` 空区间（data/README §4.3）。整体肿瘤可用 `brain tumor=176`（`label_dict_ctmr.json:177`），或将亚区分别映射到 dummy 槽位。`weighted_loss_label` 同步改为肿瘤对应的新索引（up-weight 肿瘤区域的 L1，`train_controlnet.py:454-462`）。

### 2.4 训练数据格式（P2）

每例一对：`(image=*_emb.nii.gz [4通道latent], label=*_combined_label.nii.gz [整数掩码])`；label 空间尺寸须为 latent 的 **4×/轴**（ControlNet 内部下采样 4×），label 用最近邻重采样；`fold` 字段跨多折分布否则训练集为空；`modality` 必填。详见 `skills/train_controlnet_image-from-mask.md` 与 `scripts/utils.py:261-340`。

---

## 3. 路径 C —— LoRA / 部分层冻结（不推荐本期投入）

### 3.1 网络结构里可插 LoRA 的位置

`diffusion_unet_def` 是 `DiffusionModelUNetMaisi`（`config_network_rflow.json:39-65`）：

- 注意力只在**最深两层**开启：`attention_levels=[false,false,true,true]`，`num_head_channels=[0,0,32,32]`（即 ch=256、512 两级有 self-attention）。
- 卷积主体：4 级 `num_channels=[64,128,256,512]`，`num_res_blocks=2`。
- 条件注入：`include_spacing_input=true`、`num_class_embeds=128`（class embedding）、`include_fc=true`。

可插 LoRA 的自然位置 = 两个 attention 级的 QKV/proj 线性层，以及各级 resblock 卷积。attention 只在 2/4 级，LoRA 覆盖面有限。

### 3.2 改造量评估

- **monai 兼容性**：`DiffusionModelUNetMaisi` 是 MAISI 定制网络（`monai.apps.generation.maisi.networks.diffusion_model_unet_maisi`），非 HuggingFace diffusers 模型，**没有现成的 PEFT/`peft` LoRA 适配器**。需手写 LoRA 包装：定位 attention/resblock 线性层 → 包一层低秩旁路 → 冻结主干。
- **需改文件**：
  - 网络实例化：`define_instance(args, "diffusion_unet_def")`（`diff_model_train.py:154`）后需插入 LoRA 包装逻辑 —— 改 `diff_model_train.py::load_unet`。
  - 优化器：现 `create_optimizer` 直接 `model.parameters()`（`diff_model_train.py:209`），需改成只收集 LoRA 参数。
  - checkpoint 存取：现只存/取 `unet_state_dict`，需额外存 LoRA 权重。
  - 推理：`diff_model_infer.py` 加载时须同样注入 LoRA 才能复现结构 —— 推理侧也要改。
- **收益对比**：训练量省的是显存与优化器状态，但 3D whole-volume batch_size=1 下显存压力本可控；DM 全参微调在 6–10k scans 量级主要风险是**遗忘**，而 LoRA 防遗忘收益可用**更简单的回放混合（路径 A §1.4）**等价达成。
- **结论**：改造侵入网络定义 + 训练 + 推理三处，工程量大、无现成支持、收益可被回放混合替代。**本期不值得**，列为未来可选优化。

---

## 4. 文献先例与 P1 超参起点

> 本节数字来源见 §6 引用；以下为综合结论。

### 4.1 基础模型本身的数据点（来自本仓库/官方）

- VAE：从头训 280 epochs / 58k 数据（docs/training.md:37）。
- DM（rflow-mr-brain）：config 默认 `lr=1e-5, n_epochs=1000, batch_size=1`（`config_maisi_diff_model_rflow-mr-brain.json:2-7`）——这是**从头训**量级，微调应显著低于此。
- ControlNet（rflow-mr 微调示例）：`lr=1e-5, n_epochs=100, batch_size=1`（`config_maisi_controlnet_train_rflow-mr.json:2-7`）。

### 4.2 P1 全参微调超参起点（建议）

BraTS 全子挑战约 6–10k scans，相对 58k 从头训属小数据。起点建议：

| 超参 | 起点值 | 依据 |
|---|---|---|
| 初始化 | `diff_unet_3d_rflow-mr-brain_v1.pt` 全参加载 | §1.1 |
| lr | **1e-6 ~ 5e-6**（低于从头训的 1e-5） | 微调惯例取预训练 lr 的 1/10~1/2；小数据防过拟合/遗忘偏低端 |
| batch_size | 1（3D whole-volume，显存约束） | config 默认；DCU 显存允许可试 2 |
| n_epochs | **50–200** 区间起步，按验证/质量监控早停 | 远小于从头训 1000 |
| scheduler | 沿用 `PolynomialLR(power=2.0)` | 仓库内置，无需改 |
| loss | L1 + rectified-flow 速度目标 | 仓库内置 |
| scale_factor | 优先复用 ckpt 值（可选改代码），或接受重算 | §1.2 |
| **回放混合** | MR-RATE : BraTS ≈ **1:1 起步**，遗忘明显则上调 MR-RATE 占比 | 防遗忘；见 §6 文献 |

### 4.3 Rectified flow 微调注意点

- 本仓库 `RFlowScheduler`：`num_train_timesteps=1000, use_discrete_timesteps=false, use_timestep_transform=true, sample_method="uniform", scale=1.4`（`config_network_rflow.json:134-141`）。
- 训练时 `timesteps = noise_scheduler.sample_timesteps(images)`（uniform 采样，`diff_model_train.py:299`）；目标 `model_gt = images - noise`（线性路径速度）。
- **微调无需改 timestep 采样**——沿用与预训练一致的 uniform 采样即可，保持训练/预训练分布一致是微调稳妥做法。文献对 rectified flow 微调是否需改 timestep 分布（如 logit-normal/mode sampling 偏向中段）见 §6。

### 4.4 文献先例（BraTS/脑肿瘤 3D 生成微调 + 回放比例 + RF 注意点）

**BraTS / 脑肿瘤 3D 生成模型训练/微调超参量级**（典型：lr≈1e-4 从头训、batch 1–2 受显存限、微调以 pretrained 初始化继续约 1e5 步量级）：

| 论文 | 数据 | 初始化 | LR / batch / 训练量 | 防遗忘 |
|---|---|---|---|---|
| 3D Healthy Brain Inpainting ([arXiv:2403.14499](https://arxiv.org/html/2403.14499v1)) | BraTS2023（1251 训练扫描） | 从头训 | lr 1e-4（AE 3e-4），batch 1–2，扩散模型 3.25e5 迭代；**换数据集微调额外 1e5 迭代** | 单任务微调 |
| Biophysically-Conditioned LDM ([arXiv:2510.09365](https://arxiv.org/html/2510.09365v1)) | BraTS2021+UCSF 等 3602 受试者 | 从头训 | AdamW lr 1e-4, wd 0.01, batch 2, cosine，2×H100 约 2.5 周 | — |
| Med-DDPM ([GitHub](https://github.com/JotatD/med-ddpm-brats)) | BraTS2021 精选 193 张 4 模态 | 从头训 | 语义条件 3D DDPM | — |
| LDM-seg ([ISBI 2025](https://www2.die.upm.es/im/papers/ISBI2025.pdf)) | BraTS2023 Adult Glioma | 压缩模型→无条件 LDM→ControlNet | LDM 250 epoch；**ControlNet 700 epoch，冻结预训练 U-Net** | ControlNet 冻结主干天然防遗忘 |
| Few-Shot Continual LoRA ([arXiv:2602.23533](https://arxiv.org/pdf/2602.23533)) | BraTS2023 Glioma, IXI | 冻结主干 + 每任务 LoRA | 只训 adapter | **设计上消除遗忘**；对照：顺序全参微调 Dice 0.80→0.16 |
| Multi-Scale Reward RL ([arXiv:2603.06173](https://arxiv.org/html/2603.06173)) | BraTS2019, OASIS-1 | 预训练→PPO 微调 50k 步 | — | KL 正则锚定预训练策略 |

> 对本任务：BraTS2023（~千级扫描）属小数据。文献主流 = pretrained 初始化 + 较低 lr 继续训练约 **1e5 迭代量级**。**注意：文献更倾向用结构化冻结（ControlNet 冻结主干 / LoRA）从根上规避遗忘，而非依赖大规模 replay**——这进一步强化 P2 用路径 B；对 P1，路径 A 全参微调是改分布所必需，但应配回放混合或 KL 锚定。本 repo 无 LoRA 现成支持（§3），故 P1 以防遗忘回放为主。

**回放混合比例先例**（跨领域收敛，无医学影像扩散权威标准）：

| 来源 | 比例 | 结论 |
|---|---|---|
| Continual pre-training of LMs (OpenReview) | **10% 回放**已保留大量知识；50% ≈ 全量上界 | 回放「怎么选」比「选多少」更关键，多样性优先 |
| Experience Replay Face Gen ([arXiv:2412.02627](https://arxiv.org/html/2412.02627)) | buffer **1:1** 时任意方法逼近上界；30% 需精选 | 1:1 是稳健基准 |
| Kazakh LLM 持续预训练 ([arXiv:2503.01493](https://arxiv.org/html/2503.01493v1)) | 新:旧 **1:1 最优** | 等比兼顾新习得与旧保持 |
| Latent Replay T2I ([arXiv:2509.10529](https://arxiv.org/html/2509.10529v1)) | 存 latent 回放 | 随机均匀选取优于相似度选取 |

> 实操：**MR-RATE : BraTS 从 1:1 起步**是文献支持的稳健默认；源数据成本高时 **10–30% 回放仍有显著收益**，优先保证回放多样性（随机均匀抽样即可）。

**Rectified flow 微调注意点**：

- SD3（[arXiv:2403.03206](https://arxiv.org/pdf/2403.03206)）提 logit-normal / mode sampling 偏中段；但 **Improving Rectified Flows**（[NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/7343a5c976f8399880b695267f1f9e9f-Paper-Conference.pdf)）指出 **1-Rectified Flow**（MAISI-v2 即此类）端点预测平凡、有效学习在中段，**logit-normal 或 uniform 均可**。
- 微调实证（ReinFlow [arXiv:2505.22094](https://arxiv.org/html/2505.22094v1)；[arXiv:2411.09998](https://arxiv.org/html/2411.09998)）：对采样分布鲁棒，**uniform 基线在两阶段 curriculum 中最终 FID 最低**。
- **结论**：仓库现配置 `sample_method="uniform", use_timestep_transform=true, scale=1.4`（`config_network_rflow.json:134-141`）与文献不冲突，**微调沿用预训练的 uniform 采样即可，无需改**；小数据避免强偏置采样反而降低分布漂移风险。

---

## 5. P3 前置侦察 —— src→tgt 跨模态条件改造面

当前 ControlNet 条件 = **掩码**（`binarize_labels` → 8 通道）。P3 要改成 **src 影像 latent 作条件**（如 T1→T2 跨序列）。改造面清单：

### 5.1 网络侧（条件输入通道）

- `controlnet_def.conditioning_embedding_in_channels`：现 `8`（掩码 8 位）。src 影像 latent 是 **4 通道**（`latent_channels=4`）→ 改为 `4`（`config_network_rflow.json:85`）。条件嵌入卷积栈 `conditioning_embedding_num_channels=[8,32,64]` 首层相应适配。
- ControlNet 的 `in_channels` 仍 = 4（noisy latent），不变。

### 5.2 训练侧（`scripts/train_controlnet.py` + `scripts/utils.py`）

- 条件构造点：`compute_model_output` 里 `controlnet_cond = binarize_labels(labels...)`（`train_controlnet.py:190`）。改为：**读取 src 影像 latent**（数据 list 增加 `src_image` 字段，经同一 VAE 编码成 4 通道 latent，作为 `controlnet_cond`，可不做 binarize）。
- 数据加载：`prepare_maisi_controlnet_json_dataloader`（`utils.py:261-340`）现只 load `image`/`label` 两 key。需扩展加载 `src_image`（src 序列的 `*_emb.nii.gz`），transform 增加对应 key；JSON schema 增加 src 字段。
- 维度关系：src latent 与 noisy latent 同空间尺寸（同为 latent 网格），无需像掩码那样 4×；条件嵌入栈把 src latent 处理到对应分辨率。
- classifier-free guidance / uncond 分支：现掩码版用「去肿瘤掩码」作 uncond（`train_controlnet.py` region-contrastive 相关，`infer_image_from_mask.py:126-135`）。影像条件版需重新定义 uncond（如零 latent 或 dropout 条件）。

### 5.3 推理侧（已是条件模态无关核心，利好）

- `run_controlnet_conditioned_image_dm` **本身与条件模态无关**——调用方提供 `controlnet_cond_tensor`，仅要求通道数匹配 `conditioning_embedding_in_channels`（`utils_infer.py:110-`，docstring 明确「Conditioning-modality-agnostic」）。
- 掩码版 wrapper `infer_image_from_mask.py` 里掩码专属的 `binarize_labels` 与去肿瘤 uncond（`infer_image_from_mask.py:120-135`）。**P3 只需新增一个影像条件 wrapper**（代码注释已预留 `infer_image_from_image.py` 之名，`infer_image_from_mask.py:100`），内部把 src 影像归一化/编码成 latent 作为 `controlnet_cond_tensor`，复用同一核心。

### 5.4 需改文件清单（P3）

| 文件 | 改动 |
|---|---|
| `configs/config_network_rflow.json` | `conditioning_embedding_in_channels: 8→4`；条件嵌入栈首层通道 |
| `scripts/utils.py` | `prepare_maisi_controlnet_json_dataloader` 增加 `src_image` key 加载 |
| `scripts/train_controlnet.py` | `compute_model_output` 条件构造改为影像 latent；uncond 分支重定义 |
| 新增 `scripts/infer_image_from_image.py` | 影像条件 wrapper，复用 `run_controlnet_conditioned_image_dm` |
| 数据预处理 | src 序列经同一 VAE 编码成 latent；JSON 增加 src 字段 |

### 5.5 官方 brain cross-sequence ControlNet 动态

- 本仓库 README：`rflow-mr-brain` 的 ControlNet 标注 **「Coming soon」**（README.md:102）；但特性列表宣称「Brain MRI synthesis with cross-sequence ControlNet for generating matched multi-contrast brain volumes (T1w, T2w, FLAIR, SWI)」（README.md:31）——官方有此能力规划，尚未发布代码/权重。
- 截至 **2026-08**，NVIDIA 官方仓库（[NVIDIA-Medtech/NV-Generate-CTMR](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR)）模型表中 `rflow-mr-brain` 的 ControlNet 仍标注 **"Coming soon"**，无公开权重；`rflow-mr-brain` 基座 v1（2026-08）发布**不含** brain cross-sequence ControlNet 权重。
- 开发者仓库 [Can-Zhao/NV-Generate-CTMR_MRBrain](https://github.com/Can-Zhao/NV-Generate-CTMR_MRBrain) 含 ControlNet **训练代码/教程**（`train_controlnet_tutorial.ipynb`、`scripts/`），即训练方法已给出，但预训练 brain cross-sequence ControlNet 权重未发布。
- **结论**：P3 不能等官方权重，需按 §5.4 自行改造训练；官方训练代码可作参考实现。
- 模型权重：[nvidia/NV-Generate-MR-Brain](https://huggingface.co/nvidia/NV-Generate-MR-Brain)

---

## 6. 引用与链接

### MR-RATE 回放数据（路径 A 防遗忘）

- 数据集：[Forithmus/MR-RATE](https://huggingface.co/datasets/Forithmus/MR-RATE)（[GitHub](https://github.com/forithmus/MR-RATE)，[获取指南](https://github.com/forithmus/MR-RATE/blob/main/data-preprocessing/docs/dataset_guide.md)）。
- 规模：约 705,254 MRI 体数据 / 98,334 次检查 / 83,425 患者；含 T1w/T2w/FLAIR/SWI/MRA（NV-Generate-MR-Brain 用其中 T1/T2/FLAIR/SWI，模型卡称实际训练约 28,000 scans）。2026-03 新增 37 类 SNOMED/RadLex 病理标签（含肿瘤相关）。
- **获取**：HF **gated**——需登录并同意共享联系信息后下载；主仓库 native-space 约 8.1 TB（按 study 打 zip），官方提供独立 Python 下载脚本与在线 Dataset Explorer。
- **许可注意（重要）**：MR-RATE 数据为 **CC BY-NC-SA 4.0（非商业）**；而 NV-Generate-MR-Brain 模型权重为 NVIDIA Open Model License（可商用）。**若微调产物需商用，用 MR-RATE 回放会引入非商业许可链约束**——回放数据选择需同时权衡防遗忘效果与许可。这是路径 A §1.4 的一个前置决策点。

### 文献（BraTS 微调 / 回放比例 / RF 采样）

- BraTS 3D 生成微调：[arXiv:2403.14499](https://arxiv.org/html/2403.14499v1)、[arXiv:2510.09365](https://arxiv.org/html/2510.09365v1)、[Med-DDPM](https://github.com/JotatD/med-ddpm-brats)、[LDM-seg ISBI2025](https://www2.die.upm.es/im/papers/ISBI2025.pdf)、[arXiv:2602.23533](https://arxiv.org/pdf/2602.23533)、[arXiv:2603.06173](https://arxiv.org/html/2603.06173)
- 回放比例：[arXiv:2509.10529](https://arxiv.org/html/2509.10529v1)、[arXiv:2412.02627](https://arxiv.org/html/2412.02627)、[arXiv:2503.01493](https://arxiv.org/html/2503.01493v1)
- Rectified flow 采样：[arXiv:2403.03206](https://arxiv.org/pdf/2403.03206)、[NeurIPS2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/7343a5c976f8399880b695267f1f9e9f-Paper-Conference.pdf)、[arXiv:2505.22094](https://arxiv.org/html/2505.22094v1)、[arXiv:2411.09998](https://arxiv.org/html/2411.09998)
- 官方 brain ControlNet 状态：[NVIDIA-Medtech/NV-Generate-CTMR](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR)、[Can-Zhao/NV-Generate-CTMR_MRBrain](https://github.com/Can-Zhao/NV-Generate-CTMR_MRBrain)、[nvidia/NV-Generate-MR-Brain](https://huggingface.co/nvidia/NV-Generate-MR-Brain)

### 本仓库代码证据索引

- `scripts/diff_model_train.py:154-171`（load_unet 续训）、`:174-195`（scale_factor 重算）、`:198-223`（optimizer/scheduler）、`:296-341`（RF 目标）、`:369-400`（save ckpt）
- `scripts/train_controlnet.py:292-318`（冻结 DM + copy 初始化 + 读 scale_factor）、`:351-355`（优化器）、`:190`（binarize 条件）、`:454-462`（weighted loss）
- `scripts/utils.py:173-188`（binarize_labels 8 位）、`:261-340`（controlnet dataloader）
- `scripts/utils_infer.py:110-`（条件模态无关核心）、`scripts/infer_image_from_mask.py:100,120-135`（掩码 wrapper + uncond）
- `configs/config_network_rflow.json:39-90`（unet/controlnet def）、`:134-141`（RFlowScheduler）
- `configs/environment_maisi_diff_model_rflow-mr-brain.json:10`（existing_ckpt_filepath）
- `configs/config_maisi_controlnet_train_rflow-mr.json:2-7`（ControlNet 超参）
- `configs/label_dict_ctmr.json:177,347-349`（brain tumor=176；亚区 401-403 超 255）
- `configs/modality_mapping.json`（脑模态 9-32）
- `data/README.md:215-258`（微调示例 §4.3）、`docs/training.md:37,74`（VAE 280ep/58k；256 类上限）
