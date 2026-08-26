# BraTS2023 全子挑战数据普查

> 解决 issue #3（BraTS2023 全子挑战数据普查），隶属父地图 issue #2（BraTS2023 Rectified Flow 微调配方）。
> 产出：数据清单表 + 预处理要求清单 + 获取步骤 + 与本仓 MAISI 管线的 gap 清单 + 240×240×155 网格方案候选。
> 供「BraTS 样例数据获取」（issue #8）与 P2 标签设计（issue #11）使用。
> 所有外部事实均标注来源 URL；本仓事实标注文件路径。

---

## TL;DR（结论速览）

- **BraTS 2023 是「挑战集群」，共 8 个任务，其中 5 个是分割子挑战**：Adult Glioma（GLI）、Sub-Saharan Africa（SSA/BraTS-Africa）、Meningioma（MEN）、Brain Metastases（METS）、Pediatrics（PED）；另有 3 个非分割任务（BraSyn 缺失模态合成、局部 inpainting、augmentation 评测）。**2023 没有 "MENA" 子挑战**（经核 2023/2024/2025 官方清单均无 MENA；这三年唯一的区域性子挑战是 Sub-Saharan Africa，MENA 仅以埃及机构向 METS 供数的形式出现）。issue 中的 "Breast-METS" 实为 **Brain Metastases（BraTS-METS）**——脑转移瘤（原发灶可来自乳腺/肺等），是单个子挑战。
- **标签数值在 5 个分割子挑战间统一为 1/2/3，但语义按子挑战略有差异；2023 年起 ET 由 4 改名为 3，全集群已不使用 label 4**。WT/TC/ET 派生结构一致（WT=1∪2∪3，TC=1∪3，ET=3）。
- **数据格式高度统一**：全部 NIfTI（`.nii.gz`）、SRI24 atlas 空间、1mm 各向同性、已 skull-stripped，且 **5 个分割子挑战每例均严格 240×240×155**（已逐项确认，PED/METS 亦不例外）；**官方不做强度归一化**（留给参赛者）。
- **对本仓的关键结论**：
  - 网格约束比 issue 假设的「4 的倍数」更严——**实为 image 边长需 ÷32**（VAE 4× 下采样 + 4 级扩散 U-Net 要求 latent ÷8）。BraTS 的 240（latent 60，非 ÷8）与 155（latent 38.75，非整数）两轴都需处理。
  - **强度归一化由本仓管线自动完成**（MR 走 0–99.5 百分位 → [0,1]），BraTS 原始强度可直接喂入，无需预归一化。
  - **P2 最大缺口：本仓只发布 CT ControlNet，无 MR 版权重**；CT 专用的 NV-Segment + `add_body_envelope`(200) 伪标签管线不适用于 MR 脑。**但仓内已有 BraTS 脑瘤标签约定**：扩展词表 `label_dict_ctmr.json` 内置 `401=NCR/NETC、402=ED、403=ET`（语义正好对应 BraTS 的 1/2/3），且 `augmentation.py` 有配套肿瘤合成增广 + `controlnet_brats` 实验配置——P2 应基于此脚手架新建 MR ControlNet，而非微调现成权重。

---

## 1. 数据清单表（2023 五个分割子挑战）

> 病例数格式为「训练（带 GT）/ 验证 / 测试」（验证/测试 GT 均不公开）。每个病例 = 4 模态（t1n, t1c, t2w, t2f）+ 1 个 seg（仅训练含 seg）。
> 各子挑战计数、Synapse ID、标签语义均经一手来源（各子挑战论文 + MDPI Sensors 综述 + Synapse + BrainLesion/BraTS 官方仓库）逐项交叉核对。

| 子挑战 | 官方全名 | 训练 / 验证 / 测试 | 每例 4模态+seg | seg 标签(1/2/3) | Synapse ID | 许可/门槛 |
|---|---|---|---|---|---|---|
| **GLI** | Adult Glioma（RSNA-ASNR-MICCAI BraTS 连续评测，术前；数据完全复用 BraTS 2021） | **1251 / 219 / 570**（总 1470 患者/5880 扫描） | 是 | 1=NCR, 2=ED, 3=ET | 父项目 `syn51156910` | CC BY-NC 4.0（非商业）+ Synapse 注册 + data-access 申请审批；禁止再分发 |
| **SSA** | BraTS-Africa（Sub-Saharan Africa 成人胶质瘤，1.5T 低场强、低图像质量） | **60 / 15 / 30** | 是 | 1=NETC, 2=SNFH, 3=ET | `syn51514109` | 同上 |
| **MEN** | ASNR-MICCAI Intracranial Meningioma（术前） | **1000 / 141 / 283**（测试集私有；总 1424） | 是 | 1=NETC, 2=SNFH, 3=ET | `syn51514106` | 同上 |
| **METS** | ASNR-MICCAI Brain Metastasis（治疗前 MRI） | **402 / 31 / 59**（402 含 3076 病灶；另有 474 例可选训练，缺原始 T2） | 是 | 1=NETC, 2=SNFH, 3=ET | `syn51514107` | 同上 |
| **PED** | BraTS-PEDs（CBTN-CONNECT-DIPGR-ASNR-MICCAI，儿童高级别胶质瘤含 DIPG） | **98 / 44 / 24**（挑战口径 167 患者；更大汇编队列 99/45/84，228 患者） | 是 | 1=NC(含囊变), 2=ED, 3=ET | 父项目 `syn51156910` | 同上 |

> **METS 计数口径已澄清**：官方公开训练 = **402 例（3076 病灶）**＝5 机构 238 例（Duke 26 + Cairo 32 + Missouri 16 + WashU 27 + Yale 137）+ NYU 164 例（单独托管）；二手表单里的「238」是不含 NYU 的子集口径，二者非矛盾。
> **网格一致性已确认**：2023 全部 5 个分割子挑战**每例均严格 240×240×155**（PED/METS 已逐项确认；PED 额外做 de-facing 隐私处理）。对比：BraTS-GLI **2024 起**才改为 MNI152 / 182×218×182——勿混用年份数据。
> **PED 计数两口径**：挑战实际评测用 **98/44/24**（结果论文）；**99/45/84** 是含未排名病例的更大汇编队列（设计论文+综述）。引用时注明口径与出处年份。

**关于 "MENA"**：经核 **2023 / 2024 / 2025 官方清单均无 "BraTS-MENA" 子挑战**（对照 BrainLesion/BraTS 官方仓库按年份列出的 task 清单）；这三年唯一的区域性子挑战都是 **BraTS-Africa（Sub-Saharan Africa，SSA）**。MENA 区域仅以埃及一家机构向 METS 贡献数据的形式出现。故 issue 中列举的 "MENA" 不纳入本普查（按非 2023 处理；若未来确需，再核实其归属年份）。

来源：

- 集群结构（2023 = 5 分割 + 3 非分割任务）、各子挑战标签映射、ET 4→3、计数：MDPI Sensors 综述《A Comprehensive Review of MRI Segmentation Datasets from BraTS Challenges (2012–2025)》§4.10 — <https://www.mdpi.com/1424-8220/25/6/1838> （PMC 版：<https://pmc.ncbi.nlm.nih.gov/articles/PMC11945730/> ）
- GLI（1251/219/570、复用 BraTS 2021、SRI24、1mm、240×240×155、skull-stripped）：BraTS 2023 GLI — <https://pmc.ncbi.nlm.nih.gov/articles/PMC10441440/> ；BraSyn 论文（注明复用 2021 数据 + ET 4→3）— <https://arxiv.org/html/2305.09011v6>
- SSA / BraTS-Africa（60/15/30、NETC/SNFH/ET）— <https://arxiv.org/abs/2305.19369> ；PMC <https://pmc.ncbi.nlm.nih.gov/articles/PMC10312814/>
- MEN（1000/141/283、syn51514106、TC/WT 定义）— <https://arxiv.org/html/2405.09787v2> ；MELBA <https://www.melba-journal.org/papers/2025:003.html>
- METS（402/31/59、3076 病灶、238 vs 402 口径、syn51514107、WT/TC/ET 公式、ET 4→3 声明）— <https://arxiv.org/html/2306.00838v3> ；PMC <https://pmc.ncbi.nlm.nih.gov/articles/PMC10312806/>
- PED（挑战口径 98/44/24：结果论文 <https://arxiv.org/html/2407.08855> ；大队列 99/45/84、228 患者、NC 含囊变：设计论文 <https://arxiv.org/html/2305.17033v7> ）
- 获取与许可：Synapse 门户 <https://www.synapse.org/Synapse:syn51156910> ；CC BY-NC + 需先完成官方 data-access 申请 — <https://arxiv.org/html/2607.22135v2>
- MENA 不属于 2023/2024/2025（BrainLesion/BraTS 官方仓库按年份列 task）— <https://github.com/BrainLesion/BraTS>
- 社区事实表（辅助核对）：<https://github.com/openmedlab/Awesome-Medical-Dataset> （BraTS2023-SSA / -MET / -PED 等资源页）

---

## 2. 数据格式事实（与 issue 三问对应）

| 维度 | 事实 | 来源 |
|---|---|---|
| 文件格式 | 全部 **NIfTI（`.nii.gz`）**；每例 5 个文件：`{id}-t1n.nii.gz`（native T1）、`-t1c`（增强 T1/T1Gd）、`-t2w`（T2）、`-t2f`（T2-FLAIR）、`-seg`（GT，仅训练集） | <https://www.mdpi.com/1424-8220/25/6/1838> |
| 解剖空间 | 统一**刚性配准到 SRI24 atlas** | 同上 |
| 分辨率 | **1mm 各向同性**（1×1×1 mm）重采样 | 同上 |
| 网格 | **240×240×155** voxels | 同上 |
| skull-strip | **已 skull-stripped**（DL 方法，兼作去标识化防面部重建） | 同上 |
| 方向 | 配准到 SRI24（社区常以 LPS 描述；本仓管线会再 `Orientationd(..., "RAS")` 统一） | 本仓 `scripts/diff_model_create_training_data.py:58` |
| **强度归一化** | **官方不做**任何强度归一化/标准化——提供的是 skull-stripped + 配准 + 重采样后的**原始强度**；z-score 等由参赛者自行施加（惯例：脑掩膜内、逐模态 z-score） | <https://www.mdpi.com/1424-8220/25/6/1838> 及多处文献 |

> **对本仓的含义**：BraTS 原始强度可直接喂入本仓 embedding 脚本——脚本按 JSON 里的 `modality` 字段自动做 MR 强度变换：`ScaleIntensityRangePercentilesd(lower=0, upper=99.5, b_min=0, b_max=1)`（0–99.5 百分位 → [0,1]，**非 z-score**），见 `scripts/transforms.py:64`。**为对齐 base 模型 `rflow-mr-brain v1` 的训练分布，应采用本仓自己的百分位归一化，而不要预先对 BraTS 做 z-score**（那会与 base 模型训练时用的归一化不一致）。

### 分割标签语义（issue 核心问题：「2023 的第 4 类标签」）

**结论：2023 全集群不再使用 label 4。** 自 2023 起，增强肿瘤 ET 的标签值由 **4 改为 3**；NCR/NET(1)、ED(2) 不变。各子挑战的**数值**统一为 1/2/3，但**语义命名**按子挑战微调：

| label | GLI（成人胶质瘤） | SSA / MEN / METS | PED（儿童） |
|---|---|---|---|
| 0 | 背景 | 背景 | 背景 |
| 1 | NCR（坏死肿瘤核心） | **NETC**（非增强肿瘤核心） | **NC**（非增强成分 = 非增强肿瘤+囊变+坏死） |
| 2 | ED（瘤周水肿/浸润） | **SNFH**（瘤周非增强 FLAIR 高信号） | ED（瘤周水肿） |
| 3 | ET（增强肿瘤，**由 4 改来**） | ET（增强肿瘤） | ET（增强肿瘤） |

- **WT/TC/ET 派生结构在 5 个分割子挑战一致**（METS/PED/MEN 论文均有逐字公式）：WT = 1∪2∪3（全肿瘤）；TC = 1∪3（肿瘤核心）；ET = 3（增强肿瘤）。即无论各子挑战对 1/2 的具体命名如何（NCR/NETC/NC、ED/SNFH），派生规则相同。
- **最实质的语义差异在 PED**：其 label 1（NC）把**囊变成分（cystic component, CC）**并入非增强成分（原始 4 亚区 ET/NET/CC/ED 合并为 3），而成人 GLI 的 label 1（NCR）不含独立囊变标注。混训 PED 与 GLI 时 label 1 的解剖含义略有偏移（对生成建模影响有限，但做精确亚区条件时需注意）。label 2（ED vs SNFH）与 label 3（ET）各子挑战语义基本对应，仅命名不同。
- **数据准备实战 gotcha**：若把 ≤2022 旧 BraTS（ET=4）与 2023（ET=3）混用，ET 索引会错位；本普查只用 2023，统一按 1/2/3 处理即可。
- 来源：MDPI 综述 §4.10 — <https://www.mdpi.com/1424-8220/25/6/1838> ；METS（含 ET 4→3 逐字声明 + WT/TC/ET 公式）<https://arxiv.org/html/2306.00838v3> ；PED（NC 含囊变）<https://arxiv.org/html/2407.08855>

---

## 3. 获取渠道、门槛与许可

**主渠道：Synapse 平台**（<https://www.synapse.org/）。BraTS> 2023 全集群统一经 Synapse 分发，门户父项目 `syn51156910`；各子挑战下载入口见 §1 表（SSA `syn51514109` / MEN `syn51514106` / METS `syn51514107`；GLI 与 PED 经父项目子页面）。来源：<https://www.synapse.org/Synapse:syn51156910>

- **门槛**：注册 Synapse 账号 → 在官方门户完成 **data-access application / 接受 DUA（Data Use Agreement）** → 组织者审批通过后方可下载。审批为人工/机构流程，**非即时**，执行 issue #8 时应预留数天审批时间。
- **许可**：**CC BY-NC 4.0（非商业研究用途）**，**禁止再分发**（每位使用者须自行完成申请，不可替代签署）。本项目据此产出（embeddings、合成数据、下游模型）的对外再分发同样受限。

**其他渠道**：

- **CBICA / UPenn**（成人胶质瘤 GLI 的历史分发口，CaPTk/IPP）。来源：<https://www.med.upenn.edu/cbica/captk/>
- **Kaggle 社区镜像**（如 ASNR-MICCAI BraTS 2023 GLI 训练数据）——下载快但非官方，须注意合规。来源：<https://www.kaggle.com/datasets/luumsk/asnr-miccai-brats-2023-gli-challenge-training-data>
- **Hugging Face**：**无官方 BraTS 数据集**。存在社区镜像——`MedOtter/brats2023-gli-dataset`、`MedOtter/brats2023-men-dataset`、`MedOtter/brats2023-ssa`、`MedOtter/brats2023-ped-dataset`（标称 CC-BY-4.0）、`obi77/brats23-first-10-examples`（前 10 例小集）、`sohonjit/brats2023_multidomain_i2i`（仅 2D 中间层切片，非完整体积）。
  - ⚠️ **合规警示**：BraTS DUA 明确禁止再分发，上述 HF/Kaggle 镜像很可能**不合规**。**正式获取（尤其用于本项目交付）应走 Synapse + 自有 DUA**；HF 的 `obi77/brats23-first-10-examples` 之类仅可作一次性冒烟参考，且须先确认许可，不应作为正式数据来源。

**获取步骤（建议，供 issue #8）**：

1. 注册 Synapse 账号，定位 BraTS 2023 门户（`syn51156910`）与目标子挑战页面。
2. 提交 DUA，等待审批（预留数天）。
3. 用 Synapse CLI/客户端批量下载目标子挑战训练集（含 seg）。
4. 抽样核对：每例是否 5 文件齐全、网格是否 240×240×155、seg 标签值是否 ∈ {0,1,2,3}。
5. 进入预处理管线（见 §4 gap 清单）。

---

## 4. 与本仓 MAISI 管线格式的 gap 清单

> 参照 `data/README.md` 的微调格式（P2 ControlNet 见 §4.3；P1 image-only 扩散见 §3.x + `scripts/diff_model_train.py`）。issue 文中写「§3.3 微调示例」，实际「`*_emb.nii.gz` + combined label + JSON data list」对应 **§4.3**；两阶段格式我都列出。

### 4.0 预处理要求清单（有序步骤，供 issue #8 落地）

以单个子挑战训练集为例，把 BraTS 原始发布处理成 MAISI 可训练格式的有序步骤：

0. **获取与核验**：Synapse 下载（见 §3）；核对每例 5 文件齐全、网格 240×240×155、seg 值 ∈ {0,1,2,3}。
1. **（仅 P2）标签重映射**：BraTS seg 的 1/2/3 在 MAISI 词表无对应，必须重映射——推荐 1→401 / 2→402 / 3→403（`label_dict_ctmr.json` 语义天然对应，见 §4.3 路线 R1），或 released 0–255 空闲码（R2）；脑背景可用 `brain=22`。
2. **逐 (病例×模态) 生成 VAE embedding**：`scripts/diff_model_create_training_data.py` + `autoencoder_v1.pt`。**网格处理见 §5**（默认 resize 到 128 倍数 vs 推荐 pad 到 256×256×160）；脚本按 `modality` 自动做 MR 百分位归一化（0–99.5→[0,1]），**无需对 BraTS 预归一化**。
3. **标注 modality**：每 (病例×模态) 标注 t1n/t1c/t2w/t2f → `mri_t1=9 / mri_t1c=17 / mri_t2=10 / mri_flair=11`（注意 t1c skull-stripped 无专用码，见 G4）。
4. **生成 P1 侧 sidecar `<emb>.json`**（含 spacing + modality）——embedding 脚本**不写**此文件，需自建（见 G2）。
5. **（仅 P2）combined label**：把重映射后的肿瘤掩码叠到脑掩码（+背景）上，最近邻重采样到 **4× latent** 网格（见 §4.2 / §5）。
6. **构建 JSON data list**：P1 = 主 JSON 列原始影像路径 + 逐例 sidecar；P2 = 单文件内联 `image`(emb)+`label`(combined)+`spacing`+`dim`+`modality`+`fold`。注意 fold 切分（默认 fold 0 = 验证）。

### 4.1 目标格式速查

**P1（image-only 扩散微调，`diff_model_train.py`）**——每个 (病例 × 模态) 一条样本：

- 主 JSON `training` 列表：`image` 指向**原始**影像相对路径（脚本自动把 `.nii.gz` 换成 `_emb.nii.gz`），见 `scripts/diff_model_train.py:82`。
- **每例 sidecar `<emb>.nii.gz.json`**：含 `spacing`（3 floats）、`modality`（字符串）；若为 ddpm 还需 `top/bottom_region_index`。加载器从 sidecar 读取（`_load_data_from_file`），见 `scripts/diff_model_train.py:450-456,120-133`。
- **无需 `label`**（image-only）。`modality` 经 `modality_mapping.json` 转 int，作为 U-Net 的 class 条件（`num_class_embeds=128`），训练时还有 modality dropout（CFG）。

**P2（肿瘤掩码→影像 ControlNet，`train_controlnet.py` / §4.3）**——每个病例一条样本：

- 单文件内联 JSON：`image`=`*_emb.nii.gz`、`label`=`mask_combined_label*.nii.gz`、`spacing`、`dim`（informational）、`modality`、`fold`；`top/bottom_region_index` 仅 ddpm-ct 需要（rflow 忽略）。
- combined label 的实际体素网格必须 = **4× latent**（最近邻重采样，勿用线性），见 `skills/finetune_image-from-mask_data-prep.md` Step 4。
- `fold` 切分：等于配置 `fold` 者作验证，其余训练（默认 fold 0=验证），见 `data/README.md` §4.3。

### 4.2 gap 清单（BraTS 现状 → 需补齐）

| # | gap | 说明 / 影响 | 严重度 |
|---|---|---|---|
| G1 | **VAE embedding 未生成** | BraTS 给原始 NIfTI；需用 `scripts/diff_model_create_training_data.py` + `autoencoder_v1.pt` 逐 (病例×模态) 生成 `*_emb.nii.gz`。注意该脚本默认把每轴 **resize 到 128 的倍数**（trilinear），见 §5 网格讨论——可能需改造。 | 必有工序 |
| G2 | **P1 per-embedding sidecar JSON 缺失** | embedding 脚本**只写 `_emb.nii.gz`，不写 sidecar** `<emb>.json`（spacing+modality）。需自建一步生成 sidecar，否则 P1 训练读不到 spacing/modality 条件。 | 必补 |
| G3 | **modality 字段需逐 (病例×模态) 标注** | embedding 脚本读内联 `modality`（驱动强度归一化）；P1 训练还从 sidecar 读。BraTS 每例 4 模态，需把 t1n/t1c/t2w/t2f 映射到 `modality_mapping.json` 的码：`mri_t1=9 / mri_t1c=17 / mri_t2=10 / mri_flair=11`。 | 必补 |
| G4 | **skull-stripped modality 码不全（P1 条件错位风险）** | BraTS 已 skull-stripped，但 `modality_mapping.json` 的 skull-stripped 码只有 `mri_t1_skull_stripped=29 / mri_t2=30 / mri_flair=31 / mri_swi=32 / mri_mra=33`，**没有 `mri_t1c_skull_stripped`**。增强 T1（肿瘤信息最关键）只能用非 skull-stripped 码 17 或借用 29，存在条件语义错位。是否新增码属 P1 配方（issue #10）决策，此处仅标记。 | 需决策 |
| G5 | **P2 无已发布的 MR ControlNet checkpoint** | 本仓**只发布 CT ControlNet**；无 MR 版权重，`finetune_image-from-mask_data-prep` skill 也标注 CT-only。**但仓内有 MR 脑瘤 ControlNet 脚手架**（见 §4.3）：`configs/environment_maisi_controlnet_train_rflow-mr.json`（`exp_name:"controlnet_brats"`，用 MSD Task01_BrainTumour + `autoencoder_v2.pt` + 通用 `diff_unet_3d_rflow-mr.pt`）+ `augmentation.py` 的 brats 肿瘤合成增广。即 P2 需**基于脚手架训 MR ControlNet**，而非微调现成 MR 权重。属 issue #11 配方范畴。 | 架构级 |
| G6 | **P2 伪标签管线不适配 MR 脑，但仓内已有 BraTS 标签约定** | CT 流程的 NV-Segment（CT 器官分割）+ `add_body_envelope`(CT HU→200) 对 MR 脑**不适用**。但仓内扩展词表已内置脑瘤亚区标签（见 §4.3），脑条件可用 `brain=22`，肿瘤三亚区有专用码。BraTS seg 的 1/2/3 如何映射见 §4.3 的两条路线（待 issue #11 决策）。 | 架构级 |
| G7 | **网格不匹配（240×240×155）** | 见 §5。155 非 ÷4，240 的 latent(60) 非 ÷8。需 pad/crop/resize。 | 必处理 |
| G8 | **spacing 分布偏移** | base MR-brain 训练分布约 0.94×0.94×1.36mm（见 `configs/config_maisi_diff_model_rflow-mr-brain.json`），BraTS 是 1mm 各向同性。DM 有 `include_spacing_input:true`（spacing 作条件），故可传 [1,1,1]；但 1mm iso 相对训练分布略 off-distribution，属 issue #4（分布差距）范畴。 | 提示 |
| G9 | **许可约束须写入数据卡** | BraTS research-only、禁再分发；本项目产出（如 embeddings）的再分发也受限。落盘管线需记录来源与 DUA。 | 流程 |

### 4.3 关键发现：本仓已内置 BraTS 脑瘤标签约定（两本词表）

排查中发现本仓存在**两本标签词表**，对 P2 标签设计影响重大：

- **`configs/label_dict.json`（released，默认）**：132 类、0–255；`brain=22`、`body=200`、dummy 空槽 129/130/131 等；**无脑瘤亚区**。released 环境配置（`environment_rflow-mr.json` / `environment_ddpm-ct.json` / `environment_rflow-ct.json`）默认都引用它。README §4.3 的「0–255、dummy 槽」说法对应这本。
- **`configs/label_dict_ctmr.json`（扩展 CTMR 词表）**：349 类、最大 500；在 released 词表基础上新增脑亚结构（Brain-Stem=220、temporal lobe 178/179、chiasm=177、eyeball 174/175 等）与**脑瘤亚区**——
  - `brain tumor: 176`（generic）
  - **`brain tumor NCR/NETC: 401`**、**`brain tumor ED: 402`**、**`brain tumor ET: 403`**
  - 注意：**此词表的 body envelope 是 `body=500`**（非 released 词表的 200）。

**配套脚手架已存在**：

- `scripts/augmentation.py` 的 `augmentation()` 有专门 `augmenting brats tumor` 分支，按 **401/402/403** 做肿瘤弹性形变合成；`remove_tumors()` 把 401/402/403/176 映射回 `brain=22`（healthy 路径）。见 `scripts/augmentation.py:455-470,506,519`。
- `configs/environment_maisi_controlnet_train_rflow-mr.json`（`exp_name:"controlnet_brats"`）是 NVIDIA 内部 MR 脑瘤 ControlNet 实验配置：数据用 MSD `Task01_BrainTumour_t1`、`autoencoder_v2.pt`、通用 `diff_unet_3d_rflow-mr.pt`（**非** brain_v1）、`weighted_loss_label:[129]`。⚠️ 该配置自我矛盾（用 401/402/403 增广却 weighted_loss 129），是**未完成的脚手架**，仅供参考。

**对 P2（BraTS mask→image）标签设计的两条候选路线**（最终取舍属 issue #11）：

| 路线 | BraTS 1/2/3 映射 | 用词表 | 优点 | 风险 |
|---|---|---|---|---|
| **R1 扩展 CTMR 约定** | 1→**401**、2→**402**、3→**403**（core/edema/enhancing 语义天然对应） | `label_dict_ctmr.json` | 语义预定义、可直接复用 401/402/403 肿瘤增广、贴合 NVIDIA 内部 brats 实验 | 标签 >255，需确认 CTMR ControlNet 支持扩展词表；无现成 MR 权重；body=500 与 released 不一致 |
| **R2 released 0–255 约定** | remap 到空闲码（如 150/151/152 或 dummy 129/130/131） | `label_dict.json` | 与 released「0–255」条件一致、简单 | 语义需自定义；401/402/403 增广分支不会触发（除非再映射）；需自备脑掩码 22 |

> **数据侧结论**：无论 R1/R2，BraTS seg（1/2/3）都必须**重映射**——MAISI 词表里没有 1/2/3 对应的脑瘤亚区（`label_dict.json` 的 1=liver、2=dummy1、3=spleen，直接喂会语义错乱）。**推荐优先评估 R1（401/402/403）**，因其语义与增广脚手架均已就位；但该选择牵涉 P2 用哪本词表/哪个 base DM，属 issue #11 决策。来源：本仓 `configs/label_dict_ctmr.json:347-350`、`scripts/augmentation.py`、`configs/environment_maisi_controlnet_train_rflow-mr.json`。

---

## 5. 240×240×155 网格方案候选

**约束来源（本仓架构事实）**：

- VAE（`AutoencoderKlMaisi`，`num_channels=[64,128,256]`，3 级 → **4× 空间下采样**，`latent_channels=4`）：image 边长须 **÷4** 才能得整数 latent。155÷4=38.75 ✗；240÷4=60 ✓。见 `configs/config_network_rflow.json`。
- 扩散 U-Net（`DiffusionModelUNetMaisi`，`num_channels=[64,128,256,512]`，4 级 → 3 次下采样）：latent 边长须 **÷8** 以保证编/解码跳跃连接尺寸匹配。即 image 边长须 **÷(4×8)=÷32**。240→latent 60，60÷8=7.5 ✗。
- **所有已发布配置 latent 均为 8 的倍数**（512→128、256→64、128→32，皆 ÷8），对应 image 边长为 128 的倍数（`round_number` 按 128 取整，`scripts/diff_model_create_training_data.py:77-90`）。MONAI 对非整除输入的惯用解法是 `DivisiblePad`。
- base 模型 `rflow-mr-brain v1` 的原生推理网格 = **256×256×128**（latent 64×64×32，spacing 0.94×0.94×1.36），见 `configs/config_maisi_diff_model_rflow-mr-brain.json`。

> 因此 **issue 假设的「÷4」只是 VAE 层的必要条件，不充分**；要同时满足 4 级扩散 U-Net，实务上应瞄准 **image 边长 ÷32**。（该 ÷8 latent 约束建议在原型票 issue #9 中实证确认 MONAI 是否内部容错。）

**候选方案**（针对 240×240×155）：

| 方案 | 目标网格 | latent | 做法 | 优点 | 代价/风险 |
|---|---|---|---|---|---|
| **A. 沿用默认脚本**（推荐作为零改动基线） | 256×256×128 | 64×64×32 | 直接用 `diff_model_create_training_data.py` 的「128 倍数 + trilinear resize」 | 零改动、与 released 管线一致、latent 规整 | 各向异性失真：155→128 严重压缩（z 向）、240→256 拉伸；改变有效 spacing，肿瘤形状/位置分布被扭曲（影响验收的「尺寸/位置分布对齐」） |
| **B. 就近 pad 到 ÷32**（推荐作为保真首选） | **256×256×160** | 64×64×40 | 面内 240→256 对称 pad 16；z 向 155→160 pad 5（`DivisiblePad`/常量 pad） | 保留全部 155 层解剖、近原生分辨率、几乎无失真、latent 全 ÷8 | 需自定义 pad（绕过默认 resize）；需实证 64×64×40 latent 下 U-Net 正常 |
| **C. 就近 crop 到 ÷32** | 224×224×128 | 56×56×32 | 中心裁剪 | latent 规整、无需 pad | 裁剪有截断肿瘤/颅骨风险，**肿瘤生成场景不推荐** |
| **D. 仅 z 向 ÷4，面内不动** | 240×240×156（或 ×152） | 60×60×39（或 ×38） | 只把 155 pad/crop 到 4 倍数，面内保 240 | 改动最小、面内零失真 | **latent 39/38 非 ÷8**，4 级 U-Net 很可能报错或需内部 pad——**仅在实证 DM 容忍非 ÷8 latent 后才可用** |

**建议**：默认以 **B（pad 到 256×256×160）** 为保真首选、**A** 为零改动回退；C 排除；D 待 issue #9 原型实证后再评估。最终网格选择建议与「肿瘤尺寸/位置分布对齐」的验收层（issue #13）联动确认——A 的 z 向压缩会系统性改变肿瘤表观尺寸。

---

## 6. 给下游票的交接

- **issue #8（样例数据获取）**：用 §3 获取步骤（Synapse + DUA，预留审批期）；优先拉 GLI 训练集子集做冒烟；拿到后先核对 §1 的计数与 240×240×155 网格、seg 标签值 ∈{0,1,2,3}。
- **issue #11（P2 标签设计）**：依据 §2 标签表（1/2/3 统一、语义按子挑战、无 label 4）+ §4.3 的两本词表发现。核心取舍：BraTS seg 必须重映射，候选 **R1（→401/402/403，复用仓内 brats 增广与语义）** vs **R2（→released 0–255 空闲码）**；P2 需基于 `controlnet_brats` 脚手架新建 MR ControlNet，条件为「肿瘤亚区掩码 + 脑掩码（brain=22）」，不套用 CT 的 NV-Segment+`add_body_envelope` 管线。
- **issue #9（VAE 重建冒烟）**：实证 §5 的 ÷8 latent 约束（方案 B vs D），并验证 MR 百分位归一化对 BraTS 的重建质量。
- **issue #4（分布差距）**：参考 G8（spacing 偏移）与 G4（t1c skull-stripped 条件码缺失）。

---

## 附：本仓关键文件锚点

- 数据格式权威文档：`data/README.md`（§3.4 base 模型数据、§4.3 ControlNet 微调示例）
- VAE/扩散/ControlNet 网络定义与级数：`configs/config_network_rflow.json`
- MR 强度归一化（0–99.5 百分位→[0,1]）：`scripts/transforms.py:64`
- embedding 生成 + 128 倍数 resize：`scripts/diff_model_create_training_data.py:77-90,174-191`
- P1 训练数据清单（sidecar、modality 条件、spacing×100）：`scripts/diff_model_train.py:82,120-133,440-471`
- P2 ControlNet 数据清单与 fold：`scripts/train_controlnet.py`、`skills/finetune_image-from-mask_data-prep.md`
- modality 码表：`configs/modality_mapping.json`；label 词表（含 brain=22、dummy 槽、body=200）：`configs/label_dict.json`
