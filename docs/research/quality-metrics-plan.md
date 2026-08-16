# 三层质量验收指标方案调研（issue #7）

> 父地图：issue #2「BraTS2023 Rectified Flow 微调配方」。本文是 research 票 #7 的 findings，供后续「三层验收标准与持出集钉板」grilling 票与 ADR 使用。
>
> **适用范围**：基座模型 `rflow-mr-brain v1`（MR-RATE 训练），在 BraTS2023 全部子挑战上做 rectified-flow 微调（P1 image-only 带肿瘤 / P2 掩码→影像 / P3 跨模态）。质量 = 三层验收：**定量指标 + 肿瘤尺寸/位置分布对齐 + 人工目检**；下游分割器验证仅可选加分、不设门槛（地图 #2 已定约束）。

---

## 0. 三层验收总览与「够用最小集」

| 层 | 验收什么 | 最少必算指标 | 一件工具 | 通过线（建议） |
|----|----------|--------------|----------|----------------|
| **L1 定量** | 合成 vs 真实的整体特征分布距离 | **2.5D FID**（RadImageNet-ResNet50，MAISI 协议）+ **KID/bootstrap CI** 校正样本量偏倚 | 改造版 `scripts/compute_fid_2-5d_ct.py` → `compute_fid_2-5d_mr.py` | 各平面/平均 FID ≤ 「真实训练集 vs 真实持出集自比 FID」的 k 倍（k≈2–3 标定）；KID/CI 与真实-真实基线重叠 |
| **L2 分布对齐** | 肿瘤/器官**尺寸、位置、强化比例、分区强度**与真实分布一致 | **WT/ET 体积分布（KS + 1D Wasserstein）** + **ET/WT 强化比（TOST 等价）** + **centroid 解剖分区（卡方）** + **分区强度离群率** | 新增 `scripts/tumor_stats.py` + 复用改造 `scripts/quality_check.py` | 主判据 **TOST 等价（90% CI ⊂ ±0.2–0.5×SD）+ bootstrap CI**；KS p>0.05 仅作初筛 |
| **L3 人工目检** | 解剖合理性、肿瘤真实性、伪影（FID 盲区） | **real/synthetic 视觉图灵准确率** + **4 维 5 分 Likert** | 评分表 + 分层抽样清单（§3.3） | 视觉图灵准确率 ≈50%（接近随机）；各维 Likert 均值 ≥4（"倾向真实"）；Fleiss' κ 报告 |

**设计原则（防指标海啸）**：每层只留 1–3 个「能定通过线、能驱动 go/no-go 决策」的指标；其余（MS-SSIM、PSNR、LPIPS、3D-FID、核 MMD、FRD/RaD…）一律列为**可选诊断项**，不进门槛。依据见各节「坑/权衡」。

---

## 1. L1 定量：现有 FID 工具分析与 MR 适配

### 1.1 `compute_fid_2-5d_ct.py` 现状

repo 内**唯一的分布距离工具**（已全仓 grep 确认：无 MR 版 FID，tutorials 也不含 FID）。管线：

1. **特征提取器**：`--model_name`，默认 `radimagenet_resnet50`（`torch.hub.load("Warvito/radimagenet-models")`，`compute_fid_2-5d_ct.py:498-499`），备选 `squeezenet1_1`（torchvision ImageNet，`:503`）。
   - **RadImageNet ResNet50 不是 CT 专用**：RadImageNet 含 135 万张 CT/MRI/超声多模态影像，ResNet50 为 2D 网络、输出 2048 维特征，是 MAISI/MONAI 3D 评估事实标准，**可直接用于 MR**。证据：<https://github.com/Warvito/radimagenet-models>、<https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/>。
2. **2.5D 切片**：3D 体数据沿 XY(轴)/YZ(矢)/ZX(冠) 三正交面逐片切开，每片过 2D 网络提特征、`spatial_average` 全局空间平均成向量，逐平面算 FID 再三平面取平均（`get_features_2p5d` `:250-349`；聚合 `:710-720`）。单通道复制成 3 通道（`:279-280`）。
3. **FID 计算**：MONAI `FIDMetric`（`:115`,`:710`）。**注意：`FIDMetric`/`get_fid_score(y_pred,y)` 只吃已提取好的 `(N, feat)` 特征向量，不含提取器**；非 1D 特征须先全局空间平均（本脚本正是如此）。证据：<https://monai.readthedocs.io/en/1.3.0/_modules/monai/metrics/fid.html>。
4. **预处理（CT 专用，需改）**：`ScaleIntensityRanged(a_min=-1000,a_max=1000,b_min=-1000,b_max=1000,clip=True)`（`:570`，HU 窗）；`SpatialPadd(value=-1000)`（`:565`，CT 空气值填充）；`empty_threshold=-700`（`:258`，HU 判空，`drop_empty` 默认关 `:257`）。

### 1.2 MR 适配清单（最小改动）

**特征网络与 2.5D 三平面协议都不用动**（RadImageNet 覆盖 MR；2.5D 是 MAISI 默认、与社区可比）。只改 **CT 强度语义 → MR 强度语义** 三处，方向直接对齐训练时的 MR 归一化（`scripts/transforms.py:64`：MR 用 `ScaleIntensityRangePercentilesd(lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False)`；CT 才用固定 HU 窗 `transforms.py:65`）：

| 位置 | CT 现状 | MR 改为 | 依据 |
|------|---------|---------|------|
| 强度归一化 `:570` | `ScaleIntensityRanged(-1000,1000→-1000,1000)` | `ScaleIntensityRangePercentilesd(lower=0, upper=99.5, b_min=0, b_max=1, clip=False)` | 与训练一致（`transforms.py:64`）；MR 无 HU |
| 填充值 `:565` | `value=-1000`（CT 空气） | `value=0`（MR [0,1] 背景≈0） | MR 归一化后背景≈0 |
| 判空阈值 `:258` | `empty_threshold=-700`（HU） | 小正数如 `0.01`（[0,1] 下空片 max≈0），或保持 `drop_empty=False` | HU 阈值对 [0,1] MR 无意义 |

> **结论**：MR 适配 = 新建 `compute_fid_2-5d_mr.py`（或给现脚本加 `--modality mr` 分支），仅改上述三处预处理；特征网络继续 `radimagenet_resnet50`，2.5D 三平面协议保持（与 CT 结果、与 MAISI 社区可比）。因模型按模态条件生成（`configs/modality_mapping.json`：mri_t1=9/t2=10/flair=11/t1c=17/swi=20…），**FID 应按模态分层各算一份**，而非全模态混在一起。

### 1.3 定量层补充指标（可选诊断，不进门槛）

- **KID 或 bootstrap FID 置信区间**：FID 有偏、小样本不稳；KID 无偏，或对 FID 做 bootstrap 得 CI。**建议进最小集**作 FID 的样本量校正。证据：<https://arxiv.org/html/2510.22070v1>。
- **MMD**：MONAI 内置 `MMDMetric`/`compute_mmd(y,y_pred)`，**但只实现线性核（`torch.mm`）、无 RBF/高斯核、无可调 sigma**——实质是均值差代理，对高阶分布差不敏感；要核 MMD 需自实现。列为可选。证据：<https://monai.readthedocs.io/en/1.3.0/_modules/monai/metrics/mmd.html>。
- **FDD / 域内特征距离**：FRD（Frechet Radiomic Distance，用标准化 radiomics 特征替代深度特征，MedIA 2026）与 RaD——OOD 检测/小样本更稳、可解释、支持 3D NIfTI。列为可选交叉校验。证据：<https://pmc.ncbi.nlm.nih.gov/articles/PMC13224392/>、<https://arxiv.org/html/2412.01496v1>、<https://github.com/RichardObi/frd-score>。
- **真 3D 特征 FID（MedicalNet 3D ResNet，单通道）**：补 2.5D 漏掉的层间一致性；conv1 权重形状 `[64,1,7,7,7]`，可直接吃单通道 MRI。列为可选（层间伪影更多靠 L3 目检）。证据：<https://github.com/Tencent/MedicalNet>、<https://github.com/Project-MONAI/MONAI/issues/2836>。

### 1.4 FID 在脑 MR 上的已知坑（决定它不能单独当门槛）

- **对解剖失真不敏感**：即使脑区宏观形态明显不真实，FID/MMD 也可能显示「相似」（ResNet/Inception 抓不住脑区解剖）→ 必须用 L3 目检 + L2 解剖/肿瘤统计补盲。证据：<https://arxiv.org/html/2409.08463v1>。
- **特征选择与人类判断可能不一致**：Woodland 等（MD Anderson, MICCAI 2024）在 16×StyleGAN2 × 4 模态 × 11 提取器上发现 **ImageNet 自监督特征（SwAV/FSD）排名更一致、与专家图灵测试显著相关，RadImageNet-FD 反而波动大** → 保留 RadImageNet 为主（可比性），必要时用 ImageNet-SwAV 交叉校验。证据：<https://arxiv.org/html/2311.13717v5>、<https://github.com/mckellwoodland/fid-med-eval>。
- **强度窗敏感**：归一化选择直接改 FID 数值 → 必须固定 §1.2 方案并写进验收 SOP，否则不可复现。
- **记忆化风险**：扩散模型可能复现训练样本 → 需另做最近邻相似度排查（见 §2.4）。

**L1 通过线建议**：拒绝绝对阈值（随数据集/归一化漂移）。改用**相对基线**——先算「真实训练集 vs 真实持出集」的 FID 作基线，合成 vs 真实持出集的 FID ≤ 基线的 k 倍（k≈2–3，由 grilling 钉板），且 KID/bootstrap CI 与基线重叠。

---

## 2. L2 肿瘤/器官分布对齐

### 2.1 为什么尺寸/位置分布是第一类验收轴

MAISI 掩码生成**本身就以 10 维 `anatomy_size` 向量为条件**（`scripts/sample_mask.py:16,52`），并区分 `available_controllable_tumor` 与 `available_controllable_organ`（`sample_mask.py:237-239`）——**器官/肿瘤尺寸是模型显式可控的条件维度**。故「生成体的肿瘤/器官尺寸分布是否与真实一致」直接检验模型是否学到正确尺寸先验，是天然验收轴。

**BraTS 标签双套对应**（算体积/位置前必须统一）：
- **原生 BraTS seg 标签**：0=背景，1=NCR/NETC（坏死/非强化核心），2=ED/SNFH（水肿/FLAIR 高信号），3=ET（强化肿瘤）；**2023 起 ET 由 4 改为 3**。子区 **WT=1+2+3、TC=1+3、ET=3**。成人胶质瘤用 NCR/ED，脑膜瘤/转移瘤改叫 NETC/SNFH（仅命名不同）。证据：<https://www.synapse.org/Synapse:syn51156910/discussion/threadId=10189>、<https://arxiv.org/pdf/2305.07642>。
- **MAISI combined-label（本仓 ControlNet 训练用）**：`brain tumor NCR/NETC=401`、`ED=402`、`ET=403`、`body=500`（`configs/label_dict_ctmr.json:347-350`，NV-Segment 解剖区占 1–345）。验收脚本按所用数据格式选其中一套映射到 WT/TC/ET。

### 2.2 可计算指标 + 统计检验 + 通过线

**最小可计算集**（每个都能从 image+mask 对自动算出，真实/合成各成经验分布）：

| 指标 | 类型 | 统计检验 | 通过线（建议） |
|------|------|----------|----------------|
| **WT/TC/ET 体积**（cm³ 或占脑体积比） | 连续 | 两样本 KS + 1D Wasserstein(EMD) | **TOST 等价：90% CI ⊂ ±0.2–0.5×SD**；KS p>0.05 仅初筛 |
| **ET/WT 强化体积比**（BraTS 核心临床量比） | 连续 | KS + TOST | 同上 |
| **肿瘤位置**：centroid 落到解剖分区（label_dict 含 ~300 脑区）的占比 | 分类 | **卡方** | 卡方 p>0.05 初筛 + 各分区占比差 <5pp |
| **centroid 到中线/脑室距离** | 连续 | KS / Wasserstein | TOST 等价 |
| **分区强度离群率**：ET/ED/正常脑区中位强度是否在 mean±3σ / 0.5–99.5 百分位内 | 比例 | 复用 `quality_check.py` | 离群率 ≤ 真实集离群率 + 裕量 |

**通过线先例与依据**：
- 肺癌 GAN 统计验证用 **KS（连续）+ 卡方（分类）、α=0.05，「p>0.05 判分布一致」**，82.76% 特征通过。证据：<https://www.mdpi.com/2079-9292/11/20/3277>。
- 但「p>0.05」只是「无法拒绝≠等价」；更严谨用 **TOST 等价检验（90% CI 落在 ±δ，δ≈0.2–0.5×SD）+ bootstrap CI**。证据：<https://aaroncaldwell.us/TOSTERpkg/articles/robustTOST.html>。
- 连续量还可用 Levene（方差齐性）、Spearman（保序）；多维联合用 Mahalanobis/Hotelling T²、QQ R²（前列腺合成 MRI 即用此组合评估影像组学分布）。证据：<https://www.mdpi.com/2313-433X/12/3/130>。
- **Wasserstein 阈值文献无统一值，需自行用「真实-真实」标定**。

**工具**：新增 `scripts/tumor_stats.py`（从 image+mask 算 WT/TC/ET 体积、centroid、强化比、分区占比 → 输出真实/合成经验分布 + KS/EMD/卡方/TOST 结果）。

### 2.3 影像-标签一致性（P2 掩码→影像专用）

生成影像的肿瘤应确实长在给定 mask 位置：
- **轻量（进最小集，无需分割器）**：比较 ET/WT mask 区**内 vs 外**的强度差是否落在真实数据分布内（复用 `quality_check.py` 思路，`quality_check.py:69-148`）。
- **较强（可选加分，需分割器）**：用现成分割器（nnU-Net/TotalSegmentator）回切生成影像、与输入 mask 算 **Dice + HD95**。MAISI-v2/CoPeDiT 均用此管线；ControlNet 对小肿瘤回切 Dice 仅约 0.57，说明该检查必要。证据：<https://arxiv.org/html/2508.05772v1>、<https://arxiv.org/html/2504.04532v2>。
  - **注意与地图 #2 约束的关系**：此「回切 Dice」是 **P2 一致性检查**（肿瘤是否长在 mask 处），与地图里「下游分割器验证 = 可选加分、不设门槛」不同——后者指用分割任务性能代理影像效用。门槛侧只用无需分割器的强度一致性；回切 Dice 列可选。

### 2.4 「MSD」辨析（issue 用词澄清）

issue 写「MSD-器官与肿瘤尺寸分布」，与 MMD、FDD（均为分布距离）并列且修饰「尺寸分布」。判定：**此 MSD 应读作「器官/肿瘤尺寸分布的分布距离」（自定义），不是分割语境的 Mean Surface Distance（Hausdorff 族，量表面边界重合，与尺寸分布无关）**；「mean slice-wise difference」无标准定义，排除；MSD 也可能是 Medical Segmentation Decathlon 缩写，排除。本文按「尺寸/体积分布对齐」实现（§2.2 体积行）。证据：<https://arxiv.org/abs/2410.02630>、<https://monai-dev.readthedocs.io/en/fixes-sphinx/metrics.html>。

### 2.5 记忆化 / 最近邻排查（防复制训练集）

扩散模型可能记忆并复现训练样本 → 对合成影像做**最近邻相似度**（在特征空间找最近的真实训练样本，报告相似度分布/SSIM），确认非简单复制。证据：<https://www.nature.com/articles/s41597-024-03073-x>。列为可选诊断（数据合规相关）。

---

## 3. L3 人工目检

### 3.1 评审维度（针对脑 MR 生成）

3D 医学影像先例正用「外观 / 层间一致性 / 解剖正确」三类（证据：<https://www.nature.com/articles/s41598-023-34341-2.pdf>）。本任务细化四维：

1. **整体真实感**（一眼像不像真 MR）。
2. **解剖合理性**：脑室/皮层/白质/中线结构形态（FID 盲区，必查）。可辅以 **WASABI** 客观形态测量（皮层厚度、皮层下体积等 52 项）做旁证。证据：<https://pmc.ncbi.nlm.nih.gov/articles/PMC13102318/>。
3. **肿瘤真实性/边缘**：肿瘤位置是否合理（不长在脑室/颅外）、边缘与水肿/强化关系是否可信。
4. **伪影与层间一致性**：运动/截断/折叠伪影、棋盘格、矢/冠状面翻页查层间连续。

### 3.2 抽样与评审配置

- **分层**：模型按模态条件生成（`modality_mapping.json`），**按模态分层抽样、每模态单独计分**；再**按 BraTS 子挑战分层**（病种影像特征不同）。
- **样本量**：先例每位评审 **30–100 例**（范围 10–200），real:fake 多为 **1:1**（专家稀缺故样本偏小）。证据：<https://pmc.ncbi.nlm.nih.gov/articles/PMC9683859/>、<https://www.nature.com/articles/s41598-023-28175-1>。建议：**每模态×每子挑战抽 50 合成 + 50 真实**（取自持出集），盲法交错呈现。
- **评审**：**2–4 名（神经）放射科医生**为宜。
- **一致性**：多人报 **Fleiss' κ**；对真值报 Cohen's κ；评审间相关报 Spearman ρ（文献 κ 从 ~0.1 到 0.74 差异大，须如实报告）。证据：<https://assets-eu.researchsquare.com/files/rs-8458482/v1_covered_1a70bcf0-a6c6-47d3-a6c9-6ed82fa91fe6.pdf>、<https://pmc.ncbi.nlm.nih.gov/articles/PMC9911730/>。
- **视觉图灵「通过」判据 = 准确率 ≈50%**（接近随机即不可区分）；脑 MR 先例中神经放射科医生仅 ~70%。证据：<https://pmc.ncbi.nlm.nih.gov/articles/PMC13102318/>、<https://arxiv.org/html/2606.20812v1>。

### 3.3 检查单模板（评审记录表）

每例一行，评审盲填：

| 字段 | 取值 |
|------|------|
| 样本 ID / 模态 / 子挑战 | （背对背记录，评审不可见真伪标签） |
| ① 整体真实感 | Likert 1–5（1=必假 … 5=必真） |
| ② 解剖合理性 | Likert 1–5 |
| ③ 肿瘤真实性/边缘 | Likert 1–5；无肿瘤标本记 N/A |
| ④ 伪影/层间一致性 | Likert 1–5（1=严重伪影 … 5=无伪影） |
| real/fake 二分类 | real / fake（图灵测试） |
| 主要缺陷定位 | 自由文本（脑区/伪影类型） |

**判定**：real/fake 准确率 ≈50% 为佳；各维 Likert 均值 ≥4；Fleiss' κ 如实报告（不强求阈值，作评审质量参考）。

---

## 4. 持出集划分（BraTS2023）

### 4.1 BraTS2023 子挑战与规模（train/val/test 量级）

| 子挑战 | train | val | test | 备注 |
|--------|-------|-----|------|------|
| Adult Glioma（成人胶质瘤） | ~1251 | 219 | 570 | NCR/ED/ET 标注 |
| Meningioma（脑膜瘤） | 1000 | 141 | 283 | NETC/SNFH/ET |
| Metastases（转移瘤，METS） | >2000 | — | — | 12 中心 |
| Pediatrics（儿科，PEDs） | 99 | 45 | 84 | 小样本 |
| Sub-Saharan Africa（SSA） | 60 | 15 | 20 | 小样本、低场强 |
| BraSyn（缺模态合成） | — | — | — | 合成子挑战 |
| Inpainting（局部合成） | — | — | — | 合成子挑战 |

证据：BraTS2023 综述 <https://pmc.ncbi.nlm.nih.gov/articles/PMC10441440/>、Meningioma <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10441446/>、METS <https://pmc.ncbi.nlm.nih.gov/articles/PMC11296866/>、PEDs <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10246083/>、SSA <https://pubs.rsna.org/doi/abs/10.1148/ryai.240528>、Inpainting <https://ar5iv.labs.arxiv.org/html/2305.08992>。

### 4.2 验收持出集划分建议

- **病例级（subject-level）划分**：同一受试者所有模态/序列落同侧，绝不跨训练/验收（防同患者多切片泄漏）。证据：<https://pmc.ncbi.nlm.nih.gov/articles/PMC11099244/>。
- **每个子挑战内部独立分层留出 15–20%**，完全不进微调与调参；按子挑战/部位/中心分层。
- **外部 held-out**：METS/SSA 等多中心任务**整中心保留**作外部测试（跨厂商/跨站点交叉评估有先例）。证据：<https://pmc.ncbi.nlm.nih.gov/articles/PMC7393676/>。
- **最终验收仅用此 held-out 集**算 L1/L2/L3 的「真实参照分布」；小样本子挑战（PEDs 99 / SSA 60）可改为「验证集即持出集」或加大留出比例，须在 ADR 钉板。
- **合成质量联合判定（可选）**：BraSyn 用 SSIM（肿瘤+健康区）+下游分割 Dice(ET/TC/WT) 等权 rank-sum；Inpainting 用 SSIM/PSNR/MSE rank-sum；官方承认「人类专家感知与 ML 指标不完全一致」、建议引入人类代理评估（与 L3 呼应）。证据：<https://pmc.ncbi.nlm.nih.gov/articles/PMC10441440/>、<https://ar5iv.labs.arxiv.org/html/2305.08992>。

---

## 5. 「够用最小集」最终建议

**门槛（必算，决定 go/no-go）**：

1. **L1**：按模态分层的 **2.5D FID**（RadImageNet-ResNet50，MR 适配版）+ **KID/bootstrap CI**；通过线 = 相对「真实-真实」基线的 k 倍。
2. **L2**：**WT/ET 体积分布（KS+EMD，TOST 等价）** + **ET/WT 强化比（TOST）** + **centroid 解剖分区（卡方）** + **分区强度离群率（quality_check 复用）**。
3. **L3**：按模态×子挑战分层的 **视觉图灵准确率（≈50%）** + **4 维 Likert（均值≥4）**，2–4 名神经放射科医生、报 Fleiss' κ。
4. **持出集**：subject-level、每子挑战分层留 15–20%、完全独立于微调/调参，多中心任务整中心外部 held-out。

**可选诊断（不进门槛，按需启用）**：MS-SSIM/PSNR/LPIPS、核 MMD（自实现）、FRD/RaD、3D-FID（MedicalNet）、ImageNet-SwAV 交叉 FID、回切 Dice（P2 一致性加强）、最近邻记忆化排查、WASABI 形态测量、BraSyn 式 SSIM+下游 Dice。

每层一件工具落点：**L1 → `compute_fid_2-5d_mr.py`（改造）**；**L2 → `tumor_stats.py`（新增）+ `quality_check.py`（复用改造）**；**L3 → 评分表 + 分层抽样清单（§3.3）**。

---

## 6. 证据索引

### 仓库文件（file:line）
- `scripts/compute_fid_2-5d_ct.py:498-503`（特征网络 radimagenet/squeezenet）、`:250-349`（2.5D 三平面）、`:570`/`:565`/`:258`（CT 专用预处理）、`:115,710`（MONAI FIDMetric）
- `scripts/transforms.py:64-65`（MR 百分位归一化 vs CT HU 窗——MR 适配依据）
- `scripts/quality_check.py:69-148`（分区强度 mean±3σ/百分位离群检测，L2 复用）
- `scripts/sample_mask.py:16,52,237-239`（anatomy_size 条件向量、可控肿瘤/器官——尺寸分布验收依据）
- `configs/label_dict_ctmr.json:347-350`（脑瘤 NCR/NETC=401、ED=402、ET=403、body=500）
- `configs/modality_mapping.json`（按模态条件生成：mri_t1=9…——目检/评估按模态分层依据）
- `docs/evaluation.md`（FID 工具说明 + MAISI 在 autoPET 上的 FID 基线表）

### 文献 / 外部
- RadImageNet：<https://github.com/Warvito/radimagenet-models>、<https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/>
- MONAI FIDMetric：<https://monai.readthedocs.io/en/1.3.0/_modules/monai/metrics/fid.html>；MMDMetric（仅线性核）：<https://monai.readthedocs.io/en/1.3.0/_modules/monai/metrics/mmd.html>
- 脑 MR 生成评估 / 解剖合理性：<https://arxiv.org/html/2409.08463v1>
- FID 特征提取器医学评估批评（MICCAI 2024）：<https://arxiv.org/html/2311.13717v5>、<https://github.com/mckellwoodland/fid-med-eval>
- FRD（radiomics 距离）：<https://pmc.ncbi.nlm.nih.gov/articles/PMC13224392/>、<https://arxiv.org/html/2412.01496v1>、<https://github.com/RichardObi/frd-score>
- FID 样本量偏倚 / KID：<https://arxiv.org/html/2510.22070v1>
- MedicalNet（3D 单通道）：<https://github.com/Tencent/MedicalNet>、<https://github.com/Project-MONAI/MONAI/issues/2836>
- 肿瘤分布统计检验 / 通过线：KS+卡方 α=0.05 <https://www.mdpi.com/2079-9292/11/20/3277>；TOST 等价 <https://aaroncaldwell.us/TOSTERpkg/articles/robustTOST.html>；影像组学分布组合检验 <https://www.mdpi.com/2313-433X/12/3/130>
- MSD 辨析：<https://arxiv.org/abs/2410.02630>、<https://monai-dev.readthedocs.io/en/fixes-sphinx/metrics.html>
- BraTS2023 标签：<https://www.synapse.org/Synapse:syn51156910/discussion/threadId=10189>、<https://arxiv.org/pdf/2305.07642>
- 影像-标签一致性（回切 Dice）：<https://arxiv.org/html/2508.05772v1>、<https://arxiv.org/html/2504.04532v2>
- 记忆化排查：<https://www.nature.com/articles/s41597-024-03073-x>
- 目检：样本量/图灵 <https://pmc.ncbi.nlm.nih.gov/articles/PMC9683859/>、<https://www.nature.com/articles/s41598-023-28175-1>、<https://pmc.ncbi.nlm.nih.gov/articles/PMC13102318/>、<https://arxiv.org/html/2606.20812v1>；维度 <https://www.nature.com/articles/s41598-023-34341-2.pdf>；一致性 κ <https://assets-eu.researchsquare.com/files/rs-8458482/v1_covered_1a70bcf0-a6c6-47d3-a6c9-6ed82fa91fe6.pdf>、<https://pmc.ncbi.nlm.nih.gov/articles/PMC9911730/>
- BraTS2023 子挑战与划分：<https://pmc.ncbi.nlm.nih.gov/articles/PMC10441440/>、<https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10441446/>、<https://pmc.ncbi.nlm.nih.gov/articles/PMC11296866/>、<https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10246083/>、<https://pubs.rsna.org/doi/abs/10.1148/ryai.240528>、<https://ar5iv.labs.arxiv.org/html/2305.08992>
- 持出集划分惯例：<https://pmc.ncbi.nlm.nih.gov/articles/PMC11099244/>、<https://pmc.ncbi.nlm.nih.gov/articles/PMC7393676/>
