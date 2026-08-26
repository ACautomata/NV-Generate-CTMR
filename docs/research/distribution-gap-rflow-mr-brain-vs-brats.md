# 分布差距分析：rflow-mr-brain v1 与 BraTS2023

关联 issue：#4（父地图 #2）。结论供「VAE 重建冒烟原型」与「P1 配方钉板」使用。

## TL;DR

- **P1 微调距离中等偏近**：不是「学全新外观」，而是「强化既有能力 + 学 BraTS 特有的标准化/增强模式」。MR-RATE 是含肿瘤的临床数据，模型见过肿瘤与部分增强 T1；但肿瘤不可条件化、增强 T1 不可选择。
- **FOV 非 OOD**：BraTS 240×240×155 mm 落在 mr-brain axial 脑 FOV 分布内。
- **t1c 需新增 skull-stripped 索引（34）**：`num_class_embeds=128` 无需扩层、`strict=False` 兼容，只需微调训练该 embedding 行。
- **VAE 适用性有正面证据**：autoencoder_v1 训练含 skull-stripped 胶质母细胞瘤（Upenn GBM）；重点验证增强肿瘤的高信号强度保真（99.5 百分位归一化的 clip 风险）。

---

## 1. 微调距离评估（P1：image-only 带肿瘤生成）

**MR-RATE 是否含肿瘤样本？含。**
MR-RATE 是临床诊断数据（非健康志愿者）：705,254 个体积、98,334 个检查、83,425 名患者，每个检查配放射报告；报告 findings 经 LLM 分为 **37 类 SNOMED CT 病理标签**（含脑/脊髓病理，肿瘤在其中）。证据：HF 数据卡 <https://huggingface.co/datasets/Forithmus/MR-RATE> 与数据指南 <https://github.com/forithmus/MR-RATE/blob/main/data-preprocessing/docs/dataset_guide.md> （「705,254 non-contrast and contrast-enhanced brain and spine MRI volumes」「classified into 37 brain and spine MRI pathology categories」）。

**是否含 post-contrast T1（t1ce）？含，但不可分。**
MR-RATE 的 T1w 采集「可以是**非增强或钆增强**，元数据**目前没有对比增强标签**」（数据指南原文）。即增强 T1 混在 T1w 桶里，无标签可拆。证据：同上 dataset_guide.md（「T1-weighted acquisitions in MR-RATE can be either non-contrast or contrast-enhanced (gadolinium-based). The metadata does not currently include a contrast-enhancement label.」）。

**DM 训练用的序列**：rflow-mr-brain v1 用 MR-RATE（batches 0–27）的 T1w/T2w/FLAIR/SWI/MRA 五类，whole-brain 与 skull-stripped 两种预处理。证据：`data/README.md` §3.4（L166–174）。官方 NV-Generate-MR-Brain 亦只列 T1/FLAIR/T2/SWI（无独立 t1c 条件）。

**对 P1 的含义**：

1. 模型在训练中**见过**脑肿瘤与（混杂的）增强 T1 外观 → 这部分**在分布内**，P1 不是从零学新外观。
2. 但 DM 是 **image-only、仅以 modality code 为条件** 训练（`scripts/diff_model_train.py` L322 `class_labels: modality_tensor`）；**肿瘤不是条件**。模型不能「被要求」生成肿瘤，P1 需要让肿瘤外观在 BraTS 数据分布下被稳定重现。
3. **增强 T1 不可条件化**：MR-RATE 无 t1c 标签，模型学的是混合 T1w；BraTS 的 t1c 是明确的钆增强 T1。要稳定生成 BraTS t1c 需新增独立条件（见 §2）。
4. **预处理分布差**：BraTS 是 1mm 各向同性、共配准、统一 skull-stripped 的标准化数据；MR-RATE 是原生异质临床数据。BraTS 的「干净/标准化」外观本身是需要微调适配的部分。

**结论**：P1 微调距离 = 中等偏近。骨干能力（脑解剖、T1/T2/FLAIR 对比、肿瘤纹理先验）已在；需强化的是 (a) 把肿瘤外观在 BraTS 分布下条件化/稳定化，(b) 独立出 t1c 增强对比，(c) 适配 BraTS 标准化预处理外观。

---

## 2. Modality 标签映射提案

当前映射（`configs/modality_mapping.json` L7–21）：t1n/t2n/t2f 可映射到 skull-stripped 变体，**t1c 无 skull-stripped 索引**（只有 17=mri_t1c 全脑）。

| BraTS 模态 | 含义 | 现有映射 | 语义匹配？ |
|---|---|---|---|
| t1n | native T1 | 29 = mri_t1_skull_stripped | ✅ 匹配（BraTS 全 skull-stripped） |
| t2n | T2 | 30 = mri_t2_skull_stripped | ✅ 匹配 |
| t2f | FLAIR | 31 = mri_flair_skull_stripped | ✅ 匹配 |
| t1c | 增强 T1 | **无 skull-stripped**（仅 17=mri_t1c 全脑） | ❌ 缺口 |

**提案：新增 `34 = mri_t1c_skull_stripped`。**

### 对网络的影响（新增索引是否需扩权重层）

- `num_class_embeds = 128`（`configs/config_network_rflow.json` L62 与 L87，UNet 与 ControlNet 相同）。**34 < 128 → embedding 表大小不变，无需扩展权重层**。新增索引落在既有 embedding 表的未用行内。
- 训练加载用 `strict=False`（`scripts/diff_model_train.py` L166、L168）。索引 34 对应的 embedding 行**从未被训练（随机初始化）**，`strict=False` 加载既有 ckpt 不冲突；**但该行必须在微调中被训练**，否则 t1c 条件等价于随机向量。
- modality 作为 `class_labels` 传入 UNet（`scripts/diff_model_infer.py` L137、L193；`scripts/diff_model_train.py` L322）。
- 标签增强兼容：`augment_modality_label` 中 `mask_mri = modality_tensor >= 9`（`scripts/diff_model_train.py` L58），34 ≥ 9 会被当作 MRI 以概率置为 8（generic mri）做 CFG 增强 — 行为正确。
- 推理强度缩放分支：`modality >= 8` 走 MR 分支 [0,1000]（`scripts/diff_model_infer.py` L226–227），34 ≥ 8 正确落入 MR 缩放；CT 检查 `1<=modality<=7`（L295）不触发。

### 落地清单

1. `configs/modality_mapping.json` 增加一行 `"mri_t1c_skull_stripped": 34`。
2. BraTS 数据 JSON data list 的 `modality` 字段：t1c 样本标 `"mri_t1c_skull_stripped"`（经 `scripts/diff_model_train.py` L132 的 `modality_mapping[...]` 映射为 34）。
3. `docs/inference.md` Modality codes 表补 34。
4. **不改** `config_network_rflow.json`（num_class_embeds 保持 128）。
5. 微调时确保 t1c 样本参与训练，让索引 34 的 embedding 行得到梯度。

---

## 3. FOV / spacing 匹配结论

BraTS：240×240×155 voxel @ 1mm 各向同性 → **FOV ≈ 240×240×155 mm**，axial 采集。

mr-brain 推荐 FOV（MR-RATE 训练分布，`docs/inference.md` L136–160）：

- T1 axial：240 × 240 × 174
- T2 axial：240 × 240 × 158
- FLAIR axial：250 × 250 × 175

**结论：BraTS FOV 在分布内，非 OOD。** x/y=240 与 T1/T2 axial 完全一致；z=155 略低于 T1 的 174、几乎等于 T2 的 158。`docs/inference.md` L160 说明「dim[i]×spacing[i] 落在推荐值附近即可」。BraTS 为 axial，避开了样本极少的 SWI sagittal/coronal 与 MRA 全平面（L140 的质量告警不涉及 BraTS 四模态）。

---

## 4. VAE（autoencoder_v1）适用性：风险点与验证点

**正面证据：VAE 训练含 skull-stripped 带肿瘤脑 MR。**
`data/README.md` §2.1：autoencoder_v1 的 MRI 训练数据（17,887 体积）来自「brain, **skull-stripped brain**, chest, below-abdomen」（L38），其中：

- L61：**TCIA Upenn GBM Brain MR（skull-stripped）2550 体积** —— GBM = 胶质母细胞瘤，即 VAE **见过带高级别胶质瘤的 skull-stripped 脑 MR**；
- L66、L67：Aomic / QTIM Brain MR skull-stripped 各 2630 / 1275。

这对 BraTS（skull-stripped 胶质瘤）的 VAE 适用性是直接正面证据。

**归一化方式（冒烟需重点复核的强度映射）**：

- 训练/预处理：MRI 用 `ScaleIntensityRangePercentilesd(lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False)`（`scripts/transforms.py` L64）→ 按 0–99.5 百分位线性映射到 [0,1]，**clip=False**（超出 99.5 的体素不截断，会 >1）。
- 推理反归一化：[0,1] → [0,1000]，且 `np.clip(data, a_min=0, None)`（`scripts/diff_model_infer.py` L226–227）。
- `configs/config_maisi_vae_train.json` 的 `data_option` 未显式给出额外归一化，强度归一化在 data-prep 的 `define_fixed_intensity_transform` 完成。

**冒烟原型应重点验证**：

1. **增强肿瘤的高信号强度保真**：t1c 增强灶是极端高信号，恰落在 99.5 百分位上限之外（clip=False 会保留 >1，但反归一化后分布是否漂移需实测）。验证增强核心重建后强度/对比是否被压缩。
2. **肿瘤边界重建**：增强核心 / 水肿（edema）/ 坏死核心的边界清晰度，编码–解码后是否模糊（L1 recon + KL，patch 64³，`configs/config_maisi_vae_train.json`）。
3. **skull-stripped 脑掩码边缘**：背景为 0 的硬边界重建是否产生振铃/渗漏。
4. **重建质量基线**：对同一批 BraTS 体积算 whole-brain vs skull-stripped 的 PSNR/SSIM 对比，确认 skull-stripped 不劣化。

---

## 证据索引

- `configs/modality_mapping.json` L7–21（现有映射）
- `configs/config_network_rflow.json` L62、L87（num_class_embeds=128）
- `scripts/diff_model_train.py` L58、L132、L166、L168、L322
- `scripts/diff_model_infer.py` L137、L193、L226–227、L295
- `scripts/transforms.py` L64（MRI 99.5 百分位归一化）
- `configs/config_maisi_vae_train.json`（patch_size、recon_loss=l1、kl_weight）
- `data/README.md` §2.1 L38、L61、L66、L67；§3.4 L166–174
- `docs/inference.md` L136–160（mr-brain FOV 表）、L162–180（Modality codes）、L140（稀疏组合告警）
- <https://huggingface.co/datasets/Forithmus/MR-RATE>
- <https://github.com/forithmus/MR-RATE/blob/main/data-preprocessing/docs/dataset_guide.md>
