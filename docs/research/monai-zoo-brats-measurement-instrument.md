# MONAI Model Zoo BraTS 分割 bundle 作为 L2 测量仪器的核验

> 对应 GitHub issue [#27](https://github.com/ACautomata/NV-Generate-CTMR/issues/27)「MONAI Model Zoo 现成 BraTS 分割权重能否作 L2 测量仪器」。
>
> 研究截点：2026-08-19。Model Zoo 证据固定到官方 `dev` commit [`b9e4d04bb2a073110bde9e5c05c9690241e938b6`](https://github.com/Project-MONAI/model-zoo/commit/b9e4d04bb2a073110bde9e5c05c9690241e938b6)；BraTS Orchestrator 证据固定到维护方仓库 commit [`de6a27ed1bae4f97ddb77e9c4cddfbdc14b5b7b5`](https://github.com/BrainLesion/BraTS/tree/de6a27ed1bae4f97ddb77e9c4cddfbdc14b5b7b5)。
>
> 本记录只读取一手发布方的 Model Zoo 配置/manifest/许可证、BraTS 维护方源码和 Synapse 条款。**没有下载 checkpoint、BraTS 数据或其他受限资产；没有训练、微调或运行分割器；没有改动 GitHub issue/map。**

## TL;DR：SegResNet 是可审计的技术候选，不是诊断工具；目前不能定最终 L2/P2 门槛

在 issue #27 明确允许考察**非 nnU-Net** MONAI Model Zoo 权重的边界下，`brats_mri_segmentation` 不应再仅因其是 SegResNet 而被技术性排除：它是 MONAI Bundle 原生推理的、已发布的四模态 BraTS-2018 分割候选，具备可记录的配置版本、模型结构、输入/输出契约和远程权重 manifest。它的三张区域概率图可以冻结地导出 `WT`、`TC`、`ET` 测量量。

但这**不等于已经批准为测量仪器**，更不等于可定生成模型的最终通过线。当前一手证据仍有四个决定性缺口：

1. `model.pt` / `model.ts` 的官方 manifest 给出 NVIDIA URL 与 MD5，但 0.5.4 registry 的 archive checksum 为空，且 registry 指向 Hugging Face、manifest 仍指向 NVIDIA；一手来源没有解释二者的文件同一性。本研究也未请求二进制，因此只能证明「官方发布了可定位的远程资产」，不能证明研究日仍可成功取得某一确定字节序列。
2. bundle 的 Apache-2.0 许可证明确，但一手来源没有明确把**外部** `model.pt` / `model.ts` 纳入该许可证，也没有给出权重的独立许可。BraTS-2018 数据提示和 BraTS-2023 Synapse 条款也不填补这个空缺；不能据此推定权重的受控使用、再分发，或与目标数据 DUA 的相容性。
3. 该 bundle 声明训练数据为 BraTS 2018、285 个原发脑肿瘤胶质瘤四模态体积；它没有声称在 BraTS2023 的 SSA、MEN、METS、PED 上验证。对五子挑战和合成影像的泛化均为**未知**，不是「已失败」也绝不能被当作「已通过」。
4. 预先锁定的 10% 真实开发集校准尚未发生。特别是五个子挑战必须分别报告误差和不确定性；小 cohort 的 10% 若使区间过宽，不能因为名称仍是“10%”就视为足以支持门槛。

故本票的三级判定如下：

| 问题 | 现在的判定 | 一手证据与边界 |
|---|---|---|
| 可作临床/诊断工具？ | **否。** | bundle metadata 的 `intended_use` 明示它是 example、不可用于诊断；本研究也没有临床验证或监管证据。[
metadata](<https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/metadata.json>) |
| 可作受控的研究测量仪器？ | **潜在的有条件候选，尚未获准。** | 技术输入/输出和 MONAI 推理路径可审计；但必须先补齐权重受控使用权、资产锁定、目标 cohort 分层校准与合成输入适用性证据。 |
| 可据此设置最终 L2 肿瘤尺寸/位置分布与 P2 条件一致性通过线？ | **否（NO-GO）。** | 无权重许可结论、无已验证内容哈希链、无五子挑战校准结果、无合成影像域适用性证据；任何一项缺失都使误差包络尚不存在。 |

此外，仓库单一上下文当前仍将 L2 仪器写为「冻结的 **MONAI nnU-Net**」；见 [`CONTEXT.md`](../../CONTEXT.md)。issue #27 授权本研究审计 SegResNet，并不自动修改该项目约束。因此，即使后续技术/合规/校准全部通过，主会话仍须作出明确的架构/验收政策决定，才可把本候选接入正式 L2；本票不改该文件。

## 1. 候选普查：冻结 Model Zoo 中唯一的 BraTS 分割 bundle

以冻结 commit 的 [`models/`](https://github.com/Project-MONAI/model-zoo/tree/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models) tree 和各 bundle 的官方 metadata 为范围，带 BraTS 身份的目录有三个。只有一个输出肿瘤分割；其余两个自述为扩散生成，不能从影像产生 WT/TC/ET mask，故不是 L2/P2 测量候选。

| Model Zoo 目录 | 官方配置中的任务 | 是否是 issue #27 分割候选 | 结论 |
|---|---|---:|---|
| [`brats_mri_segmentation`](https://github.com/Project-MONAI/model-zoo/tree/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation) | 多模态脑肿瘤亚区分割；SegResNet | 是 | 唯一的非 nnU-Net BraTS 分割技术候选，详见后文。 |
| [`brats_mri_generative_diffusion`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_generative_diffusion/configs/metadata.json) | MRI latent-diffusion generation | 否 | 无分割输出/区域语义，不能充当影像→mask 仪器。 |
| [`brats_mri_axial_slices_generative_diffusion`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_axial_slices_generative_diffusion/configs/metadata.json) | axial-slice latent-diffusion generation | 否 | 同上，且不是三维肿瘤区域分割器。 |

这个普查结论只表示：在该官方冻结 Model Zoo tree 中，没有另一项被发布方标记为 BraTS 分割的 bundle。它**不**是「互联网不存在其他模型」的断言，也不把非 MONAI 的 BraTS Orchestrator、MIC-DKFZ nnU-Net 或第三方 Hub 资产改写成 MONAI Model Zoo 候选。

## 2. `brats_mri_segmentation`：已证实的技术契约

### 2.1 Bundle、权重定位和可复现标识

官方 [`metadata.json`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/metadata.json) 将该 bundle 标为：

- bundle version **0.5.4**；声明 MONAI **1.4.0**、PyTorch **2.4.0**；
- task 是 `Multimodal Brain Tumor Subregion Segmentation`；
- data source 是 **BraTS 2018**；网络输入为四通道，输出为三张区域 mask；
- `intended_use` 为 example，且明确不是诊断用途。

冻结目录本身不包含 `models/` 二进制目录；它包含配置、文档、许可证、数据许可提示、准备脚本和 [`large_files.yml`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/large_files.yml)。该 manifest 宣告：

| 远程路径 | 发布方给出的定位 / MD5 | 本研究能与不能证明的事 |
|---|---|---|
| `models/model.pt` | NVIDIA download URL；`870e677b782a5184cbc48db1456b78e8` | 可证明官方 manifest 声明了用于 bundle 的外部 PyTorch checkpoint 与完整性检查值；未请求该文件，未验证它当前可下载、大小、内容或 MD5。 |
| `models/model.ts` | NVIDIA download URL；`c82f693c8f671e9899d21c2f241892f0` | 同上；不可把它同 `model.pt`、任意旧 archive 或新 hosting 自动视为同一权重。 |

[`model_info.json`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/model_info.json) 还登记 `brats_mri_segmentation_v0.5.4` 的 versioned Hugging Face source，但 `checksum` 是空字符串。metadata changelog 记录过 Hugging Face hosting 更新，而 0.5.4 `large_files.yml` 仍给 NVIDIA URL；官方一手文本未说明两个路径的字节同一性。历史 [`hosting_storage_v1` release](https://github.com/Project-MONAI/model-zoo/releases/tag/hosting_storage_v1) 中虽有较旧 `brats_mri_segmentation` archive，但 release API digest 为空，且没有一手证据把该 archive 与 0.5.4 资产等同。

**审计结论：**版本化 Git 配置、两项官方 MD5 和远程发布意图足以使它进入技术候选表；却不足以建立“已验证、可重建的 0.5.4 二进制供应链”。若获合法使用批准，后续执行必须选定**一个**正式来源，记录下载时间、实际大小和本地 SHA-256，并将其与 `model.pt`/`model.ts`、配置 commit、环境一起锁定；不得在校准后静默切换 hosting 或二进制。

### 2.2 原生 MONAI 推理栈，而非 nnU-Net

官方 [`inference.json`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/inference.json) 指定：

- `_target_` 是 MONAI `SegResNet`，`in_channels=4`、`out_channels=3`；不是 nnU-Net、不是 `nnUNetV2Runner`，也不要求 MIC-DKFZ/custom trainer；
- `CheckpointLoader` 在 `load_pretrain: true` 时加载 `models/model.pt`；官方 README 的入口是 `python -m monai.bundle run --config_file configs/inference.json`，即 MONAI Bundle 路径；[
README](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/docs/README.md)
- 推理采用 sigmoid、0.5 threshold、滑窗 `roi_size=[240, 240, 160]`、`overlap=0.5`；已发布 README 还警告三线性插值造成非确定性。

因此“非 nnU-Net”在 issue #27 的研究边界内不是技术故障，反而是明确的架构事实：必须按此 Bundle/SegResNet stack 冻结运行，不能把它称作 nnU-Net 或偷偷用另一个分割器替换。

### 2.3 四模态输入和预处理：可映射，但必须一次性提供完整合成病例

metadata 的通道定义是：

| SegResNet channel | 语义 |
|---:|---|
| 0 | T1c |
| 1 | T1 |
| 2 | T2 |
| 3 | FLAIR |

官方 [`prepare_datalist.py`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/scripts/prepare_datalist.py) 也按 `t1ce + t1 + t2 + flair` 形成 `image` list。推理配置只做 `LoadImaged` 与 `NormalizeIntensityd(nonzero=True, channel_wise=True)`，没有在此配置内替调用方补齐任意的方向、配准、spacing 或 skull-strip；输入几何因此属于冻结的上游契约，而非可在校准后自由调节的参数。

BraTS 维护方的官方 [`README`](https://github.com/BrainLesion/BraTS/blob/de6a27ed1bae4f97ddb77e9c4cddfbdc14b5b7b5/README.md) 对 2023 adult-glioma、SSA、MEN、METS、PED 的四模态调用均使用 `t1c`、`t1n`、`t2f`、`t2w`。按语义向本 bundle 堆叠时，必须是：

```text
[T1c, T1, T2, FLAIR] = [t1c, t1n, t2w, t2f]
```

这不是把文件名按字母排序，也不是沿用调用示例的参数排列。它应在任何校准前写进 asset/input lock，并对每例的 header、shape、affine、spacing 和空间配准做失败即计入分母的验收。

更关键的是：该候选不能对一张单模态生成影像做测量。L2/P2 每一个被测样本都必须是同一病例、同一肿瘤、同一空间的**四张完整生成模态**。用三张真实模态补一张生成模态，会改变被测对象；将独立生成且未配准的四张影像硬拼，也不满足训练输入契约。若 P1/P2 当前只能产生单模态，则本候选在那个产物形状上不可用，不能借真实通道绕过这一限制。

### 2.4 输出不是三类 softmax；必须冻结为区域映射

metadata 将三个网络输出 channel 定义为 `TC`、`WT`、`ET`。inference config 对每个 channel sigmoid 和 threshold 后，再以固定优先级得到保存的单标签 NIfTI：**ET→4 优先；否则 TC→1；否则 WT→2；否则 background→0**。故：

| 保存值 | 导出的区域 | 用于统计的区域归属 |
|---:|---|---|
| 0 | background | 不属于 WT/TC/ET |
| 1 | TC-only 的离散编码 | TC、WT |
| 2 | WT-only 的离散编码 | WT |
| 4 | ET 的离散编码 | ET、TC、WT |

等价地，区域测量必须采用 `ET={4}`、`TC={1,4}`、`WT={1,2,4}`。这说明三张网络输出本质是层级重叠的区域概率，而非互斥的三类 softmax。

因此，源 bundle 的 `0/1/2/4` 不能直接和 P2 条件 mask 的原始整数值逐个比较；必须在双方都冻结地投影为 `WT/TC/ET` 后再比较。值 `4` 与某个目标 cohort 使用的 ET 整数编码不同，本身**不是**区域测量的技术否决理由；但如果不锁定这张映射表、阈值和优先级，任何后续 Dice、体积或 centroid 都不可解释。BraTS 维护方的 evaluator configuration 也以 region merge（例如 `TC=[1,3]`、`WT=[1,2,3]`）而不是只靠一个模型无关的类名来定义指标，见 [`config_MenPre.yaml`](https://github.com/BraTS/BraTS_evaluation/blob/273f79fa3da1681b63e0e0d3c1d14a317293e8e5/brats_evaluation/configs/config_MenPre.yaml)。不能把该**当前** evaluator 文件反向当作所有历史挑战的 raw-code 证明；它只强化了「目标 cohort 的 mapping 必须逐项锁定」这一要求。

### 2.5 已报告的训练域与未报告的泛化域

bundle README 说它训练于 **285 个 BraTS-2018 三维四通道体积**，并给出本 bundle 的 200/42/43 train/validation/test split；对象是原发脑肿瘤胶质瘤亚区。它记录了本 split Dice（TC/WT/ET），但没有列出 SSA、meningioma、brain metastases 或 pediatric cohort，也没有报告对生成影像的行为。[README](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/docs/README.md)

相比之下，BraTS 维护方 README 将目标群体分开列为：adult glioma pre-treatment、Sub-Saharan-Africa adult glioma、meningioma、brain metastases 和 pediatric brain tumors。[官方任务目录](https://github.com/BrainLesion/BraTS/blob/de6a27ed1bae4f97ddb77e9c4cddfbdc14b5b7b5/README.md) 即使四模态名字相同，也不能从输入格式推导肿瘤类型、扫描条件、标注语义或误差分布相同。

所以合理的适用范围是：

- **GLI 是最接近、但仍未被本记录实测验证的域。**BraTS-2018 provenance 不自动等于 BraTS2023 GLI 性能/偏差已知。
- **SSA、MEN、METS、PED 均是域外推。**没有一手结果时，应标为 unknown；不把它们预先判定为不可用，也不允许把 GLI 的成绩外推为五子挑战通过。
- **合成四模态影像还是第二层域偏移。**真实开发集 calibration 只说明真 BraTS 输入上的误差；不会自动证明相同误差在生成伪影、强度分布或跨模态不一致上成立。

## 3. 许可证、数据条款和受控使用边界

这三层必须分开审计，不能由其中之一推出另外两层。

| 层级 | 一手事实 | 允许的结论 | 不能作出的结论 |
|---|---|---|---|
| Bundle 代码/配置 | bundle [`LICENSE`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/LICENSE) 与 README 表示 Apache-2.0。 | 目录中发布的 bundle 代码与配置有明确 Apache-2.0 文本。 | 不能仅据此断言远程 NVIDIA/HF 二进制权重也被该文本许可。 |
| 训练数据提示 | [`docs/data_license.txt`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/docs/data_license.txt) 说明 BraTS-2018 数据使用/引用提示。 | 必须保留相应数据来源与引用意识。 | 它不是对外部 checkpoint 的独立许可证，也没有证明权重可再分发。 |
| 目标 BraTS2023 数据 | Synapse 官方 Data Access 页面要求注册用户在 Files tab `Request Access`、审阅并接受 post-challenge Terms 后下载；页面写明 Data 为 CC-BY-NC、仅 non-commercial use，且给出署名/引用要求。[
Data Access/Download](<https://repo-prod.prod.sagebase.org/repo/v1/entity/syn51156910/wiki/627000>) | 自有 Synapse 访问且按条款的非商业受控真实数据 calibration 有明确入口。 | 该页没有明确授予或禁止预训练权重、预测 mask、导出统计量或其他衍生物的再分发；也没有声明它使 BraTS-2018 checkpoint 当然与本 DUA 相容。 |

BraTS Data Conditions 对挑战排名的额外训练数据/预训练限制，亦不能替代 post-challenge access 条款；见[官方 Data Conditions](https://repo-prod.prod.sagebase.org/repo/v1/entity/syn51156910/wiki/622360)。

**当前合规状态：**不能写成“权重许可相容”，也不能写成“权重必然不合法”。正确记录是：独立外部权重许可和权利人对本项目受控本地推理的授权仍为 **unknown**；在未补齐前，不下载、不复制到仓库、不发布 checkpoint、预测 NIfTI、病例级数据或可能受约束的衍生物。这个未知项单独足以阻止最终 L2/P2 门槛批准。

## 4. 10% 真实开发集：只可形成经证实的误差包络，不能先验宣称足够

以下是后续在获得权限后才可执行的最小协议，不是本次已运行实验。它不训练或微调分割器；任何变动都必须新建候选，而不能重用旧 calibration。

### 4.1 先锁边界，再运行

1. 由拥有目标数据访问权限的人，在 DUA 受控位置创建固定病例级 70%/10%/20% manifest；subject ID 和完整清单不得进入 Git。10% 用于仪器 calibration，20% 不参与分割器选择、预处理选择或门槛调节。
2. 以**每个** GLI、SSA、MEN、METS、PED 为层分别固定 10% 子集和 GT；GLI-only 的校准只能支持 GLI-only 的临时结论。全五子挑战的最终声明要求五份独立结果，不能用合并平均数掩盖域外失败。
3. 在开始前记录并冻结：Model Zoo commit、bundle version、实际权重来源/大小/SHA-256、所选 `model.pt` 或 `model.ts`、MONAI/PyTorch 版本、容器/驱动/硬件、完整 inference config、四通道文件清单和顺序、affine/spacing/orientation、强度归一化、sliding-window 参数、sigmoid/threshold、标签投影、确定性设置。
4. 在每次运行前验收四通道的存在性、同一病例身份、空间一致性、必要预处理和输出可写性。失败、空输出和格式不符均留在分母；不得通过重排/重采样/换阈值把失败病例从统计中删除。
5. 因 bundle README 明示插值非确定性，同一已锁输入至少重复运行三次；报告 run-to-run 体积、centroid、区域 mask 与失败状态差异。若运行噪声本身超过候选 P2/L2 margin，则停止定门槛。

### 4.2 每病例校准量与所需误差包络

| 类别 | 固定分割器对真实 GT 的最小记录 | 用于 L2/P2 的保守量 |
|---|---|---|
| 输入/运行失败 | 四通道契约失败、加载/导出失败、空 mask、层级违法、重复运行超差 | `R_fail` 及其置信区间；失败不得被平均分掩盖。 |
| 区域重叠 | WT/TC/ET 各自 Dice、recall、precision | 每区域的保守低界 `D_low`，但仅作为测量误差描述，不宣称下游分割效用。 |
| 边界和位置 | HD95（必要时 ASSD）、centroid 距离、connected components、假阳性组件 | `E_HD95`、`E_centroid` 与病灶检出/碎裂风险的上界。 |
| 尺寸和组成 | voxel count × 体素体积得到 mL；signed bias、绝对/相对体积误差、ET/WT | `E_vol`、ET/WT 误差的保守上界及 bias/limits of agreement。 |
| 重复性 | 同一病例多次推理的上述量 | `E_repeat`，必须计入所有 margin。 |

每个 cohort/region 报告病例级原始量、median/IQR、tail quantile、bootstrap 置信区间和失败率。数值与置信水平须在运行前预注册；本研究不虚构数值。若某子挑战的 10% 太小、tail/error CI 过宽、ET 空病例比例过高，或其 `R_fail` 无法接受，则“10%”不构成充分 calibration，正确动作是保留 NO-GO 或扩大**预先重新定义的**受控开发样本，而不是放松生成模型门槛。

### 4.3 如何且仅如何把误差包络用于 L2/P2

在满足上节所有条件后：

- **L2 分布对齐：**真实和生成影像均以完全同一、冻结的四通道输入和区域投影测量 WT/TC/ET 体积、ET/WT、centroid/位置。分布差异的等价 margin 至少包含相应 `E_vol`、`E_centroid`、`E_repeat` 和估计不确定性。若两分布在该包络内，只能说“在本仪器分辨率内未检出差异”，不能夸大为生物学或临床等价。
- **P2 条件一致性：**将生成的完整四模态病例送入同一冻结仪器，再与输入 condition mask 的**WT/TC/ET 投影**作成对比较。任何 P2 margin 都不能比真实 GT calibration 的 `D_low`、`E_vol`、`E_centroid`、`E_HD95` 与 `R_fail` 更苛刻，且应扣入 `E_repeat`。这衡量“生成影像所呈现的肿瘤是否与条件相符”，不是把分割器的临床/下游任务性能作为生成模型效用门槛。
- **合成域适用性闸门：**真实 GT calibration 不证明分割器在合成伪影或跨模态失配上稳定。应先记录合成样本的输入契约失败、异常空 mask、阈值边界行为和与真实开发集不同的运行诊断；若这些显示输入已脱离受校准范围，则不得把真实误差包络外推给该生成子群。L3 专家检查也不由该仪器替代。

这就是为什么一个“冻结分割器 + 10%”可以成为**条件性**测量方案，却不能在校准前或跨域外推时自动成为最终 L2/P2 gate。

## 5. 从当前 NO-GO 升级为“有条件测量仪器”的最小证据包

下列项目全部完成前，不得改写 §TL;DR 的最终门槛结论：

1. **政策边界：**维护方明确记录是否为 issue #27 的 SegResNet 例外更新当前 `CONTEXT.md` 的「MONAI nnU-Net」验收限定；没有该决定，候选只能留作研究发现。
2. **权利与资产链：**权利人/发布方的文字证据允许本项目的受控非商业本地使用；选择一个 0.5.4 权重来源，确认它与发布方声明相符，并记录真实文件 SHA-256、大小和下载时间。没有明确再分发授权时，继续不再分发。
3. **冻结推理 contract：**固定 MONAI Bundle 配置、权重、版本/容器、四通道 `[t1c,t1n,t2w,t2f]`→`[T1c,T1,T2,FLAIR]` 顺序、几何和强度前提、TC/WT/ET 投影，以及 `model.pt`/`model.ts` 选择。每个待测样本必须是同一空间的四张**生成**模态。
4. **五份预锁 calibration：**以自有 Synapse access 在 GLI、SSA、MEN、METS、PED 各自真实 10% development split 上运行，不训练/微调/挑选分割器；按 §4 报告误差包络、失败率和 interval precision。只有 CI 与失败标准在预注册界内的 cohort 可进入候选使用范围。
5. **最终门槛前复核：**对每个拟纳入的生成子群验证合成输入没有脱离校准适用范围；L2/P2 仅使用对应 cohort 的保守误差包络。五子挑战任一项缺失、权利不清、四模态产物不存在或校准不能收敛时，完整 spec 继续 NO-GO。

## 6. 与既有 nnU-Net 记录的关系：收窄结论，而不改写事实

既有 [`nnunet-brats-measurement-instrument.md`](nnunet-brats-measurement-instrument.md) 和 [`monai-nnunet-brats-asset-compliance.md`](monai-nnunet-brats-asset-compliance.md) 正确地指出 `brats_mri_segmentation` 是 SegResNet，不能冒充 MONAI nnU-Net；当前 `CONTEXT.md` 的严格 nnU-Net 限定下，该排除仍然成立。本票不推翻这一事实。

issue #27 额外允许审计非 nnU-Net 后，以下三种更严格的表述取代了可能过度的推论：

1. **“不是 nnU-Net”不等于“不能是研究测量候选”。**它阻止把该 bundle 填入 nnU-Net 槽位；在获得项目例外后，不阻止评估其原生 MONAI SegResNet 路径。
2. **“not for diagnostic purposes”不等于官方明示禁止受控研究测量。**它足以拒绝诊断定位，不能单独证明研究测量不可能；但也绝不构成跨域/合规/临床有效性的正面证据。
3. **`0/1/2/4` 不等于 `0/1/2/3` 不等于自动不能算 WT/TC/ET。**原始编码不可直接比对，但通过已发布且冻结的区域投影可测量；真正的阻塞是若未锁定投影、目标 cohort 标签与 P2 条件的对应关系。
4. **缺少权重许可文字的正确状态是 unknown。**不能把 Apache bundle 许可证推导为权重许可，也不能把缺少该推导写成已证明的禁止；对最终 gate 而言，unknown 本身即为 NO-GO。

## 一手来源索引

- MONAI Model Zoo 固定点和目录：[commit `b9e4d04…`](https://github.com/Project-MONAI/model-zoo/commit/b9e4d04bb2a073110bde9e5c05c9690241e938b6)、[`models/`](https://github.com/Project-MONAI/model-zoo/tree/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models)、[`brats_mri_segmentation`](https://github.com/Project-MONAI/model-zoo/tree/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation)。
- 候选的技术/资产证据：[metadata](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/metadata.json)、[inference config](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/inference.json)、
  [input-list script](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/scripts/prepare_datalist.py)、[bundle README](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/docs/README.md)、[large-file manifest](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/large_files.yml)、[registry entry](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/model_info.json)、[historic hosting release](https://github.com/Project-MONAI/model-zoo/releases/tag/hosting_storage_v1)。
- 许可证与数据提示：[bundle Apache-2.0 LICENSE](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/LICENSE)、[BraTS-2018 data notice](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/docs/data_license.txt)。
- 其他两个 BraTS-named Model Zoo bundle 的排除证据：[3D generative diffusion metadata](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_generative_diffusion/configs/metadata.json)、[axial-slice generative diffusion metadata](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_axial_slices_generative_diffusion/configs/metadata.json)。
- BraTS 维护方的任务/四模态契约：[BraTS Orchestrator README @ `de6a27e…`](https://github.com/BrainLesion/BraTS/blob/de6a27ed1bae4f97ddb77e9c4cddfbdc14b5b7b5/README.md)、[evaluation mapping example @ `273f79f…`](https://github.com/BraTS/BraTS_evaluation/blob/273f79fa3da1681b63e0e0d3c1d14a317293e8e5/brats_evaluation/configs/config_MenPre.yaml)。
- BraTS/Synapse 条款：[Data Access/Download](https://repo-prod.prod.sagebase.org/repo/v1/entity/syn51156910/wiki/627000)、[Data Conditions](https://repo-prod.prod.sagebase.org/repo/v1/entity/syn51156910/wiki/622360)。
