# nnU-Net BraTS 肿瘤测量仪器选型与校准 findings

> 对应 GitHub issue [#22](https://github.com/ACautomata/NV-Generate-CTMR/issues/22)。
> 研究基线：仓库 `29b51e51d5395f61a2cb20138ed7eeab65c3e8e6`；MONAI integration commit [`43c0aaeda1482cdb7fd956c72a4bb508607e3008`](https://github.com/Project-MONAI/MONAI/tree/43c0aaeda1482cdb7fd956c72a4bb508607e3008)；MONAI model-zoo `dev` commit [`b9e4d04bb2a073110bde9e5c05c9690241e938b6`](https://github.com/Project-MONAI/model-zoo/tree/b9e4d04bb2a073110bde9e5c05c9690241e938b6)。
> 本票只做来源核验和协议设计；没有实现代码、下载 BraTS 数据、下载任何权重、改动 GitHub issue 或把受限文件写入仓库。

## TL;DR：没有满足 MONAI-only 约束的可选仪器，L2 目前阻塞

**本票不能选出一个合格的冻结 MONAI nnU-Net BraTS 四模态权重包。**原因不是找不到 MONAI 的 BraTS 分割模型，而是官方 MONAI 提供的两条路径分别不满足“MONAI 的 nnU-Net”与“冻结的可复现权重”这两个必要条件：

1. **MONAI 的 nnU-Net 集成是适配器，不是 MONAI 自带的 nnU-Net 权重/实现。**官方 `nnUNetV2Runner` 文档明确要求 `nnunetv2` 高于 2.2；安装文档要求用户另外安装 `nnunetv2`，并把 `nnUNet_results`/训练输出目录交给外部 nnU-Net。预测函数实际从 `nnunetv2.inference.predict_from_raw_data` 导入 `nnUNetPredictor`。来源：[MONAI `nnunetv2_runner.py`](https://github.com/Project-MONAI/MONAI/blob/43c0aaeda1482cdb7fd956c72a4bb508607e3008/monai/apps/nnunet/nnunetv2_runner.py#L41-L70)、[MONAI 安装说明](https://github.com/Project-MONAI/tutorials/blob/main/nnunet/docs/install.md)。
2. **官方 MONAI model-zoo 的具体 BraTS bundle 不是 nnU-Net。**在冻结 commit [`b9e4d04…`](https://github.com/Project-MONAI/model-zoo/tree/b9e4d04bb2a073110bde9e5c05c9690241e938b6) 的 `models` 树里，BraTS 分割包是 `models/brats_mri_segmentation`；其推理配置的网络明确为 `SegResNet`，不是 `nnUNetV2Runner` 产生/消费的 `nnUNet` 训练目录。来源：[bundle `inference.json`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/inference.json#L19-L36)。

因此不能用该 SegResNet bundle 冒充 nnU-Net，也不能用 MIC-DKFZ 的独立 BraTS checkpoint 代替 MONAI-only 候选。**在拿到一个由 MONAI 明确提供、且包含可核验 checkpoint、版本、目录、许可证的 nnU-Net BraTS 包之前，L2（肿瘤尺寸/位置分布对齐）和 P2（mask→影像回切一致性）不得设置最终通过线。**

## 1. 候选审计结果

### 1.1 最近的可复现 MONAI BraTS 包（排除项，不是本票选型）

为避免把“MONAI BraTS 模型”误写成“MONAI nnU-Net”，记录唯一直接相关的官方 bundle：

| 项目 | 官方证据 | 核验结果 |
|---|---|---|
| Bundle 身份 | [model-zoo 目录（冻结 commit）](https://github.com/Project-MONAI/model-zoo/tree/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation) | `brats_mri_segmentation`，不是 nnU-Net 模型目录 |
| Bundle schema / 依赖 | [`configs/metadata.json`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/metadata.json) | bundle version `0.5.4`；MONAI `1.4.0`；PyTorch `2.4.0`；模型说明为 BraTS 2018 |
| 网络 | [`configs/inference.json`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/inference.json#L19-L36) | `SegResNet`，`in_channels=4`、`out_channels=3`；没有 `nnUNetPlans`/`fold_X`/`checkpoint_final.pth` nnU-Net 目录契约 |
| 权重来源 | [`large_files.yml`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/large_files.yml) | `model.pt` URL 为 NVIDIA 官方下载地址，MD5 `870e677b782a5184cbc48db1456b78e8`；TorchScript `model.ts` MD5 `c82f693c8f671e9899d21c2f241892f0`。本票只读取 manifest，没有下载文件或验证本地哈希 |
| 目录完整性 | [冻结 bundle tree](https://github.com/Project-MONAI/model-zoo/tree/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation) + [`large_files.yml`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/large_files.yml) | Git tree 中有配置、README、license、manifest；权重只在远程 large-file manifest 中声明。由于禁止下载权重，无法声称已检查 zip/文件实体的内部目录；更重要的是该 manifest 也不是 nnU-Net 的 fold 目录 |
| 原生入口 | [`docs/README.md`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/docs/README.md#monai-bundle-commands) | `python -m monai.bundle run --config_file configs/inference.json`，是 MONAI Bundle/SegResNet 入口，不是 nnU-Net predictor |
| 预期用途 | [`metadata.json`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/metadata.json) | `intended_use` 明确为示例、不可用于诊断；不能直接充当本项目的冻结测量仪器 |

该包可作为“MONAI BraTS segmentation bundle 存在”的来源证据，但**不能作为本票选择的仪器**。

### 1.2 仓库现状

在本票基线 commit 上，仓库 tracked tree 中没有 `nnunet`/`nnUNet` 源码、nnU-Net model directory 或分割 checkpoint；相关命中只有通用的 [`LICENSE.weights`](../../LICENSE.weights) 和一张示意 GIF。也就是说，仓内不存在可直接冻结使用的 nnU-Net/分割权重；本 findings 不会把外部权重复制进仓库。

## 2. 六项核验

### (1) Exact 权重包、来源、模型目录完整性

**MONAI nnU-Net：未满足。**MONAI 的 runner 初始化 `nnUNet_results` 目录，并把它作为用户提供的训练结果路径；没有官方下载器、内置 BraTS checkpoint 或可冻结的 MONAI nnU-Net model card。其预测 API 要求 `model_training_output_dir` 下存在各个 `fold_X` 子目录。来源：[runner 的结果目录设置](https://github.com/Project-MONAI/MONAI/blob/43c0aaeda1482cdb7fd956c72a4bb508607e3008/monai/apps/nnunet/nnunetv2_runner.py#L176-L200)、[runner 的预测参数和目录契约](https://github.com/Project-MONAI/MONAI/blob/43c0aaeda1482cdb7fd956c72a4bb508607e3008/monai/apps/nnunet/nnunetv2_runner.py#L833-L913)。

**最近的 MONAI BraTS bundle：可核验 manifest，但不合格。**`large_files.yml` 给出了 `model.pt`/`model.ts` 的官方 URL 和 MD5，但没有 nnU-Net 的 `fold_0`…`fold_4`、plans 或 `checkpoint_final.pth`。model-zoo Git tree 本身不含大文件实体；在本票禁止下载权重的前提下，内部文件清单只能写成“未核验”，不能假称完整。

### (2) nnU-Net 版本和原生推理入口

MONAI runner 源码只给出 **`nnunetv2 > 2.2`** 的下限，没有固定到具体 PyPI 版本或 git commit；官方教程的安装命令是 `pip install nnunetv2`，同样不是冻结版本。预测时的原生实现来自外部：

```python
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
```

随后调用 `initialize_from_trained_model_folder(...)` 和 `predict_from_files(...)`。来源：[版本要求与环境说明](https://github.com/Project-MONAI/MONAI/blob/43c0aaeda1482cdb7fd956c72a4bb508607e3008/monai/apps/nnunet/nnunetv2_runner.py#L41-L70)、[实际预测调用](https://github.com/Project-MONAI/MONAI/blob/43c0aaeda1482cdb7fd956c72a4bb508607e3008/monai/apps/nnunet/nnunetv2_runner.py#L895-L923)。

因此目前没有可记录为“MONAI nnU-Net vX + exact checkpoint”的组合。未来若补齐，必须同时锁定 MONAI commit、`nnunetv2` 版本/commit、PyTorch/CUDA/container digest、plans、trainer、checkpoint hash 和 fold 集合。

### (3) 四模态 channel 顺序与文件命名

MONAI runner 本身不规定 BraTS 的语义顺序，只把输入 `modality` 列表按顺序写入 dataset JSON；数据转换器按数组 channel 顺序复制成 `_0000.nii.gz`、`_0001.nii.gz` 等文件。来源：[MONAI `create_new_data_copy`](https://github.com/Project-MONAI/MONAI/blob/43c0aaeda1482cdb7fd956c72a4bb508607e3008/monai/apps/nnunet/utils.py#L63-L115) 和 [`create_new_dataset_json`](https://github.com/Project-MONAI/MONAI/blob/43c0aaeda1482cdb7fd956c72a4bb508607e3008/monai/apps/nnunet/utils.py#L137-L168)。预测 API 也明确要求使用正确的 `_0000` 等 channel 编号和与训练集相同的文件后缀。[runner 参数说明](https://github.com/Project-MONAI/MONAI/blob/43c0aaeda1482cdb7fd956c72a4bb508607e3008/monai/apps/nnunet/nnunetv2_runner.py#L852-L865)。

**仅对排除的 MONAI SegResNet bundle 可确定顺序：**

- channel 0 = T1c
- channel 1 = T1
- channel 2 = T2
- channel 3 = FLAIR

该顺序同时出现在 [`metadata.json` 的 `channel_def`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/metadata.json) 和 [bundle README 的 Input 段](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/docs/README.md#input)。这不能外推为任意 MONAI nnU-Net checkpoint 的顺序；未来候选必须以其 dataset.json/model card 为准。

### (4) 输出 0/1/2/3 到 WT/TC/ET 的映射

**当前没有合格 MONAI nnU-Net 输出映射。**MONAI runner 创建的 dataset JSON 只把前景类命名为通用 `class1`、`class2`…；不会自动知道 `WT/TC/ET`。来源：[dataset JSON 生成代码](https://github.com/Project-MONAI/MONAI/blob/43c0aaeda1482cdb7fd956c72a4bb508607e3008/monai/apps/nnunet/utils.py#L152-L162)。

排除项 `brats_mri_segmentation` 也**不是 0/1/2/3**：

- 网络输出 channel 0 = TC、channel 1 = WT、channel 2 = ET；来源：[metadata `label_classes`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/metadata.json)。
- 推理后处理的官方 lambda 是：先把 ET channel 写成 `4`，否则 TC 写成 `1`，否则 WT 写成 `2`，否则背景 `0`；来源：[inference `Lambdad`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/inference.json#L89-L113)。因此其单标签结果是 `0=background, 1=TC, 2=WT, 4=ET`，不是本票要求的 `0/1/2/3`。

未来候选必须提供可复核的原始标签语义和后处理；不能根据“BraTS”名称猜测 label 3 的含义。

### (5) 权重许可、署名、再分发与 BraTS Synapse DUA / 项目 NC

需要分开看三层权利，不能把一个许可证推导成另一个许可证：

1. **MONAI bundle 的仓库代码/配置**带 Apache License 2.0；来源：[bundle `LICENSE`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/LICENSE)。
2. **训练数据**：该 bundle metadata 指向 BraTS 2018，bundle 的第三方许可文件要求保留 BraTS 相关论文引用；来源：[metadata `data_source`/`references`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/metadata.json) 和 [`docs/data_license.txt`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/docs/data_license.txt)。BraTS 2018 官方数据页说明可用于研究，但要求引用指定文献；[官方 data page](https://www.med.upenn.edu/sbia/brats2018/data.html#data) 与 [official request page](https://www.med.upenn.edu/sbia/brats2018/registration.html#data) 是该旧数据集的直接来源。
3. **BraTS 2023 Synapse DUA**：官方 Data Access/Download 页面要求注册 Synapse、接受 post-challenge 条款，并明确 CC-BY-NC、仅非商业使用、署名句和用户信息共享；来源：[Synapse 官方 wiki API（`syn51156910/wiki/627000`）](https://repo-prod.prod.sagebase.org/repo/v1/entity/syn51156910/wiki/627000)，人类可读入口为 [`Synapse:syn51156910/wiki/627000`](https://www.synapse.org/Synapse:syn51156910/wiki/627000)。官方 Data Conditions 页面还写明，参加挑战排名时不得使用在额外数据上预训练的模型；来源：[Synapse `wiki/622360`](https://repo-prod.prod.sagebase.org/repo/v1/entity/syn51156910/wiki/622360)。

**对本项目 NC 场景的结论：**在完成相应 Synapse 访问并保留官方署名的前提下，非商业研究/评估用途与 CC-BY-NC 的“使用”方向一致；但是本票没有找到该 DUA 对“由数据训练出的模型权重再分发”是否自动许可的明确条文，也没有找到上述 MONAI `model.pt` 的独立权重许可证/署名清单。Apache-2.0 的 bundle 代码许可证不能替代训练数据条款，仓库 [`LICENSE.weights`](../../LICENSE.weights) 的 NVIDIA NC 许可也不能替代 BraTS DUA。因此**不能在本票中承诺 checkpoint、导出的 mask 或衍生模型可公开再分发**；需要逐项保留数据方条款并取得项目/法务确认。未解决事实详见 §5。

### (6) 10% 开发集最小校准协议

下面是“未来拿到合格 MONAI nnU-Net checkpoint 后”的最小协议，不是本票已执行的实验；本票没有下载真实 BraTS 数据。

#### A. 数据与冻结边界

1. 通过 Synapse 自有账号和 DUA 获取目标 cohort 的训练数据；从病例级（不是切片级）抽取 **10% development set**，按子挑战/中心（若可用）和肿瘤大小分层，四模态始终同一病例；其余 90% 不用于阈值调参。数据访问和 CC-BY-NC 条款以 [Synapse Data Access/Download](https://repo-prod.prod.sagebase.org/repo/v1/entity/syn51156910/wiki/627000) 为准。
2. 把以下内容写入实验记录并冻结：MONAI commit、`nnunetv2` exact version/commit、容器 digest、PyTorch/CUDA、checkpoint 和每个 fold 的 hash、plans/trainer、输入 affine/spacing/orientation、四模态顺序、强度归一化、后处理和确定性设置。任何失败重跑不得换 preprocessing。
3. 先在真实 GT 上校准测量误差，再把阈值应用于生成影像；不使用生成样本反向调节分割阈值。ET 小、无 ET 和严重类别不平衡病例必须单独报告，不能只报三类平均值。

#### B. 至少要测的误差类别

| 误差类别 | 每病例的最小测量 | 要回答的问题 |
|---|---|---|
| 输入/预处理失败 | 四文件存在率、channel/shape/spacing/header 校验失败率、重采样/导出失败率 | 仪器是否在输入契约外静默地产生结果？失败应计为失败，不得从分母删除 |
| 区域重叠 | WT/TC/ET 各自 Dice、敏感度/召回、precision；中位数、IQR、bootstrap 95% CI | 哪个区域是可测量下限，尤其是 ET？ |
| 边界/形状 | 各区域 HD95（必要时 ASSD）、表面错误的 p95 | 体积相同但边界错位时，测量是否仍可信？ |
| 体积/比例 | 体素数×体素体积得到 mL；signed bias、绝对误差、相对体积误差；WT/TC/ET 及 ET/WT | L2 的尺寸和强化比例误差有多大？报告 Bland–Altman bias 与 95% limits，而非只报 Dice |
| 空间与病灶组件 | centroid 距离、WT/TC/ET connected-component 数、假阳性组件数、病灶检出率 | 肿瘤是否被放错位置或碎裂成测量上不同的对象？ |
| 层级/标签合法性 | 检查 `ET ⊆ TC ⊆ WT`、非法 label、空 mask 与矛盾体素比例 | 输出是否能安全映射到单一 WT/TC/ET 统计？ |
| 重复性/非确定性 | 同一输入至少 3 次运行；体积、Dice、centroid、HD95 的 run-to-run CV/最大差 | MONAI 配置、GPU 或插值是否给测量引入额外噪声？ |

汇总时按病例保存原始指标，至少报告每区域 median/IQR、p95、bootstrap 95% CI 和失败率；不要只报 `mean Dice`。MONAI bundle 自己也警告其三线性插值会导致非确定性，来源：[bundle README 的 nondeterministic note](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/docs/README.md#performance)，所以重复性一项不能省略。

#### C. 把校准结果变成 L2/P2 约束

对每个区域 `r ∈ {WT, TC, ET}`，从真实 development set 计算：

- `D_r,low`：真实 GT 上 Dice 的保守下界（例如单侧 5th percentile 的 bootstrap 下界）；
- `E_r,vol`：绝对相对体积误差的保守上界（例如 p95 及其 bootstrap 上界）；
- `E_r,centroid`、`E_r,HD95`：空间/边界误差的保守上界；
- `R_fail`：输入失败、层级违反和重复运行超差率。

具体分位数和置信水平要在 protocol 中预注册；这里不虚构数值。阈值规则是：

- **L2（生成分布对齐）**：用仪器导出的 WT/TC/ET 体积、ET/WT 比、centroid/解剖分区做真实-vs-生成比较。等价性 margin 至少覆盖 `E_r,vol` 和 `E_r,centroid` 加上重复性误差；优先使用真实-真实 split 作为基线，再用 KS/Wasserstein/TOST 和 bootstrap CI。若生成差异小于该误差包络，只能表述为“在仪器分辨率内不可区分”，不能把更紧的绝对阈值称为通过。
- **P2（mask→影像回切）**：对生成影像运行冻结仪器，再与输入 condition mask 按 WT/TC/ET 比较。区域 Dice 下限不应高于 `D_r,low`（可再扣除已测的 repeatability margin）；体积、centroid、HD95 误差不得超过对应 `E_*` 上界；层级违反率不得高于 `R_fail`。这样不会要求回切分割器达到其在真实 GT 上达不到的精度，也不会让小 ET 的随机波动变成假阴性。
- **门禁**：如果 calibration CI 太宽、ET 的失败率不可接受、输入顺序/label 映射无法复核，停止设置 L2/P2 阈值；先换/补齐仪器，而不是放宽生成模型门槛或偷偷调分割器。

现有仓库的 L2 方案已把 WT/ET 体积分布、ET/WT 比、centroid 和真实-真实基线列为验收轴，见 [`docs/research/quality-metrics-plan.md`](quality-metrics-plan.md#2-l2-肿瘤器官分布对齐)；本节补充的是把这些轴限制在一个经 10% 真实 BraTS 误差校准过的测量分辨率内。

## 未能从一手来源确定的事实（阻塞项）

1. **没有找到 MONAI 官方仓库/model-zoo 提供的冻结 nnU-Net BraTS 四模态 checkpoint。**能确认的官方 BraTS bundle 是 SegResNet；不能据此推断存在隐藏的 nnU-Net 权重。
2. **MONAI runner 的 `nnunetv2` exact 版本/commit 未锁定。**源码只要求高于 2.2；没有可记录的官方 MONAI+nnU-Net 版本组合。
3. **MONAI BraTS `model.pt`/`model.ts` 的独立权重许可、完整署名和再分发条件未在 bundle manifest/README/metadata 中给出。**只能分别引用 bundle Apache-2.0、BraTS data notice 和远程文件 hash，不能把三者合并成权重再分发许可。
4. **BraTS 2023 Synapse CC-BY-NC 条款没有在可见页面明确说明衍生分割 checkpoint 的公开再分发边界。**非商业使用和署名要求是明确的；模型权重、导出的标签及 embeddings 的再分发需要单独取得许可/法务判断。
5. **排除的 MONAI SegResNet bundle 训练于 BraTS 2018，输出 `0/1/2/4`，且 intended use 是示例而非诊断。**它不能作为本票要求的 `0/1/2/3` nnU-Net 测量仪器或 BraTS 2023 校准替代品。

## 来源索引（仅一手来源）

- MONAI nnU-Net runner：[source at frozen commit](https://github.com/Project-MONAI/MONAI/blob/43c0aaeda1482cdb7fd956c72a4bb508607e3008/monai/apps/nnunet/nnunetv2_runner.py)、[utils](https://github.com/Project-MONAI/MONAI/blob/43c0aaeda1482cdb7fd956c72a4bb508607e3008/monai/apps/nnunet/utils.py)、[official tutorial](https://github.com/Project-MONAI/tutorials/tree/main/nnunet)。
- MONAI BraTS bundle：[metadata](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/metadata.json)、[inference config](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/inference.json)、
  [large-file manifest](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/large_files.yml)、[README](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/docs/README.md)、[bundle license](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/LICENSE)、[data license notice](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/docs/data_license.txt)。
- BraTS official terms：[BraTS 2018 data](https://www.med.upenn.edu/sbia/brats2018/data.html)、[BraTS 2018 request](https://www.med.upenn.edu/sbia/brats2018/registration.html)、[BraTS 2023 Synapse access/DUA API](https://repo-prod.prod.sagebase.org/repo/v1/entity/syn51156910/wiki/627000)、[BraTS 2023 Synapse data conditions API](https://repo-prod.prod.sagebase.org/repo/v1/entity/syn51156910/wiki/622360)。
- 本仓现状：[baseline tree](https://github.com/ACautomata/NV-Generate-CTMR/tree/29b51e51d5395f61a2cb20138ed7eeab65c3e8e6)、[weight license](../../LICENSE.weights)。
