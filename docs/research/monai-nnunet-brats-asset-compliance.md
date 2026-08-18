# 冻结 MONAI nnU-Net BraTS 权重资产取得与合规验证

> 对应 GitHub issue [冻结 MONAI nnU-Net BraTS 权重资产取得与合规验证](https://github.com/ACautomata/NV-Generate-CTMR/issues/25)，隶属
> [BraTS2023 Rectified Flow 微调配方 — wayfinder 地图](https://github.com/ACautomata/NV-Generate-CTMR/issues/2)。
>
> 研究截点：2026-08-19。证据以可定位的一手发布方来源为准；没有下载任何 checkpoint、受限 BraTS 数据或其他资产，没有训练、改代码、改 issue、提交或推送。
>
> 约束解释：本仓 `CONTEXT.md` 明确排除「以 MIC-DKFZ 独立 nnU-Net 实现替代 MONAI」。因此本记录中的 **MONAI nnU-Net** 是指 MONAI 官方发布的权重，或有可核验来源证明能按 MONAI `nnUNetV2Runner` 路径交付/运行的资产；仅写着 `nnUNet`、`nnunetv2` 或要求安装 MIC-DKFZ nnU-Net 的第三方资产不合格。

## 结论：NO-GO；没有合格的公开冻结资产

**没有找到可作为 L2 分布对齐和 P2 条件一致性测量仪器的、公开可访问且冻结的 MONAI nnU-Net BraTS 四模态 checkpoint。**截至本研究截点，既没有用户提供的候选资产，也没有 MONAI 官方模型注册中可核验的 nnU-Net BraTS 模型目录、完整训练输出目录或其不可变权重版本。

这不是「MONAI 没有 nnU-Net 实现」的结论。MONAI 确实实现了 `nnUNetV2Runner` 集成；但它是对外部 `nnunetv2` 的运行适配层，不是带 BraTS 权重的模型注册/下载器。MONAI 官方目前可定位的 BraTS 分割 bundle 是 `SegResNet`，不能被改称为 nnU-Net。第三方 Hub 中检出的 nnU-Net/`nnunetv2` 伪候选则分别明确要求 MIC-DKFZ 或 custom MedNeXt trainer，且没有 MONAI provenance。

因此：

- **不得**为 L2 或 P2 设置最终通过线；二者继续被[冻结 MONAI nnU-Net BraTS 权重资产取得与合规验证](https://github.com/ACautomata/NV-Generate-CTMR/issues/25)阻塞。
- **不得**用 MONAI `brats_mri_segmentation`（SegResNet）或任何独立 MIC-DKFZ/custom-nnU-Net 资产替换该约束。
- 这不是对整个互联网不存在未发布资产的数学断言，而是「在有可复核主来源、可冻结、可合规、可由 MONAI 路径加载的证据门槛下，没有可批准候选」的审计结论。证据不足本身即为 NO-GO。

## 1. 判定门槛与研究边界

一个候选必须同时通过以下所有门槛；任一项为未知、缺失或不满足，均不能作为测量仪器。

| 门槛 | 必需证据 | 本次状态 |
|---|---|---|
| MONAI provenance | MONAI 官方发布，或发布方可核验地证明该目录按 MONAI `nnUNetV2Runner` 路径产生/交付；不能只凭 `nnUNet` tag 推断 | 没有通过者 |
| 不可变资产 | 发布 URL、release/tag/commit、每个文件的 SHA-256（或等价内容哈希）、文件树、发布日期与访问状态 | 没有通过者 |
| Runner 可加载 | 完整 `model_training_output_dir` 契约，含相应 `fold_X`、plans/dataset 元数据、trainer/configuration，且 checkpoint 名称/格式可由 Runner 实际初始化 | 没有通过者 |
| BraTS 四模态契约 | 训练所用 cohort、四通道 `_0000…_0003` 的语义顺序、空间/强度/裁剪等预处理均由资产自身元数据声明 | 没有通过者 |
| WT / TC / ET | 明确的原始标签和后处理，将输出映射为 WT、TC、ET；不得从「BraTS」名称猜测 | 没有通过者 |
| 合规 | 权重许可证、署名、取得条件和与 BraTS Synapse DUA 的关系均有一手文字证据；对不明示的权重/衍生物再分发不作推断 | 没有通过者 |
| 量测可校准 | 可以对受控的 10% BraTS development split 固定版本运行，且能产出病例级误差包络 | 未启动；无资产不可启动 |

### 范围和方法

1. 锁定官方 MONAI `dev` 提交
   [`43c0aaeda1482cdb7fd956c72a4bb508607e3008`](https://api.github.com/repos/Project-MONAI/MONAI/commits/43c0aaeda1482cdb7fd956c72a4bb508607e3008)
   （2026-08-18）与 MONAI model-zoo `dev` 提交
   [`b9e4d04bb2a073110bde9e5c05c9690241e938b6`](https://api.github.com/repos/Project-MONAI/model-zoo/commits/b9e4d04bb2a073110bde9e5c05c9690241e938b6)
   （2026-07-08）。以下代码、模型树和配置均引用这两个不可变 Git 对象。
2. 审计 MONAI runner 代码、官方 model-zoo 的完整 `models/` 直接子树、唯一的官方 BraTS segmentation bundle 及其 large-file manifest。
3. 以 Hub 发布者自己的 model card、Hub Git revision 和 LFS 元数据复核两个名字最接近的公开第三方资产；这些检索仅用于发现/排除，不把发布者自述升级为 MONAI 背书。Hub 检索不是对所有互联网资产的穷尽性证明。
4. 阅读 BraTS 官方 Synapse Data Access/Download 条款和 BraTS 组织维护的 evaluator 配置；不从博客、镜像或二手说明推导许可、标签或可加载性。

## 2. MONAI 有 runner，不等于 MONAI 发布了可用权重

### 2.1 `nnUNetV2Runner` 的可复核边界

在冻结的 [MONAI runner 源码](https://github.com/Project-MONAI/MONAI/blob/43c0aaeda1482cdb7fd956c72a4bb508607e3008/monai/apps/nnunet/nnunetv2_runner.py#L41-L70) 中，类文档要求 `nnunetv2 > 2.2`，而非一个锁定版本。预测代码直接导入外部实现：

```python
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
```

随后调用 `initialize_from_trained_model_folder(...)` 和 `predict_from_files(...)`；见冻结源码的
[`predict`](https://github.com/Project-MONAI/MONAI/blob/43c0aaeda1482cdb7fd956c72a4bb508607e3008/monai/apps/nnunet/nnunetv2_runner.py#L833-L923)。该类没有 checkpoint registry、权重下载 URL、下载器或 BraTS model card。训练参数 `pretrained_weights` 也只是本地 checkpoint 路径，而不是官方资产标识。

Runner 对一个已有模型目录的预测入口是（签名以冻结源码为准）：

```python
runner.predict(
    list_of_lists_or_source_folder=...,  # 输入目录或每病例 channel 文件列表
    output_folder=...,
    model_training_output_dir=...,       # 必须是已有训练输出目录
    use_folds=...,
    checkpoint_name="checkpoint_final.pth",
)
```

`model_training_output_dir` 必须包含所选 `fold_X` 子目录；输入文件必须有训练时相同的后缀和正确的 `_0000` 等通道编号。`checkpoint_final.pth` 只是默认值，并不证明任何具体目录存在或可加载。公开 API 的完整参数和默认值见同一
[`predict` 定义](https://github.com/Project-MONAI/MONAI/blob/43c0aaeda1482cdb7fd956c72a4bb508607e3008/monai/apps/nnunet/nnunetv2_runner.py#L833-L889)。

**关键区分：**Runner 可以调用一个满足 nnU-Net 文件夹契约的本地目录；这不能倒推出 MONAI 曾发布该目录，也不能把 MIC-DKFZ/custom trainer 资产变成 MONAI 资产。

### 2.2 官方 model-zoo 没有 nnU-Net BraTS 发布目录

冻结 model-zoo 的根 `models` tree 是
[`1713a1a38354736891e897180e7853194224b2f9`](https://api.github.com/repos/Project-MONAI/model-zoo/git/trees/1713a1a38354736891e897180e7853194224b2f9)。其直接 BraTS 项只有：

- `brats_mri_axial_slices_generative_diffusion`
- `brats_mri_generative_diffusion`
- `brats_mri_segmentation`

该完整直接目录表中没有名为或包含 `nnunet` 的发布目录。`brats_mri_segmentation` 的冻结 tree 是
[`93cda208daae2a6f7d8c26dd12e54b5d6e2ebced`](https://api.github.com/repos/Project-MONAI/model-zoo/git/trees/93cda208daae2a6f7d8c26dd12e54b5d6e2ebced?recursive=1)：其中没有 `nnunet` path、没有 `fold_X`，也没有已纳入 Git tree 的权重文件；只包含配置、README、许可证、数据许可证提示、数据清单脚本和 `large_files.yml`。

故在 MONAI 官方可定位的注册树中，**没有一个可审计的 MONAI nnU-Net BraTS 模型目录可进入候选表**。

## 3. 最接近的官方 MONAI BraTS bundle：完整排除记录

下表保留这个接近项，是为了避免以后把「MONAI BraTS segmentation」误当成「MONAI nnU-Net」。它不是合格候选。

| 项目 | 一手证据 | 可复核事实 | 对严格约束的结论 |
|---|---|---|---|
| 发布定位/冻结点 | [`brats_mri_segmentation` tree @ `b9e4d04…`](https://github.com/Project-MONAI/model-zoo/tree/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation) | bundle metadata 版本 `0.5.4`，声明 MONAI `1.4.0`、PyTorch `2.4.0`、数据源 BraTS 2018、用途为 example 而非诊断；见 [`metadata.json`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/metadata.json) | 仅配置树可由 Git commit 冻结；不是 nnU-Net 发布包 |
| 远程权重 manifest | [`large_files.yml`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/large_files.yml) | `models/model.pt` → `https://developer.download.nvidia.com/assets/Clara/monai/tutorials/model_zoo/model_brats_mri_segmentation.pt`，MD5 `870e677b782a5184cbc48db1456b78e8`；`models/model.ts` → `https://developer.download.nvidia.com/assets/Clara/monai/tutorials/model_zoo/model_brats_mri_segmentation.ts`，MD5 `c82f693c8f671e9899d21c2f241892f0` | URL/MD5 由 frozen manifest 给出，但本研究没有下载或探测二进制；manifest 不含 plans、dataset 元数据或 `fold_X` |
| 架构 / Runner 可加载性 | [`inference.json`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/inference.json) | 网络为 `SegResNet`，`in_channels=4`、`out_channels=3`，通过 MONAI Bundle `CheckpointLoader` 加载 `models/model.pt` | **失败**：不是 nnU-Net 目录，不能直接由 `nnUNetV2Runner` 的 trained-model-folder 入口加载 |
| 输入 / 预处理 | [bundle README](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/docs/README.md) 和 [`inference.json`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/inference.json) | channel 0–3 = `T1c, T1, T2, FLAIR`；配置中的预处理是 `LoadImaged` 后 `NormalizeIntensityd(nonzero=True, channel_wise=True)` | 该顺序/预处理只属于排除的 SegResNet；不得外推给不存在的 MONAI nnU-Net 候选 |
| 输出语义 | 同一 [`metadata.json`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/metadata.json) 与 [`inference.json`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/inference.json) | 网络输出 channel 0/1/2 = `TC/WT/ET`；离散后处理产出为 `0=background, 1=TC, 2=WT, 4=ET` | **失败**：不是 issue 所需的候选输出契约，且是 BraTS 2018 而非可证明覆盖本项目所有 BraTS2023 子挑战的 nnU-Net |
| 许可 | [bundle `LICENSE`](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/LICENSE) | Apache-2.0 覆盖其中所称 Work；文本没有提及外部下载的 `model.pt`/`model.ts` | **未知/失败**：不能由 bundle 代码许可证推导外部权重许可证、BraTS DUA 相容性或再分发权 |

## 4. 补充发现的公开伪候选：有来源记录，但都不合格

以下均是发布者自己的 Hub 记录，而不是博客转述。它们被记录为**排除项**，不构成「真正候选」；各发布者对训练来源和许可证的自述也不能自动取代 BraTS 组织者的 DUA。

### 4.1 `NicoloPecco/nnUNet_BraTS_Metastases`

- **公开定位和冻结性。**资产是
  [`NicoloPecco/nnUNet_BraTS_Metastases` @ `5852879…`](https://huggingface.co/NicoloPecco/nnUNet_BraTS_Metastases/tree/5852879d40e5f2cc5b5d1dd5db849dfa7fc41355)，
  public、非 gated。发布者的
  [Hub API record](https://huggingface.co/api/models/NicoloPecco/nnUNet_BraTS_Metastases?blobs=true)
  记录其唯一权重 `checkpoint_best.pth` 的 LFS SHA-256 为
  `5dcd99861935cdb953e05449fc8b492ff1bf9444b853179292ff00b1574fa6cd`，大小
  246,962,786 bytes。本研究未下载该文件。
- **发布者所称模型契约。**其
  [pinned model card](https://huggingface.co/NicoloPecco/nnUNet_BraTS_Metastases/blob/5852879d40e5f2cc5b5d1dd5db849dfa7fc41355/README.md)
  称为 BraTS **2026** brain-metastasis、default nnU-Net residual encoder，输入
  `T1n/T1c/T2w/T2f`，输出
  `0=background, 1=NETC, 2=SNFH, 3=ET, 4=RC`。它没有给出可核验的预处理
  plans 或 dataset tree。
- **Runner / MONAI 审计。**失败。model card 明令安装 `MIC-DKFZ/nnUNet`；发布 tree
  只有一个 fold checkpoint，缺 plans、dataset metadata 和 trained-model folder。
  `checkpoint_best.pth` 也不是 Runner 默认的 `checkpoint_final.pth`。没有 MONAI
  provenance，不能声称可直接由 Runner 加载。
- **许可 / DUA。**Hub 标签为 CC-BY-4.0；这没有说明训练 BraTS 数据取得条件是否满足，
  亦没有把 BraTS DUA 对权重/输出的边界写清。**DUA 相容性未知。**

### 4.2 `GuillaumeCassez/mednext-baseline-brats2023gli`

- **公开定位和冻结性。**资产是
  [`GuillaumeCassez/mednext-baseline-brats2023gli` @ `9654da5…`](https://huggingface.co/GuillaumeCassez/mednext-baseline-brats2023gli/tree/9654da507a1185bef5582ab257245c00b77847c6)，
  public、非 gated。发布者的
  [Hub API record](https://huggingface.co/api/models/GuillaumeCassez/mednext-baseline-brats2023gli?blobs=true)
  给出五个 `fold_{0..4}/model.safetensors` 的 SHA-256：

  | fold | SHA-256 |
  |---|---|
  | 0 | `f8e3e64ad7b2bddf2db6af66767ba904fa905a9076866f3cdf49f0e80e96d5a0` |
  | 1 | `fb342c9325a061b6b4c75cdd2c8089d7253c9ec708e4e4eac7cd5c2b4b65c389` |
  | 2 | `311a624cc4865600be7df96f176644c6d8487c5d7f59cd701bf4e6a95831d44f` |
  | 3 | `e9b11a49681e524ed11b583a33882bcc360482b521bab620ceadd4ec63013101` |
  | 4 | `7a3bae0a50b52e6278e3e34cd5742d8fb31e85c3492da62c5737efd001b60124` |

  本研究未下载这些文件。
- **发布者所称模型契约。**其
  [pinned model card](https://huggingface.co/GuillaumeCassez/mednext-baseline-brats2023gli/blob/9654da507a1185bef5582ab257245c00b77847c6/README.md)
  说是 BraTS2023 GLI、`T1/T1ce/T2/FLAIR`、patch `128³`，并要求按
  `nnUNetPlans_96GB_mednext` 预处理；它没有给出可由 MONAI Runner 消费的整包/离散
  label 产物契约。每 fold 的
  [metadata](https://huggingface.co/GuillaumeCassez/mednext-baseline-brats2023gli/blob/9654da507a1185bef5582ab257245c00b77847c6/fold_0/metadata.json)
  声明 custom `nnUNetTrainerMedNeXtBaseline` 和 `safetensors`。
- **Runner / MONAI 审计。**失败。资产是 MedNeXt-B/custom trainer，不是 MONAI nnU-Net。
  其使用说明要求 MIC-DKFZ nnU-Net 及 MedNeXt trainer；不能凭含有 `plans.json` 或
  `nnunetv2` tag 证明 `nnUNetV2Runner.initialize_from_trained_model_folder(...)` 能加载。
- **许可 / DUA。**发布者标记 MIT，并声称不再分发原始 BraTS 数据；这不能证明训练权重的
  取得、使用或衍生输出与 Synapse DUA 相容。**DUA 相容性未知。**

这两项说明为什么「有公开 checkpoint」或「有 5 个 fold」不足以满足本票。前者有 hash 但
文件树不完整且明确依赖独立 nnU-Net；后者有 5 fold 和 plans，却是 custom MedNeXt、也
明确依赖外部 trainer。二者都不得被接入 L2/P2。

## 5. 四模态、标签与许可：不允许从名称外推

### 5.1 通道顺序和预处理

MONAI Runner 不规定 BraTS 模态语义；它只要求输入 channel 文件编号正确且与训练相同。BraTS 组织维护的
[README](https://github.com/BrainLesion/BraTS/blob/main/README.md) 在一个 2023 adult-glioma 示例中列出 `t1c, t1n, t2f, t2w`，但这是一条数据示例，不是任何 checkpoint 的 model-input contract。它不能用来替代候选目录的 `dataset.json`、plans 或发布方 model card。

因此本票没有可填写的「合格 checkpoint `_0000…_0003` 映射」或「合格 checkpoint 强度/几何预处理」。每个未来候选必须自行提供该事实；不允许把 §3 中 SegResNet 的 `T1c/T1/T2/FLAIR` 顺序借来使用。

### 5.2 WT / TC / ET

同理，`nnUNetV2Runner` 不会把泛化的 class index 自动解释为 WT、TC、ET。BraTS 官方维护的 evaluator 目前在
[`config_gli.yaml`](https://github.com/BraTS/BraTS_evaluation/blob/main/brats_evaluation/configs/config_gli.yaml) 中明确配置 `et: [3]`、`tc: [1, 3]`、`wt: [1, 2, 3]`；同配置还出现 `rc: [4]`。这说明区域派生规则和实际整数编码必须由目标 cohort/configuration 逐项锁定，不能把某一年或某个子挑战的标签域外推为所有资产。

本研究没有合格候选，因此下列事实全部为 **未知/不适用**：其原始输出类别、离散后处理、WT/TC/ET 映射、是否满足 `ET ⊆ TC ⊆ WT`，以及生成条件 mask 与其输出的对齐规则。

### 5.3 BraTS DUA 与资产许可证

BraTS 官方 [Data Access/Download](https://repo-prod.prod.sagebase.org/repo/v1/entity/syn51156910/wiki/627000) 页面要求注册 Synapse、在 Files tab 请求访问、并接受 post-challenge 条款；该页写明 CC BY-NC、非商业使用与指定署名/引用。它没有明示授予第三方重新分发训练权重、预测 mask 或由数据衍生的其他二进制资产的权利。

官方 [Data Conditions](https://repo-prod.prod.sagebase.org/repo/v1/entity/syn51156910/wiki/622360) 另外约束挑战排名中的外部数据/预训练模型；其规则不能替代 post-challenge Data Access 条款。以上两组条款也不能由 MONAI 的 Apache-2.0、第三方 Hub 的 CC-BY-4.0/MIT badge 或本仓 `LICENSE.weights` 自动满足。

因此即使未来出现满足技术文件树的资产，也必须单独记录：

1. 权重发布方授权什么使用、署名和再分发行为；
2. 训练所用 BraTS cohort 的合法取得和适用条款；
3. 用户自己的 Synapse access/DUA 状态；
4. 仅在受控本地评估时的权限，和任何公开分发 checkpoint、预测、embedding、mask 的权限之间的边界。

未明示的边界一律记为**未知**，而不是推断「许可相容」。

## 6. 10% development-set 最小校准：未来合格资产的可执行入口

本节是未来资产**全部过门槛后**的执行入口和产物契约，不是本次已运行的实验。它不下载数据、不训练，也不允许为适应生成样本而调节分割器。

### 6.1 资产冻结后才允许的 Runner 入口

冻结的 MONAI constructor 需要 `input_config` 的 `datalist`、`dataroot`、`modality` 三个键；随后调用 §2.1 的 `predict`。最小调用形状为：

```python
from monai.apps.nnunet.nnunetv2_runner import nnUNetV2Runner

# 所有 <...> 都来自已经批准的资产 manifest；这里不触发下载。
runner = nnUNetV2Runner(
    input_config={
        "datalist": "<controlled-10pct-datalist.json>",
        "dataroot": "<controlled-BraTS-root>",
        "modality": "<asset-declared-modality>",
    },
    trainer_class_name="<asset-declared-trainer>",
    work_dir="<controlled-work-dir>",
)
runner.predict(
    list_of_lists_or_source_folder="<controlled-10pct-imagesTs>",
    output_folder="<controlled-predictions>",
    model_training_output_dir="<approved-trained-model-folder>",
    use_folds=<approved-fold-set>,
    checkpoint_name="<approved-checkpoint-name>",
    save_probabilities=True,
    overwrite=False,
)
```

`<approved-…>` 不是可自由填写的调参项：必须逐字来自被 hash 的资产记录。特别是，不能默认写死 `checkpoint_final.pth`、`nnUNetTrainer`、`MRI`、0–4 folds、四模态顺序或 intensity transform。若一个候选不能把这些 placeholder 替换为可复核事实，它在此入口前即失败。

### 6.2 10% 校准的受控输入和最小产物

依据仓内已锁定的病例级 `70% / 10% / 20%` 分工，10% 只用于仪器误差校准；20% 最终验收集不参与阈值或 preprocessing 调整。以下数据均应置于 DUA 受控目录，不能提交到 Git：

| 阶段 | 固定输入 | 必须落盘的受控产物 |
|---|---|---|
| 资产验收 | `asset-lock.json`（source URL、release/commit、每文件 SHA-256、文件树、MONAI commit、`nnunetv2` exact version/commit、PyTorch/CUDA/container digest、fold、plans、trainer） | hash 验证记录；文件树审计；Runner 初始化/预测 smoke log。任一文件缺失即停止 |
| 输入契约验收 | 每个 development 病例的四个已编号 NIfTI、原始 header/affine/spacing 记录、候选专属 channel/preprocessing contract | `input-contract-report.jsonl`：文件存在性、顺序、shape/spacing/header、预处理和导出失败。失败不从分母删除 |
| GT 校准 | 固定 10% 病例级 split、官方 GT、批准的 WT/TC/ET 后处理 | 每病例预测 NIfTI（及概率，如有）、`per_case_metrics.csv/parquet`：WT/TC/ET Dice、recall、precision、HD95/必要时 ASSD、mL 体积误差、ET/WT、centroid 距离、component/假阳性、层级违反、失败状态 |
| 误差包络冻结 | 上一行的所有病例，不再换模型/预处理 | `instrument-calibration-summary.json`：每区域 median/IQR、p95、bootstrap 95% CI、偏差和 limits of agreement、repeatability（同输入至少 3 次）以及正式的 `D_low`、`E_vol`、`E_centroid`、`E_HD95`、`R_fail` |
| L2/P2 使用 | 已冻结误差包络和固定仪器 | L2 仅报告在仪器分辨率内的真实-vs-生成分布比较；P2 以相同 WT/TC/ET 口径与误差包络判定。不得把生成结果用于反调仪器 |

指标和误差包络的详细理由沿用现有
[`nnunet-brats-measurement-instrument.md`](nnunet-brats-measurement-instrument.md) 的「10% 开发集最小校准协议」。本节新增的是：没有经本票批准的 asset lock，就连该协议的 Runner 入口也不能启动。

## 7. 最窄下一步：用户提供合法、可审计的 MONAI 资产来源

不建议放宽「仅 MONAI nnU-Net」约束，也不建议把排除项转换为临时仪器。唯一最窄的推进路径是：由用户或权利持有人提供**一个合法可访问的候选来源**，并至少给出下列证据包；代理随后只做读取、hash/文件树/配置/条款核验，不下载受限文件。

1. **来源和冻结标识**：官方发布 URL 或可审计受控访问 URL、release/tag/Git commit、发布日期；每个 checkpoint、plans、dataset metadata、fold 文件的 SHA-256；发布者和权利持有人身份。
2. **完整 Runner 文件树**：`model_training_output_dir` 相对路径清单，所有 `fold_X`、checkpoint 名称、`plans.json`、`dataset.json`/相应元数据、trainer 和 configuration；若为 cascade，还要 previous-stage 要求。仅给单个 `.pth`/`.safetensors` 不足。
3. **MONAI provenance 与环境**：生成/验证该目录的 MONAI commit/version、`nnunetv2` exact version/commit、Python/PyTorch/CUDA/container digest；明确说明它不是要求直接调用 MIC-DKFZ/custom trainer 的替代路径。
4. **数据和标签 contract**：训练 cohort（尤其 BraTS 年份/子挑战）、四通道 `_0000…_0003` 顺序、orientation/spacing/resampling、强度标准化、crop/pad、原始/最终 label mapping、WT/TC/ET 后处理与层级规则。
5. **合规证据**：权重独立许可证、必需署名、合法取得条件，及权利持有人对本项目受控非商业本地评估的许可说明；若涉及 BraTS，附用户自己的 Synapse access/DUA 依据。没有明确的再分发授权时，不复制、不提交、不公开预测或衍生物。

在上述证据包出现并通过本文件 §1 的全部门槛前，[冻结 MONAI nnU-Net BraTS 权重资产取得与合规验证](https://github.com/ACautomata/NV-Generate-CTMR/issues/25)的正确状态保持 **NO-GO**。

## 一手来源索引

- MONAI nnU-Net integration：[`nnunetv2_runner.py` @ `43c0aaed…`](https://github.com/Project-MONAI/MONAI/blob/43c0aaeda1482cdb7fd956c72a4bb508607e3008/monai/apps/nnunet/nnunetv2_runner.py)，以及官方 [MONAI API docs](https://docs.monai.io/en/stable/apps.html#nnunet-runner)。
- MONAI model-zoo：
  [`models` tree @ `1713a1a…`](https://api.github.com/repos/Project-MONAI/model-zoo/git/trees/1713a1a38354736891e897180e7853194224b2f9)，
  [`brats_mri_segmentation` tree @ `93cda208…`](https://api.github.com/repos/Project-MONAI/model-zoo/git/trees/93cda208daae2a6f7d8c26dd12e54b5d6e2ebced?recursive=1)，
  [metadata](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/metadata.json)，
  [inference config](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/configs/inference.json)，
  [large-file manifest](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/large_files.yml)，
  [bundle license](https://github.com/Project-MONAI/model-zoo/blob/b9e4d04bb2a073110bde9e5c05c9690241e938b6/models/brats_mri_segmentation/LICENSE)。
- 检出的第三方发布者源：NicoloPecco 的 [pinned model card](https://huggingface.co/NicoloPecco/nnUNet_BraTS_Metastases/blob/5852879d40e5f2cc5b5d1dd5db849dfa7fc41355/README.md) 和 [Hub API metadata](https://huggingface.co/api/models/NicoloPecco/nnUNet_BraTS_Metastases?blobs=true)；GuillaumeCassez 的 [pinned model card](https://huggingface.co/GuillaumeCassez/mednext-baseline-brats2023gli/blob/9654da507a1185bef5582ab257245c00b77847c6/README.md)、[fold metadata](https://huggingface.co/GuillaumeCassez/mednext-baseline-brats2023gli/blob/9654da507a1185bef5582ab257245c00b77847c6/fold_0/metadata.json) 和 [Hub API metadata](https://huggingface.co/api/models/GuillaumeCassez/mednext-baseline-brats2023gli?blobs=true)。
- BraTS：官方 [Data Access/Download](https://repo-prod.prod.sagebase.org/repo/v1/entity/syn51156910/wiki/627000)、官方 [Data Conditions](https://repo-prod.prod.sagebase.org/repo/v1/entity/syn51156910/wiki/622360)、组织维护的 [evaluator `config_gli.yaml`](https://github.com/BraTS/BraTS_evaluation/blob/main/brats_evaluation/configs/config_gli.yaml) 与 [BraTS project repository](https://github.com/BrainLesion/BraTS)。
