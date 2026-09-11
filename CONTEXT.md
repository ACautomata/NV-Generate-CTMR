# NV-Generate-CTMR

在 NV-Generate-CTMR（MAISI v2 / CTMR）基础上，把预训练的 `rflow-mr-brain` v1 微调到 BraTS，同时度量并抑制灾难性遗忘。

## Language

### 空间几何（spatial geometry）

**spacing**:
体素间距，单位 mm/体素，各轴独立（可各向异性）。**这是唯一被称为「分辨率」的量。**
_Avoid_: 分辨率（单独使用时不明确）, voxel size（可接受但少用）

**dim**:
体量在各轴上的体素个数（矩阵尺寸），如 `[256, 256, 128]`。推理配置里的同义字段是 `output_size`。
_Avoid_: 分辨率, size, 尺寸（单独使用时不明确）

**FOV**:
视野，`FOV[i] = dim[i] × spacing[i]`，单位 mm。三个量里改一个必然牵动另一个；模型对 FOV 的敏感度高于 spacing——它只见过训练分布内的 FOV，条件数值再合法，FOV 越界就会产出非真实图像。
_Avoid_: 分辨率, 范围

**目标几何（target geometry）**:
BraTS 全链路（编码 / 微调 / 生成 / 评测）统一使用的空间几何：1mm 各向同性、`dim [256, 256, 128]`、中心对齐。这是本项目唯一一处与 MR-RATE 不同的几何约定。
_Avoid_: 分辨率, 裁剪尺寸, target shape

**条件（condition）**:
采样或训练时喂给模型的元数据（`spacing`、modality、region index），与权重无关。同一份权重可以按不同条件采样；训练条件与采样条件不一致即为条件外推。
_Avoid_: 配置, 参数

**latent**:
VAE 编码后的体量，各空间轴为 `dim` 的 1/4（下采样倍率 4）。

### 数据与实验（data & experiment）

**replay（回放）**:
从 MR-RATE 抽样的旧数据子集，与 BraTS 混训以抑制遗忘。其 latent 保持各 volume 的 native 几何（round-to-128 后尺寸各异），不与 BraTS 对齐。

**遗忘参照集**:
冻结的 MR-RATE val 抽样名单及其生成/FID 数值，用于跨实验点比较遗忘程度；按惯例冻结后不重算。

**冻结基线**:
T8 冻结的 pretrained v1 生成条件、FID 表与抽样名单。它是后续所有实验点的对照基准，一旦生成条件变更即整体作废。
