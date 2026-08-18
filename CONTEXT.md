# BraTS2023 Rectified Flow 微调配方

本仓库(fork of NV-Generate-CTMR)在 BraTS2023 上基于 `rflow-mr-brain v1` 微调 Rectified Flow 的领域术语表。单一上下文;当代码与本表冲突时,修其一。

## Language

### 能力阶段

**P1(image-only 带肿瘤生成)**:
仅以模态标签为条件的影像生成微调——全参续训 DM,VAE 冻结,使基模习得肿瘤外观与 t1c 模态。
_Avoid_: 文生图、无条件生成

**P2(掩码→影像配对生成)**:
以解剖/肿瘤掩码为条件的影像生成——ControlNet-only,职责是「掩码→空间布局」。掩码条件**仅属 P2**。
_Avoid_: 分割条件生成(与 P3 混称)

**P3(跨模态条件生成)**:
以源模态影像为条件生成目标模态——image→image 跨模态翻译,条件**只有源影像**(不含掩码)。
_Avoid_: 跨序列生成、模态合成、模态翻译(translation)

### P3 专属

**src 模态 / tgt 模态**:
同一病例中作条件的源序列 / 待生成的目标序列;BraTS 四模态(T1、T2、FLAIR、T1c)同空间配准,tgt 模态标签经 class_labels 同时条件 DM 与 ControlNet。
_Avoid_: 输入模态/输出模态、参考模态

**有序模态对**:
(src, tgt) 且 src≠tgt——四模态两两组合,每病例 12 对;embedding 每病例仅 4 份,12 条目引用。
_Avoid_: 模态组合、模态配对(易与 seg 配对混淆)

**T1→T1c 已知限制**:
强化肿瘤(ET)在 T1 平扫几乎不可见,该方向只能统计推断强化位置——训练保留,质量降级记入 spec。

### 模型挂接

**image-conditioned ControlNet(P3)/ mask-conditioned ControlNet(P2)**:
同一 P1-DM 挂载的两个独立旁路,推理按任务选挂;P3 旁路从 DM encoder 重新初始化,不沿用 P2 权重(条件嵌入形状与语义不同)。
_Avoid_: 「the ControlNet」不加限定地单称

**产物链**:
P1-DM(全参续训后)+ 旁路(P2 或 P3 ControlNet)构成的推理组合;DM 一旦再训,既有旁路即失配。

### 训练配方

**全参续训**:
从既有 ckpt 加载全部 DM 权重继续训练(P1 用法;含复用 ckpt `scale_factor` 的约定)。

**ControlNet-only**:
DM 与 VAE 全冻、只训旁路、从 DM encoder/mid 权重初始化(P2/P3 用法)。
_Avoid_: 旁路微调、adapter 训练

**回放混合**:
在训练数据 list 层混入基模训练数据(MR-RATE)以防灾难性遗忘——仅全参续训需要,ControlNet-only 与 P3 不用。

**病例级持出**:
按子挑战分层的 15–20% 病例级划分,病例整体不进任何阶段的训练 list,跨阶段复用同一划分。
_Avoid_: 随机切分、逐条目切分

**weighted_loss(肿瘤区加权)**:
以 label 构造的图像体素损失加权(作用于肿瘤亚区),与条件模态无关——label 进 loss 与验收,不进 P3 条件。
