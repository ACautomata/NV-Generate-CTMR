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

**阶段 0 img2img 基线(stage-0 baseline)**:
零训练比较下限——以冻结 P1-DM、src latent 与 tgt 模态标签执行 RF 插值起点(`x_t=(1-t)·src_latent+t·noise`)的 img2img 推理;契约中以 P3 run 的 `variant=stage0-baseline` 显式标记,四锚轮覆盖每病例 12 个有序模态对。它不训练任何权重(selection 钉 upstream P1-DM checkpoint)、不可挂 L1/L2/L3 正式报告、不可 conclude 终验——只作 P3 训练候选 L1 paired MAE/SSIM 的 baseline 侧与 L2 四锚轮参照,绝不冒充经 ControlNet 训练或终验通过的候选。
_Avoid_: 把阶段 0 称作 P3 候选、给基线做终验裁决、无 variant 标记的 P3 img2img run

**src 模态 / tgt 模态**:
同一病例中作条件的源序列 / 待生成的目标序列;BraTS 四模态(T1、T2、FLAIR、T1c)同空间配准,tgt 模态标签经 class_labels 同时条件 DM 与 ControlNet。
_Avoid_: 输入模态/输出模态、参考模态

**有序模态对**:
(src, tgt) 且 src≠tgt——四模态两两组合,每病例 12 对;embedding 每病例仅 4 份,12 条目引用。
_Avoid_: 模态组合、模态配对(易与 seg 配对混淆)

**T1→T1c 已知限制**:
强化肿瘤(ET)在 T1 平扫几乎不可见,该方向只能统计推断强化位置——训练与所有报告均保留；它仍受整体 FID、L2 与 L3 终验约束，但不以逐病例 MAE/SSIM 必须超越零训练 img2img 基线作为通过门槛。

### 模型挂接

**image-conditioned ControlNet(P3)/ mask-conditioned ControlNet(P2)**:
同一 P1-DM 挂载的两个独立旁路,推理按任务选挂;P3 旁路从 DM encoder 重新初始化,不沿用 P2 权重(条件嵌入形状与语义不同)。
_Avoid_: 「the ControlNet」不加限定地单称

**DM source(唯一 DM 来源)**:
通过完整终验(L1∧L2∧L3 全 pass)后被 `dm_source.json` 账本冻结的 P1-DM——记录其 checkpoint 身份、配置与 provenance,是 P2/P3 旁路唯一允许挂接的对象;后续 P1 候选通过终验即取代之,被取代后挂旧 DM 的旁路 verify 显式失配。
_Avoid_: 把任何 frozen 但未过终验的 P1 称作 DM source、旁路失配后继续引用旧链作当前结论

**产物链**:
P1-DM(全参续训后)+ 旁路(P2 或 P3 ControlNet)构成的推理组合;DM 一旦再训,既有旁路即失配。

### 验收术语

**L1 定量验收**:
按目标模态测量影像特征分布距离；P1/P2 以 MR 适配 2.5D FID 及其不确定性为证据，P3 还利用同病例的配对影像误差。它检验整体影像分布，不替代肿瘤空间分布或专家视觉判断。

**L2 分布对齐验收**:
以生成影像导出的肿瘤测量量，比较肿瘤负荷、强化组成与空间位置的分布；P2 还检验生成结果是否遵循输入掩码条件。它不等同于训练 label 的分布，也不等同于下游分割效用。

**L3 专家目检验收**:
由盲法神经放射科评审同时判断整体真实感、解剖合理性、肿瘤真实性/边缘以及伪影/层间一致性，并以真伪辨识试验检验可区分性。

### 训练配方

**全参续训**:
从既有 ckpt 加载全部 DM 权重继续训练(P1 用法;含复用 ckpt `scale_factor` 的约定)。

**ControlNet-only**:
DM 与 VAE 全冻、只训旁路、从 DM encoder/mid 权重初始化(P2/P3 用法)。
_Avoid_: 旁路微调、adapter 训练

**回放混合**:
在训练数据 list 层混入基模训练数据(MR-RATE)以防灾难性遗忘——仅全参续训需要,ControlNet-only 与 P3 不用。

**病例级持出**:
按子挑战分层的固定病例归属；病例整体不进任何阶段的训练 list，跨阶段复用同一划分。
_Avoid_: 随机切分、逐条目切分

**数据划分角色**:
所有阶段共享按子挑战分层的固定病例级 `70% 训练集 / 10% 开发集 / 20% 最终验收持出集`。开发集用于轻验收、早停与 checkpoint 选择；最终验收持出集只在候选冻结后运行完整 L1/L2/L3，不能用于调参。最终验收采用非补偿性 AND：三层全部必算主判据都必须通过，任一层失败即未通过完整 spec 终验。P3 的同一病例全部 12 个有序模态对必须落在同一侧。
_Avoid_: 将最终验收持出集用于每 epoch 监控、以高分 L1 抵消 L2/L3 失败、逐条目切分、跨阶段改变病例归属

**Split manifest**:
以 `split_id`、子挑战名和官方 subject ID 的 SHA-256 排序，按已定名额确定病例侧别的受控清单。名额由各子挑战可得全量按 70/10/20 以最大余额法取整导出，平手归终验持出侧。完整清单含受 DUA 约束的 ID，必须留在受控数据目录；ADR 仅记录算法、名额和 manifest 内容摘要。增量数据只能创建新版本 manifest，不得悄悄重排已冻结候选的病例归属。
_Avoid_: 依赖下载顺序的随机切分、逐项四舍五入取整名额、将 subject ID 清单提交到 Git

**完整 spec 终验 / provisional smoke**:
完整 spec 终验只在 GLI、SSA、MEN、METS、PED 五个分割子挑战均完成对应划分并通过完整 L1/L2/L3 后成立；「对应划分」以各子挑战**可得全量**为基数——METS 即 5 机构 238 例公开训练集口径（NYU 164 例系 syn51514107 单独托管 entity，2026-08-19 确认不可得，不构成完整性缺口；未来若可得，按增量纪律走新 manifest 版本）。缺少任一子挑战的运行只能称为 provisional smoke 或原型结果，不能外推为全量 BraTS 通过。

**undecided(终验判定态)**:
终验判定的第三态——当 L2 肿瘤测量仪器在待测生成样本上的失败率或层级违反率超出真实校准包络时，该子挑战 L2 记 undecided(仪器不可用)而非 fail(生成不合格)；undecided 同样阻塞完整 spec 终验，修复方向是仪器或重跑，而非生成候选。
_Avoid_: 把仪器失效混同为生成质量失败、以 pass/fail 二态强判 undecided 情形

**完整终验裁决(final acceptance verdict)**:
对 frozen 候选三层正式报告(l1/l2/l3 各恰好一份且校验通过)的非补偿性 AND 判定——三层读数(L1 `summary.verdict`、L2 `overall_verdict`、L3 `verdict.overall`)全 pass 才 pass,任一 fail 或 L2 undecided 即 blocked,并逐条列出可追溯阻塞原因,不被其他层高分抵消;缺失或非法附件拒绝裁决(证据补齐后仍可 conclude),裁决记录一经写出不可变。正式 L2 证据必须五挑战齐全且达冻结持出配额,provisional 运行不构成终验证据。
_Avoid_: 把单层结论或 provisional smoke 称作完整终验通过、以加权平均分代替 AND 抵消某层失败

**L2 肿瘤测量仪器**:
为受控研究自行训练、并在校准集(10% 开发集 ∪ 仪器 fold 内部验证集)真实 BraTS 上校准误差包络前冻结的 **MONAI nnU-Net** BraTS 肿瘤分割网络及其权重——五子挑战各持一个专属仪器，仅用于从生成影像推断 WT/TC/ET mask，以测量肿瘤体积、位置与强化比例分布。它不是下游分割效用验证，不以 Dice/HD95 作为生成候选的通过线。
_Avoid_: 用真实病例 mask 代替生成影像 mask、以 MIC-DKFZ 独立 nnU-Net 实现替代 MONAI、以第三方 SegResNet 作为条件性例外、将测量仪器的推断质量混同为下游分割性能

**SSA 派生 Batch-16 仪器计划**:
仅 SSA `Dataset502` / fold_0 的受控 custom-plan 修订——原始 `nnUNetPlans.json` 不变，派生 `nnUNetPlans_SSA_bs16_v1` 的 `3d_fullres_bs16` 配置仅覆写 global batch=16；8 张 DCU 各 local batch=2。它保留原始 split、patch、spacing、网络与 preprocessor，并在 DTK 26.04 的已知 compile 缺陷下固定 `nnUNet_compile=f`。正式训练另使用可 hash 的 `nnUNetTrainer250Epochs`，它仅把外部 nnunetv2 的 epoch 数设为 250，保留每 epoch 250 iterations。它是 SSA 专属仪器版本，不得称作全默认 nnU-Net，亦不自动推广至其余四个子挑战。
_Avoid_: 覆盖原始 plans、把 SSA 修订泛化为五子挑战规则、以校准结果反调 batch/plans、启用 torch.compile

**仪器使用协议**:
把生成影像组成 L2 肿瘤测量仪器四模态输入的固定方式——P2 用同掩码条件的同病例四模态(肿瘤布局由条件强制一致)；P1 用同病例四模态标签独立采样的伪四模态体(跨模态不自洽如实进入测量，不粉饰)；P3 用 4 锚轮——每轮一个真实模态作锚、其余三模态以该锚为 src 生成，四轮覆盖全部 12 个有序模态对。L2 真实参照侧用同批病例的真实四模态喂仪器。
_Avoid_: 用真实通道补缺生成体(真实通道主导测量)、以生成模型自身评估其仪器(循环)

**仪器输入几何(InstrumentGridGeometry)**:
把体数据重采样到 1mm 各向同性并居中 crop/pad 到目标网格的纯几何变换——连续体用 B-spline、label 用最近邻、背景填体数据默认像素值；仪器网格 240×240×155@1mm 为其标准实例。它是 L2 仪器输入与终验测量网格的唯一几何来源,其仪器口径经 ADR-0002/0004 冻结、不得偏离;B-spline 为冻结标准而非刻意最优。
_Avoid_: 以线性插值喂仪器(已统一收敛为 B-spline)、非居中裁剪、把 xyz 轴序作用于 zyx 数组、在各脚本中散落重写此几何

**冻结仪器调用(FrozenInstrumentCommand)**:
以冻结配置(fold 0、`nnUNetTrainer250Epochs`、镜像 TTA on;SSA 用 `3d_fullres_bs16`+`nnUNetPlans_SSA_bs16_v1`)驱动 L2 肿瘤测量仪器 nnU-Net 预测的唯一构造点——`src/ctmr/instrument/` 的 `FrozenInstrumentCommand.build(输入, 输出) -> argv` 纯方法 + 唯一 canonical 执行入口 `python -m ctmr.instrument.predict`。TTA on 为冻结不变量(无 TTA 形参、永不产出 `--disable_tta`);weights_only 白名单 scoped 收敛于 `nnunet_safe_globals()`。口径经 ADR-0009 钉板,与 ADR-0002/0004 冻结读数一致;代码不动,执行期另行落地。
_Avoid_: 在各脚本散落手写 `nnUNetv2_predict` 命令、用非标准入口名 `nnUNetv2_predict_from_raw_data`、写 `--disable_tta False` token、import 时 `add_safe_globals` 改全局状态

**仪器读数(InstrumentMeasurement)**:
把仪器网格(240×240×155)上的分割掩码派生为 WT/TC/ET 体积、质心位置、强化比等逐病例测量行的纯测量逻辑——`src/ctmr/measure/` 唯一入口 `InstrumentMeasurer.measure(pred, *, gt, condition, brain) -> CaseMeasurement`,校准列(vs GT)/生成列/回切 Dice 三列族按提供的可选 reference 显式门控,canonical 对象配 long(校准)/wide(终验)双序列化。REGIONS/Wilson/Dice 在此各唯一定义;cohort 聚合(R_fail/bootstrap/TOST/verdict)留在判定层、不属本测量。口径经 ADR-0010 钉板,与 ADR-0002/0004 冻结读数一致;代码不动,执行期另行落地。
_Avoid_: 在各脚本散落重写掩码→测量行逻辑、把 cohort 聚合混入测量层、掩码不经仪器网格直接测量

**层级违反(hierarchy_violation)**:
单一 canonical 语义——pred 掩码上 ET⊄TC 或 TC⊄WT 的 containment 违反,或值域 ⊄{0,1,2,3};单表达式、无前置守卫,即 ADR-0004 决定 4 冻结定义(收敛自终验)。空超集不豁免:ET 存在而 TC 空即真违反。
_Avoid_: 空超集前置守卫(跳过 ET 存在而 TC 空的真违反)、vacuous else-True 守卫、与校准病例可用性混称

**校准病例可用性(calibration case-usability)**:
校准侧判某真实病例可否用于包络估计的门禁——GT 值域 ∧ pred 值域 ∧ GT-WT 非空;需 GT,与层级违反(containment)是不同概念,喂校准 R_fail(ADR-0002 语义)。
_Avoid_: 与层级违反混称、用于生成数据(无 GT)

**验收层注册表(AcceptanceLayer)**:
run contract 中验收层(l1/l2/l3_report)全部接线的单点声明——附件 kind、层名、报告 schema 串、validator 构造、层读数路径与阻塞原因推导——`ACCEPTANCE_LAYERS` 一处注册,`ATTACH_KINDS`(层 kind+env,env 为非层附件)与 `LAYER_KINDS` 由此派生;新增验收层=注册表一项+新层自有类组(Rules 与 reasons builder),不再散落平行接线。validator 为组合注入:通用外壳(序幕+绑定+分派)+共享件(绑定校验/挑战读取/微工具)+层 Rules(protocol/coverage/verdict 复算/gate 检查留层——判定不同源的结构保证),非实现继承。经 ADR-0012 钉板,代码不动,执行期另行落地。
_Avoid_: 以实现继承提取三胞胎 validator(应组合注入)、把 gate 检查参数化进注册表数据、ATTACH_KINDS/LAYER_KINDS 与注册表并存双份

**冻结候选绑定(FrozenRunBinding)**:
冻结候选身份五键(run_id/phase/manifest_sha256/candidate_checkpoint_sha256/samples_sha256)的唯一提取构造点,内置 require_frozen 门禁(提取即校验 run 状态);定义于 brats_phase_run_contract.py,L1/L2/L3 生产侧经 scripts 目录 shim import 共用。性质是身份提取而非判定:身份可共享(漂移风险在双侧失同步,共享恰消除);gate 常量镜像与 verdict 复算属判定,保持双侧不同源(ADR-0006 裁判独立性);生产侧 schema 串独立声明防版本漂移。经 ADR-0012 钉板,代码不动,执行期另行落地。
_Avoid_: 生产侧重写五键提取、frozen 门禁留在各调用点、以共享身份为由合并 gate 常量镜像

**weighted_loss(肿瘤区加权)**:
以 label 构造的图像体素损失加权(作用于肿瘤亚区),与条件模态无关——label 进 loss 与验收,不进 P3 条件。

### 执行外壳

**阶段脚本外壳(PhaseHarness)**:
阶段脚本(finetune/dev_eval/launcher)中与阶段领域无关的机械骨架——公共 argparse 集与 torchrun 校验、epoch 循环与早停文件轮询、checkpoint 原子发布、训练 provenance 写盘、dev watch/select 轮询骨架、参数化启动模板(幂等守卫内置)——统一收敛于 `src/ctmr/harness/`;各阶段只以内核(数据构成/模型挂接/单 batch 损失/checkpoint payload 四方法)组合注入,外壳不持任何配方值与领域判定,配方守卫(RecipeGuard)为其一等钩子。CLI 面保持不变;口径经 ADR-0011 钉板,代码不动,执行期另行落地。
_Avoid_: 在新阶段脚本中再抄外壳骨架(应注入内核)、把配方值下沉进外壳、把 P3 生成链轻复制族(prep/jobs/wait 轮询器)混称 PhaseHarness 纳界(另立 #92)
