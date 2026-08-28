# BraTS2023 Rectified Flow 微调配方

本仓库(fork of NV-Generate-CTMR)在 BraTS2023 上基于 `rflow-mr-brain v1` 微调 Rectified Flow 的领域术语表。单一上下文;当代码与本表冲突时,修其一。

## Language

### 能力阶段

**P1(image-only 带肿瘤生成)**:
仅以模态标签为条件的影像生成微调——全参续训 DM,VAE 冻结,使基模习得肿瘤外观与 t1c 模态。
落地名:`ctmr.application.generation.modality_label`(代号不进代码命名,[ADR-0015](docs/adr/0015-ddd-layered-endstate.md))。
_Avoid_: 文生图、无条件生成

**P2(掩码→影像配对生成)**:
以解剖/肿瘤掩码为条件的影像生成——ControlNet-only,职责是「掩码→空间布局」。掩码条件**仅属 P2**。
落地名:`ctmr.application.generation.mask`。
_Avoid_: 分割条件生成(与 P3 混称)

**P3(跨模态条件生成)**:
以源模态影像为条件生成目标模态——image→image 跨模态翻译,条件**只有源影像**(不含掩码)。
落地名:`ctmr.application.generation.cross_modal`。
_Avoid_: 跨序列生成、模态合成、模态翻译(translation)

### P3 专属

**阶段 0 img2img 基线(stage-0 baseline)**:
零训练比较下限——以冻结 P1-DM、src latent 与 tgt 模态标签执行 RF 插值起点(`x_t=(1-t)·src_latent+t·noise`)的 img2img 推理;契约中以 P3 run 的 `variant=stage0-baseline` 显式标记,四锚轮覆盖每病例 12 个有序模态对。它不训练任何权重(selection 钉 upstream P1-DM checkpoint)、不可挂 L1/L2/L3 正式报告、不可 conclude 终验——只作 P3 训练候选 L1 paired MAE/SSIM 的 baseline 侧与 L2 四锚轮参照,绝不冒充经 ControlNet 训练或终验通过的候选。
落地名:`ctmr.application.generation.cross_modal.baseline`(纯代码用词;工件契约标记串 `variant=stage0-baseline` 为冻结兼容值不改,[ADR-0015](docs/adr/0015-ddd-layered-endstate.md) 命名规则⑥)。

**P3 跨模态候选(controlnet-candidate)**:
stage-0 的受训对照——冻结 P1-DM + 独立从 DM encoder 重新初始化的 image-conditioned ControlNet 旁路,以 4 通道 src latent 为条件、tgt 模态标签经 `class_labels` 同时进 DM 与 ControlNet,从纯噪声去噪且 CFG=0(默认关闭 CFG,零 latent unconditional 分支——issue #61 验收 1-2)。训练钉 P2 同等配置、条件嵌入形状为 4 通道(区别于 P2 掩码的 8 通道)、不沿用 P2 ControlNet 权重,`variant=controlnet-candidate` 显式标记。它与 stage-0 基线合流为 L1 `brats-l1-pairs/1` 三元组(reference+baseline+candidate)供 paired MAE/SSIM 判定,并可挂 L1/L2/L3 正式报告、走 conclude 终验——但候选与基线是「受训对照 vs 比较下限」的关系:候选经整体 FID/配对误差/L2/L3 检验才判通过,绝不因相对基线占优而免于终验,也绝不冒充基线(反方向防混淆:候选 selection 钉自己的 ControlNet checkpoint,不是 upstream P1-DM)。
落地名:`ctmr.application.generation.cross_modal.candidate`。
_Avoid_: 把候选当基线、给候选贴 stage0 标记、让候选 selection 选 upstream DM、在候选侧沿用 P2 掩码条件或 8 通道条件嵌入
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
同一 P1-DM 挂载的两个独立旁路,推理按任务选挂;P3 旁路从 DM encoder 重新初始化,不沿用 P2 权重(条件嵌入形状与语义不同)。旁路由 ControlNetBypass 行为实体承载,以组合注入组成 DiffusionModel 的条件前向路径;其运行时对象不代表权重身份。
_Avoid_: 「the ControlNet」不加限定地单称、将旁路 Python 对象当作 checkpoint 身份

**DM source(唯一 DM 来源)**:
通过完整终验(L1∧L2∧L3 全 pass)后被 `dm_source.json` 账本冻结的 P1-DM——记录其 checkpoint 身份、配置与 provenance,是 P2/P3 旁路唯一允许挂接的对象;后续 P1 候选通过终验即取代之,被取代后挂旧 DM 的旁路 verify 显式失配。
_Avoid_: 把任何 frozen 但未过终验的 P1 称作 DM source、旁路失配后继续引用旧链作当前结论

**产物链**:
P1-DM(全参续训后)+ 旁路(P2 或 P3 ControlNet)构成的推理组合;DM 一旦再训,既有旁路即失配。

### 验收术语

**L1 定量验收**:
按目标模态测量影像特征分布距离；P1/P2 以 MR 适配 2.5D FID 及其不确定性为证据，P3 还利用同病例的配对影像误差。它检验整体影像分布，不替代肿瘤空间分布或专家视觉判断。
落地名:`ctmr.application.acceptance.quantitative`。

**L2 分布对齐验收**:
以生成影像导出的肿瘤测量量，比较肿瘤负荷、强化组成与空间位置的分布；P2 还检验生成结果是否遵循输入掩码条件。它不等同于训练 label 的分布，也不等同于下游分割效用。
落地名:`ctmr.application.acceptance.distribution`。

**L3 专家目检验收**:
由盲法神经放射科评审同时判断整体真实感、解剖合理性、肿瘤真实性/边缘以及伪影/层间一致性，并以真伪辨识试验检验可区分性。
落地名:`ctmr.application.acceptance.expert_review`。

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
落地名:`ctmr.domain.acceptance`(非补偿 AND 裁决核)+ `ctmr.application.acceptance.contract.conclude`(终验编排)。
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
落地名:`ctmr.domain.grid`。

_Avoid_: 以线性插值喂仪器(已统一收敛为 B-spline)、非居中裁剪、把 xyz 轴序作用于 zyx 数组、在各脚本中散落重写此几何

**冻结仪器调用(FrozenInstrumentCommand)**:
以冻结配置(fold 0、`nnUNetTrainer250Epochs`、镜像 TTA on;SSA 用 `3d_fullres_bs16`+`nnUNetPlans_SSA_bs16_v1`)驱动 L2 肿瘤测量仪器 nnU-Net 预测的唯一构造点——规格侧 `ctmr.domain.instrument_spec` 的 `FrozenInstrumentCommand.build(输入, 输出) -> argv` 纯方法 + 唯一 canonical 执行入口 `ctmr measure predict`(住址随 [ADR-0015](docs/adr/0015-ddd-layered-endstate.md) 甲案迁移;被取代的旧入口反向 shim 已于 #175 删除,git 历史即复现锚)。TTA on 为冻结不变量(无 TTA 形参、永不产出 `--disable_tta`);weights_only 白名单 scoped 收敛于 `nnunet_safe_globals()`。实现已落地(#107 收编 #108),执行侧居 `ctmr.infrastructure.nnunet_runner`,规格侧居 `ctmr.domain.instrument_spec`;口径经 ADR-0009 钉板,与 ADR-0002/0004 冻结读数一致。
_Avoid_: 在各脚本散落手写 `nnUNetv2_predict` 命令、用非标准入口名 `nnUNetv2_predict_from_raw_data`、写 `--disable_tta False` token、import 时 `add_safe_globals` 改全局状态

**仪器读数(InstrumentMeasurement)**:
把仪器网格(240×240×155)上的分割掩码派生为 WT/TC/ET 体积、质心位置、强化比等逐病例测量行的纯测量逻辑——`ctmr.domain.measurement` 唯一入口 `InstrumentMeasurer.measure(pred, *, gt, condition, brain) -> CaseMeasurement`,校准列(vs GT)/生成列/回切 Dice 三列族按提供的可选 reference 显式门控,canonical 对象配 long(校准)/wide(终验)双序列化。REGIONS/Wilson/Dice 在此各唯一定义;cohort 聚合(R_fail/bootstrap/TOST/verdict)留在判定层、不属本测量。实现已落地(`ctmr.domain.measurement`,#109),6 处调用点收编归 #110;口径经 ADR-0010 钉板,与 ADR-0002/0004 冻结读数一致。
_Avoid_: 在各脚本散落重写掩码→测量行逻辑、把 cohort 聚合混入测量层、掩码不经仪器网格直接测量

**层级违反(hierarchy_violation)**:
单一 canonical 语义——pred 掩码上 ET⊄TC 或 TC⊄WT 的 containment 违反,或值域 ⊄{0,1,2,3};单表达式、无前置守卫,即 ADR-0004 决定 4 冻结定义(收敛自终验)。空超集不豁免:ET 存在而 TC 空即真违反。
_Avoid_: 空超集前置守卫(跳过 ET 存在而 TC 空的真违反)、vacuous else-True 守卫、与校准病例可用性混称

**校准病例可用性(calibration case-usability)**:
校准侧判某真实病例可否用于包络估计的门禁——GT 值域 ∧ pred 值域 ∧ GT-WT 非空;需 GT,与层级违反(containment)是不同概念,喂校准 R_fail(ADR-0002 语义)。
_Avoid_: 与层级违反混称、用于生成数据(无 GT)

**验收层注册表(AcceptanceLayer)**:
run contract 中验收层(l1/l2/l3_report)全部接线的单点声明——附件 kind、层名、报告 schema 串、validator 构造、层读数路径与阻塞原因推导——`ACCEPTANCE_LAYERS` 一处注册,`ATTACH_KINDS`(层 kind+env,env 为非层附件)与 `LAYER_KINDS` 由此派生;新增验收层=注册表一项+新层自有类组(Rules 与 reasons builder),不再散落平行接线。validator 为组合注入:通用外壳(序幕+绑定+分派)+共享件(绑定校验/挑战读取/微工具)+层 Rules(protocol/coverage/verdict 复算/gate 检查留层——判定不同源的结构保证),非实现继承。经 ADR-0012 钉板;已落地(#136 判官侧,#141 判定链):`ACCEPTANCE_LAYERS` 与三个层验证器、记录词汇/守卫/生命周期变更器/终验编排/验证器及 `ctmr accept contract` 动词面全部居于 `ctmr.application.acceptance.contract`。
_Avoid_: 以实现继承提取三胞胎 validator(应组合注入)、把 gate 检查参数化进注册表数据、ATTACH_KINDS/LAYER_KINDS 与注册表并存双份

**冻结候选绑定(FrozenRunBinding)**:
冻结候选身份五键(run_id/phase/manifest_sha256/candidate_checkpoint_sha256/samples_sha256)的唯一提取构造点,内置 require_frozen 门禁(提取即校验 run 状态);定义于 run contract 模块(ADR-0015 甲案后居 `ctmr.application.acceptance.contract`),L1/L2/L3 生产侧直接 import 共用。性质是身份提取而非判定:身份可共享(漂移风险在双侧失同步,共享恰消除);gate 常量镜像与 verdict 复算属判定,保持双侧不同源(ADR-0006 裁判独立性);生产侧 schema 串独立声明防版本漂移。经 ADR-0012 钉板;已落地(#136):`FrozenRunBinding`/`FrozenRunBindingError` 居于 `ctmr.application.acceptance.contract.binding`,三份生产侧重写已收编(`from_record`/`from_path` 入口),门禁内置。
_Avoid_: 生产侧重写五键提取、frozen 门禁留在各调用点、以共享身份为由合并 gate 常量镜像

**weighted_loss(肿瘤区加权)**:
以 label 构造的图像体素损失加权(作用于肿瘤亚区),与条件模态无关——label 进 loss 与验收,不进 P3 条件。

### 生成内核

**扩散模型实体(DiffusionModel)**:
生成侧的充血行为载体——持有 UNet 权重、scale_factor 与训练/采样配方;单步训练数学(train_step:批损失→反传→优化器一步)与去噪采样循环(sample,含 RF 插值起点与 CFG 语义)是其方法。实例由 checkpoint payload 经工厂方法重建,不含身份;epoch 循环、数据载入、进程派生与写盘编排不进实体方法(留 application)。
_Avoid_: 把 nn.Module 架构类当实体身份、实体自带 sha256 冒充身份、把 epoch 循环或写盘编排做进实体方法

**扩散调度实体(DiffusionScheduler)**:
一次具体去噪轨迹的有状态充血实体——由 DiffusionModel 在每个 sample 调用中创建,持有该轨迹的 timestep 序列与当前推进位置,并以准备/推进/完成方法执行加噪与反向步进;采样完成即终结,不持久化且不进入 checkpoint。它的身份只在本次采样会话内成立,不与 WeightsRef 的权重血统混同。
_Avoid_: 把 Scheduler 作为跨运行/跨 checkpoint 的恒久身份、在同一模型实例上复用可变采样进度、将 Scheduler 的对象地址当作领域身份

### 执行外壳

**阶段脚本外壳(PhaseHarness)**:
训练/dev_eval/生成链驱动中与阶段领域无关的机械骨架——公共 argparse 集与 torchrun 校验、epoch 循环与早停文件轮询、训练 provenance 写盘、dev watch/select 轮询骨架、幂等守卫——统一收敛于 `ctmr.application.shell`;各阶段仅以数据构成、条件张量构造与 checkpoint payload 的薄适配器组合领域实体,模型挂接与单 batch 参数更新归 DiffusionModel/ControlNetBypass,运行时精度策略以 GradientExecutor 注入。外壳不持任何配方值与领域判定,配方守卫(RecipeGuard)为其一等钩子。checkpoint 原子发布与 latest.json 协议归 CheckpointRepository(仓储 b 档);「CLI 面保持不变」已被 ADR-0015 取代为统一 `ctmr` CLI 子命令(namespace 等价断言护航迁移)。
_Avoid_: 在新用例中再抄外壳骨架(应注入内核)、把配方值下沉进外壳、把 launcher/nohup/sidecar 类编排能力写成 bash 或放入 deploy/(编排属应用层)

**历史运行器(legacy run orchestrator)**:
曾服务单一受控执行、使命完结后即退役的脚本——其 ADR-0014 的「留驻原地」处置已被 [ADR-0015](docs/adr/0015-ddd-layered-endstate.md) 取代:仓库终态零 scripts,#38 轻复制族 11 件等全部退役删除,**git 历史**是唯一复现锚(重放=取当时原样代码＋受控产物指纹,#38 nnU-Net 输入的 linear 历史口径亦由 git 历史承载)。同用途新需求一律走正式形状(`ctmr p3 generate` 等),ssh 轮询监控类允许一次性现写,均不从退役件再抄。
_Avoid_: 从 git 捡回退役件当活入口复用或从中再抄、为复现而重建常驻脚本目录(应用 git 历史＋产物指纹)、把一次性轮询器沉淀为参数化模板

### 身份与运维

**检查点身份(Checkpoint Identity)**:
模型权重的业务身份——以 checkpoint sha256 内容寻址标识的血统实例,不是 Python 对象或网络类:同一 Unet 架构会实例化出基模/P1 各 epoch 候选/P2 冻结底座等互异实体。载体是权重集 payload 及其血统记录;dm_source.json 账本是 DM source 血统的权威登记处。CheckpointRepository 是其唯一持久化协议(state_dict 存取＋tmp 原子发布＋latest.json 指针,b 档辖区——provenance 运行日志不入仓储)。DiffusionModel 行为实体不含身份——由 checkpoint payload 经工厂方法重建,身份不入实体本身。
_Avoid_: 把架构类当实体类型作身份、以文件路径代替内容寻址、绕开仓储直接读写 checkpoint

**运维面(deploy)**:
顶层 `deploy/` 目录——部署手册、集群作业提交配方(`jobs/`:仅允许 .sh 居此)、数据获取运维(`data/`)与实验记录(`experiments/`)的分层住址;独立于 Python 包与 ruff/pytest 分层管束之外。launcher/nohup/torchrun 派生/sidecar 落盘等编排能力属应用层,永不回栖运维面。
_Avoid_: 在 deploy/ 之外新产任何 .sh、把业务编排伪装成运维配方塞进 deploy/、在 deploy/ 根部平铺堆放不分职责、实验记录散落仓库其他位置

**实验记录(ExperimentRecord)**:
一次受控实验的知识聚合——目的与关联 issue、运行配置/run_id、关键读数、产物路径、结论与后续动作,单文件承载于 `deploy/experiments/YYYYMMDD-<主题>.md`。「记录落盘=实验完成」。属仓储管束的领域工件(ADR-0015 决定 11):未来一切读取/汇总/工具化写入须经统一 ExperimentRecord 接口,不另开旁路(接口本体待真实需求再实现)。
_Avoid_: 记录散落仓库他处、以聊天或 issue 评论替代落盘、绕过约定批量重写历史记录

### 测试面

**canonical 测试面(pytest)**:
仓库测试的唯一范式——新增测试一律以 pytest 形态写进 tests/,第二层目录按域模块组织(tests/domain|application|infrastructure/…)。13 处存量内嵌 selftest 在迁移中整体转正:子命令消亡、断言成为 test 函数,GPU/集群级加 `gpu` 标记自动跳过。分界线是「能否在任意机器跑」,与两级门禁的收敛级同构;形态经 ADR-0013 钉板、由 ADR-0015 整合收口。
_Avoid_: 以新增子命令形态承载无环境依赖的新测试、把 selftest 与 pytest 并列为两种测试范式、让标记只做转发而不承载断言

**selftest(废止形态)**:
生产脚本内嵌的自检子命令,曾定位 sugon 无 pytest 环境下的集群兼容入口——该前提随 ADR-0015 废止:受控执行窗口暂停,恢复时集群跑的就是 installable package＋正经 pytest。存量 13 处实现按 canonical 测试面词条整体迁入 tests/,本词条仅存史备查。
_Avoid_: 再向任何生产脚本添加 selftest 子命令、把 GPU 冒烟写成需要人肉的临时散件(应为带 gpu 标记的测试)

**两级门禁(收敛 / 冻结·集成)**:
深模块收编的两级验收口径——收敛门禁(单元级,任意机器可跑,pytest/CI 承载)与冻结/集成门禁(DCU 环境:数值逐字节重跑,或 gpu 标记测试全集——dcu_smoke、nnUNet 推理链均已 pytest 化)。ADR-0008~0012 逐份引用;ADR-0013 起收敛级由 CI 常驻承载,集成级载体经 ADR-0015 由子命令改为 gpu 标记测试。
_Avoid_: 把收敛门禁升级成需 GPU/集群的检查、两级混为一级
