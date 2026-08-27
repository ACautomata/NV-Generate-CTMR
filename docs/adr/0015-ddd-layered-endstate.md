# ADR-0015：DDD 分层终态——installable package 化、scripts/notebook 归零、deploy 运维面独立

- **状态**：已接受（2026-08-26；在重构执行之前预注册，grilling 会话产出）
- **范围**：整仓 DDD 分层终态钉板——单一生成上下文与三条应用用例族、分层包映射与命名精简规则、installable package 与统一 CLI、CheckpointRepository 辖区、deploy/ 运维面、测试面整合、notebook 处置、迁移批次 M0–M7 与各级门禁。
- **取代性声明**：
  - **取代 ADR-0011** 的：决定 1「CLI 面保持不变」（薄入口形态终结）；决定 7 中 `train_controlnet.py` 不动（上游参照底本随零保留政策退役，教程职能移交 docs＋git 历史）；决定 9 的 sugon 部署模型（sys.path shim → installable）。
  - **取代 ADR-0013** 的：决定 6「维持 no installable package」与 `pythonpath=["src", "."]`；决定 2 的 selftest「双薄入口／集群兼容入口」定位。
  - **取代 ADR-0014** 的：决定 1/2「历史运行器留驻＋注册」。#38 族 11 件连同全部 scripts 一并退役删除，git 历史即复现锚（该 ADR 后果节既有的「真要复跑从 git 历史取原样」路线升级为唯一路线）。
  - 被取代 ADR 其余全部内容维持不动：训练配方值（0005/0007）、终验冻结读数（0002/0004）、gate 常量镜像不合并（0006）、网格几何口径（0008）、仪器命令语义（0009）、测量口径（0010）、验收层注册表（0012）。**本 ADR 只改变代码住址与入口形态，不改任何领域口径数值。**

## 背景

- 痛点与授权（2026-08-26 grilling 会话）：用户痛点＝代码多处 copy、结构混乱；目标＝最佳实践级整洁。用户授权撕掉上述三份当周刚落的钉板。
- 执行窗口事实：sugon 受控执行已主动暂停，仓库无正在运行的受控链路；ADR-0011 决定 8 的时机硬条失效。此前各 run 的 contract 指纹（git_commit/script_sha256）均属历史记录，不受本次迁址影响。
- 既有资产：ADR-0008~0012 已产出五个深模块（grid/instrument/measure/harness 部分落地），是本次归位的落点而非障碍。
- 盘点事实（2026-08-26 探查）：scripts/ 共 73 文件（59 .py ＋ 14 .sh），内部 Python import 仅 7 条边且集中 L2 判官·报告簇；11 个脚本已接入 ctmr.*；util 八件套无任何脚本 import 方（错放的共享库）。根目录 5 个 tutorial notebook 全部为脚本调用型演示，唯 train_vae_tutorial 含全仓唯一的 VAE＋PatchDiscriminator GAN 训练循环。全仓 `scripts/<file>` 文本引用 167 处（docs 131、skills 27、tests 9）；docs/research 笔记占大头、按史料冻结处理。

## 决定

### 1. 单一限界上下文，P1/P2/P3 为三条应用用例族

不拆三个有界上下文。证据：P2←P1 十三符号共享 seam（原 brats_p2_dev_eval.py:66-80）、PhaseTrainKernel 三内核同构、dm_source 产物链跨阶段约束、病例级持出跨阶段复用、验收层横跨全阶段。物理拆分只会复制这些共享语义——正是要根治的病。 ubiquitous language 维持 CONTEXT.md「能力阶段」词条：阶段≠上下文。测量仪器与终验裁决为同一上下文内的支撑子域。

### 2. 甲案分层包映射

```text
src/ctmr/
├── cli.py                     # [project.scripts] 唯一入口 ctmr；子命令分派
├── domain/                    # 纯逻辑：无 IO、无进程派生；torch 张量运算允许
│   ├── identity.py            # Checkpoint/WeightsRef 权重血统实体（sha256 内容寻址）
│   ├── recipe.py              # RecipeSpec 家族(modality_label/mask/cross_modal)与 RecipeGuard 判定
│   ├── grid.py                # ← ctmr/grid/geometry（仪器网格几何）
│   ├── measurement/           # ← ctmr/measure/*（测量纯逻辑，多文件整体平移）
│   ├── instrument_spec.py     # ← ctmr/instrument/command（冻结命令构造＝规格）
│   ├── losses.py              # KL_loss 等纯损失数学（自 scripts/utils 抽出）
│   └── acceptance.py          # 终验裁决核：非补偿 AND、undecided 态、层级违反常量镜像侧
├── application/               # 编排：用例、外壳、生成链驱动；一律功能命名，阶段代号不入代码
│   ├── generation/
│   │   ├── modality_label/    # 原 P1（模态标签条件）：train/monitor/sample 等单词模块（内核注入 shell）
│   │   ├── mask/              # 原 P2（掩码条件）：ControlNet-only 同构单词模块
│   │   └── cross_modal/       # 原 P3（影像条件跨模态）：train＋monitor＋baseline(零训练 img2img 基线)＋candidate(ControlNet 受训对照)
│   ├── shell.py               # ← ctmr/harness/train_shell＋dev_eval 外壳引擎（argparse 公共集并入）
│   ├── vae_train.py           # 自 train_vae_tutorial.ipynb cell 24-30 抽取的 GAN 训练循环（全仓唯一实现）
│   └── acceptance/
│       ├── quantitative/      # 原 L1 定量验收：FID/配对误差链
│       ├── distribution/      # 原 L2 分布对齐：判官簇整体搬迁（final_acceptance(.nifti)/closing/freeze_audit/html 报告对/calibration 对/synthetic_domain_eval）
│       ├── expert_review/     # 原 L3 专家目检盲评
│       └── contract/          # run contract 编排面：frozen 五键绑定/conclude/verify
└── infrastructure/            # IO/框架适配：一切 subprocess、文件系统、三方 SDK
    ├── checkpoints.py         # CheckpointRepository（见决定 4）
    ├── dmsource.py            # dm_source.json 账本读写
    ├── nnunet_runner.py       # ← ctmr/instrument/predict＋safeglobals（argv 执行与 weights_only 白名单）
    ├── instrument_provisioning/  # 仪器供给：nnunet_prep/ddp_preflight/plan_variant/install_trainer/trainer_250_epochs（trainer 类名受 nnunetv2 注册契约钉死除外籍改名）
    ├── maiisi_engine/         # 上游 MAISI 引擎收编：diff_model_setting/diff_model_infer/diff_model_train/create_training_data/sample/utils_infer 等可复用件
    └── dataio/                # transforms/augmentation/quality_check/find_masks/sample_mask/utils_plot/NIfTI·下载装配
```

原则：**纯度上浮、IO 下沉**。同一文件内混血时，IO 骨架留 infrastructure、算法核心抽 domain；执行期逐件搬移时在 PR 描述申报 from→to 映射，防静默漂移。prototype/、data/、configs/ 本 ADR 不触碰。

### 3. installable package 与统一 CLI

- `[project]`：name=`ctmr`，requires-python>=3.11，runtime dependencies 留空；版本锁继续由 requirements.txt 承担，补齐 ADR-0013 点名缺口（SimpleITK、nibabel、cc3d、Pillow）。
- console entry point：`ctmr = ctmr.cli:main`。子命令族与包路径同名对应，代号不进 CLI：
  - `ctmr generate [modality-label|mask|cross-modal] …`（train/dev-eval/generate/manifest；别名 `ctmr gen`）
  - `ctmr measure …`（instrument 预测/校准/FID——仪器运行侧）
  - `ctmr accept [quantitative|distribution|expert-review] …`（各验收层链路）＋ `ctmr accept contract [init/select/freeze/attach/conclude/verify]`（run contract 编排面）
  - `ctmr data …`（prep/encode/download）、`ctmr experiment …`（实验记录仓储口，见决定 11）
- launcher/sidecar/bash 功能全部吸收：torchrun 由应用层以 spawn 方式派生（沿 #123 先例），幂等守卫、RUN_ROOT 布局、pid/sidecar 落盘成为子命令实现细节。**`.sh` 文件除 deploy/ 运维面外一律消亡。**
- CI 与本地开发弃用 pythonpath hack：`pip install -e . --no-deps` ＋ 显式轻科学栈清单。

### 4. CheckpointRepository（b 档辖区）

infrastructure/checkpoints.py 唯一承担：state_dict payload 存取（payload key 集 `unet_state_dict`/`controlnet_state_dict` schema 兼容保持）、tmp 原子发布＋`latest.json` 指针协议（自外壳下沉）。provenance writer 是运行日志不是模型状态，不入仓储，留在 application 层。dm_source.json 账本读写独立于 checkpoints，入 dmsource.py。`checkpoint_payload` 外壳钩子的返回值即仓储入库对象。

### 5. deploy/ 运维面

新建顶层 `deploy/`（不进 Python 包、不受 ruff 分层与 pytest 管束）：

```text
deploy/
├── README.md                  # 手册：pip 安装、DTK 环境、作业提交/nohup 模式
├── jobs/                      # 集群作业提交配方：l2_calibration_predict.sh、run_l2_synth_domain_eval.sh、p1_predict_all.sh
├── data/                      # 数据获取运维：synapse_download_wizard.sh
└── experiments/               # 实验记录新址（一次实验一个 YYYYMMDD-<主题>.md，沿用原约定；受 ExperimentRecord 仓储观约束，见决定 11）
```

CLAUDE.md 的实验记录段同步改址。其余 10 个 .sh 按 §2/§3 吸收或退役：batch_generate_p1(.gpu).sh、p3_launch_shards.sh、wait_×4 属 #38 退役族直接删除；brats_phase_launch_{train,encode}.sh 被 CLI 子命令吸收后删除；brats_phase_dcu_smoke.sh/py 转 GPU 标记测试后删除。

### 6. 测试面整合

- pytest canonical 地位不变（ADR-0013 决定 1 存续）；13 处内嵌 selftest 全部迁入 tests/：脚本中的 `selftest` 子命令随脚本迁移消亡，断言逻辑变为对应 test 函数，目录按甲案改名（tests/domain|application|infrastructure/… 对应第二层）。
- 标记分档：**torch 级**＝CPU 可跑真实执行——测试环境缺 torch 就装：CI 测试 job 一律安装 CPU torch＋monai＋nnunetv2 等全部非 GPU 依赖（单一全依赖档），torch 标记测试在 CI 用合成小夹具真实执行，**禁止缺失即跳过**（`importorskip` 兜底与「CI 不装 torch」前提作废，取代 ADR-0013 §4 的 torch 级 skip 设计）；依赖无法 CI 承载的测试一律归 gpu 级。**gpu 级**＝需真 GPU/集群环境（dcu_smoke、nnUNet 推理链），加 `@pytest.mark.gpu` 自动跳过，统一送服务器（集群恢复后/DCU）执行、不在本地/CI 跑；「任意机器可跑」切分线不变，只是载体从子命令换成标记。
- 「sugon 无 pytest 故 selftest 留驻生产文件」前提废止；集群恢复时跑的就是正经 pytest。两级门禁语义保持：收敛级 CI 常驻（torch 全依赖档），冻结·集成级＝GPU 标记测试全集在 DCU 环境（或集群恢复后）执行。
- 出生即带测试硬门延续：每个新 module 落地 PR 必须同时交付其测试；torch 计算模块（losses/generation 内核/vae_train/引擎冒烟/checkpoint 往返等）带执行级测试而非仅配置级夹具。

> 修订（2026-08-27）：§6 标记分档澄清为「torch 级禁止缺失即跳过、CI 一律装 torch」＋「gpu 级送服务器执行」，取代 ADR-0013 §4 的 torch 级 `importorskip` skip 设计与 CI 轻栈/重栈分离；随 #129 规格修订同步。

### 7. 命名精简规则

**包路径承义，文件名减负**：上下文信息（阶段、层、子域）由包路径携带，文件名只保留用例动词/资源名。具体：① 禁止包内同名前缀复读（`…/mask/mask_train.py` ✗ → `train.py` ✓）；② 文件名超过约 24 字符须给出更短名词或拆分；③ `utils.py` 这类无名库名禁止在新位置复活——按真实职责命名（losses/transforms/plotting…）；④ 连字符文件名借迁移之机消灭（compute_fid_2-5d_ct.py → fid_2d5.py），per-file-ignores 相应清空；⑤ 阶段/验收代号（P1/P2/P3/L1/L2/L3）禁止用作包名、模块名与 CLI 子命令——代码命名一律用功能词（§2 映射），代号的合法居所是 CONTEXT.md 词条、issue 标题与 ADR/研究史料；⑥ 复合流程名以单词承载于代码（dev_eval→watch/select/monitor；stage0→baseline、controlnet→candidate），连字符复合词仅允许出现在 CLI 动词层（`dev-eval`）——运行工件内的冻结契约标记串（如 `variant=stage0-baseline`）不属于命名范围，保持原值兼容既有产物与判定链；⑦ 外部契约名单豁免：`nnUNetTrainer250Epochs`（nnunetv2 注册名）、MONAI/torch API 名。

### 8. notebook 处置

四个直删：inference_tutorial、inference_diff_unet_tutorial、train_diff_unet_tutorial、train_controlnet_tutorial（内容已被脚本/CLI 完全覆盖，无独特实现）。train_vae_tutorial 先抽取后删：cell 24/26/28/30（PatchDiscriminator 构造、GAN 交替循环、AMP GradScaler、三段 warmup LambdaLR）成 application/vae_train.py，docs/training.md VAE 节由 notebook 引用改为实现引用后，删除才解锁。链接清扫范围：README.md:130,164,236-238、docs/training.md:13,41,70,122、docs/inference.md:36。

### 9. 文档处置边界

活文档改写：README、docs/ 八篇（setup/data/training/inference/evaluation/troubleshooting/performance/applications）、skills/ 7 个操作型 skill md——全部指向新 CLI/包路径。史料冻结：docs/research/*.md、docs/adr/ 旧文本、历史实验记录——其中提到的 scripts/ 路径是当时的真实记录，不做追溯改写（误导风险由本 ADR 取代声明兜底）。skills 引用必须更新，否则后续 agent 会话会照着死路径跑。

### 10. 迁移批次与门禁

| 批 | 内容 | 门禁 |
|---|---|---|
| M0 | 本 ADR ＋ CONTEXT.md 词条更新 ＋ CLAUDE.md 改址 | 文档评审 |
| M1 | dataio 工具库迁入（§2 表） | pytest 绿 |
| M2 | domain 层成形（四深模块归位＋identity/recipe/losses/acceptance 新建），tests 同步重排 | 收敛门禁逐条搬家不丢 |
| M3 | infrastructure（checkpoints/dmsource/nnunet_runner/instrument_provisioning/maisi_engine），payload schema 断言 | 保持 |
| M4 | application（generation 三族/acceptance 三层＋contract/shell/vae_train），`ctmr` CLI 上线，判官簇按既有 7 条 import 边整体搬家 | argparse namespace 等价断言延续到子命令 |
| M5 | 上游残件与 `scripts/__init__.py` 拆除、pyproject/CI 收尾（pythonpath 移除、pip -e） | 活代码与活文档 `grep 'scripts/'` 归零 |
| M6 | notebook：先抽 VAE 循环，随后五件连引删除 | docs/training.md VAE 节换实现引用 |
| M7 | skills 7 件＋README/docs 活文档清扫收官 | 全仓 grep 校验 |

每批合入时仓库全绿（pytest＋pre-commit）。硬排序：M0 先行；M6 的删除动作晚于 VAE 抽取；其余批内顺序执行期微调。

### 11. 实验记录纳入仓储管理

实验记录是领域工件而非散放笔记：以 **ExperimentRecord** 聚合建模——目的与关联 issue、运行配置/run_id、关键读数、产物路径、结论与后续动作，单文件承载于 `deploy/experiments/YYYYMMDD-<主题>.md`，「记录落盘＝实验完成」纪律不变。本 ADR 只钉抽象与住址，不投机实现工具代码：待出现真实的读取/汇总/检索需求时，才在 infrastructure 落地唯一的 ExperimentRecord 存取接口并挂 `ctmr experiment …` 子命令；在那之前人工书写即合规。

## 前置（硬阻塞）

- 无。窗口暂停状态下即可开工；本 ADR 合入即为 M0 完成的主体。

## 后果

- 终验口径：仓库不存在 `scripts/` 与根目录 `.ipynb`；除 deploy/ 外无 `.sh`；活代码与活文档中 `grep 'scripts/'` 零命中（adr/research/历史记录豁免）。
- 上游同步代价上升：MAISI 引擎经 maiisi_engine 收编后，与上游 NV-Generate 的 diff 基线改变——接受此代价（fork 早已有实质偏离，收益是结构可控）。
- 「历史运行器」「selftest 集群入口」两个词条含义变更，随 M0 写入 CONTEXT.md；「CLI 面保持不变」承诺终止，替代物为 namespace 等价断言所验证的子命令映射。
- 若需修改本处置：新 ADR 取代本条。
