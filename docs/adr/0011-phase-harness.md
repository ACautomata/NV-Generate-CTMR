# ADR-0011：阶段脚本外壳上移 CLI 边界（PhaseHarness）——src layout 收编 P1/P2 三对对称复制脚本

- **状态**：已接受（2026-08-25；在任何重构执行之前预注册）
- **范围**：P1/P2 阶段脚本族外壳对称拷贝的统一收编钉板（wayfinder 地图 #75 候选 4 / 票 #81）。三对复制：`brats_p1_finetune.py`(382 行) vs `brats_p2_finetune.py`(385 行) 213 行逐字相同（56%）、`brats_p1_dev_eval.py`(751) vs `brats_p2_dev_eval.py`(600) 的 watch/select main() 骨架与 argparse 25/26 flag 对称复制、`brats_p1_launch_train.sh`(88) vs `brats_p2_launch_train.sh`(91) 结构全对称。本 ADR 只钉决策与执行 spec，代码不动；重构与验收在执行期另行开始。

## 背景

外壳复制的证据与已兑现漂移（file:line 级）：

- **finetune 对**：argparse 公共集 6/7 同（P1 独有 `--replay-list`，`brats_p1_finetune.py:350-353`）；torchrun WORLD_SIZE 校验逐字同（p1:358-362 / p2:361-363）；`TrainProvenanceWriter` / `P2TrainProvenanceWriter` 结构同字段异（p1:300-341 / p2:309-348）；epoch 循环＋早停轮询（epoch 边界＋mid-epoch）、checkpoint tmp 原子发布＋`latest.json`、`_git_commit()` 均对称（p1:277-297 / p2:290-306）。
- **dev_eval 对**：P2 已 import P1 的 **13 个符号**（`brats_p2_dev_eval.py:66-80`——`CheckpointWatcher`、`DevCohortBuilder`、`EarlyStopRule`、`L2TrendRunner`、`MrTrendFeatures`、`RealReferenceBank`、`TrendFid`、`TrendLedger`、`STOP_FILE`、`COHORT_QUOTAS`、`MODALITY_TOKENS`、`TARGET_MODALITIES`、`PLANES`）——**真 seam 已存在但停在半拷贝**：领域类 import 复用了，main() 外壳（argparse 17 个 watch flag 逐字同、仅 `--emb-root` vs `--label-root` 一处差异；watch/select 调度骨架逐行对称，p1:655-747 / p2:500-596）仍是复制的。
- **launcher 对**：幂等守卫只修了 P2 侧（`brats_p2_launch_train.sh:35-37` 的 `if [ ! -f ]` 包住 environment json heredoc，防重跑覆盖导致运行契约 `--platform-json` 指纹失配）；P1 无守卫（`brats_p1_launch_train.sh:30` 直接 `cat >` 覆盖）——已兑现漂移①。
- **dtype 修复未回流**：`brats_p2_finetune.py:224` 的 `F.interpolate(labels.float(), …).long()`（DCU 上 interpolate 需 float、比较索引需 long）未回流上游底本 `train_controlnet.py:457`——已兑现漂移②。经查 `train_controlnet.py` **零生产引用**（全仓库无 import；`docs/training.md:163` 明示 P2 recipe 不走上游 loop；仅作 brats_p2_finetune 的上游参照底本与 docs/skills 通用教程入口）。
- **watch 侧 resilience 注释缺失**：P1 watch except 块多 3 行注释（`brats_p1_dev_eval.py:718-720`），P2 侧复制骨架时未带——已兑现漂移③。
- **配方守卫不对称**（票 #81 未列出、本次钉板纳入）：P2 有 `P2RecipeGuard`（ADR-0007 配方运行时守卫，`brats_p2_finetune.py:63-95`），P1 无对应物（ADR-0005 钉了配方但无运行时 guard）。
- **P3 语境**：#61 P3 未启动；`prototype/p3_image_cond_controlnet/`（838 行，wayfinder #18 原型）已是 `train_controlnet` 循环的又一次手写改写。若外壳形状先钉，P3 直接按新形状写，不再产生第三份拷贝（地图图先行原则）。P3 生成链轻复制族（prep/jobs/launch/wait 轮询器）与本候选不同类，另立 #92 收编。

硬约束：**训练配方值不动**（[ADR-0005](0005-p1-candidate-training-execution.md) / [ADR-0007](0007-p2-mask-conditioned-candidate-training-execution.md) 钉的 lr/batch/epochs/loss 权重/数据构成等全部保持）；**终验路径冻结读数不动**（ADR-0002/0004）；**gate 常量镜像不合并**（ADR-0006 裁判独立性）。

## 决定

1. **统一 module 落点与双侧全收纳界（src layout）**：阶段脚本外壳归于 **`src/ctmr/harness/`**（与 ADR-0008 `src/ctmr/grid/`、ADR-0009 `src/ctmr/instrument/`、ADR-0010 `src/ctmr/measure/` 平级）：
   - `src/ctmr/harness/cli.py` — 公共 argparse 集（`-e/-c/-t/-g/--no_amp/--amp_dtype`）＋ torchrun WORLD_SIZE 校验，阶段独有 flag（P1 `--replay-list`）由薄入口追加；
   - `src/ctmr/harness/train_shell.py` — **PhaseHarness 训练外壳**：epoch 循环、早停文件轮询（epoch 边界＋mid-epoch）、DDP 接线、loss all_reduce、iter 日志、checkpoint tmp 原子发布＋`latest.json`、provenance writer（通用骨架＋内核领域字段钩子）、`PhaseTrainKernel` Protocol（决定 2）、配方守卫钩子（决定 4）；
   - `src/ctmr/harness/dev_eval.py` — dev 评估外壳：watch/select 骨架（轮询去重、idle-exit、ledger、早停写 `.early_stop`、select argmin）＋ 13 个共享件收编（`CheckpointWatcher` / `EarlyStopRule` / `TrendLedger` / `TrendFid` / `RealReferenceBank` / `DevCohortBuilder` / `L2TrendRunner` / `MrTrendFeatures` / `MODALITY_TOKENS` / `TARGET_MODALITIES` / `COHORT_QUOTAS` / `PLANES` / `STOP_FILE`）；watch 骨架留阶段内核注入点（采样器领域件＋可选 trend 扩展，P2 回切 Dice 即此类扩展）；selftest 子命令随骨架收编、阶段领域自检件由薄入口注入；
   - `scripts/brats_p1_finetune.py` / `brats_p2_finetune.py` / `brats_p1_dev_eval.py` / `brats_p2_dev_eval.py` 退化为**薄入口**：领域内核类＋内核注入，CLI 面不变（flag 集、子命令、用法逐字保持）。
2. **内核 adapter 边界 = `PhaseTrainKernel` Protocol 四方法，组合注入、无实现继承**：
   - `build_loader()` — 数据构成（P1：BraTS＋MR-RATE 1:1 回放 `P1TrainDataCatalog`；P2：mask-conditioned `prepare_maisi_controlnet_json_dataloader`）；
   - `load_models()` — 模型挂接（P1：全参续训加载＋`ScaleFactorPolicy`；P2：冻结 DM＋`copy_model_state` ControlNet-only）＋ optimizer / lr_scheduler 对象构造（**配方值所在处归内核**：Adam vs AdamW、lr、PolynomialLR power）；
   - `train_batch(batch) -> loss` — 单 batch 前向＋损失（P1 裸 L1；P2 weighted tumor loss）；
   - `checkpoint_payload(epoch, avg_loss, scale) -> dict` — payload key（`unet_state_dict` vs `controlnet_state_dict`）。
   外壳持全部机械序列：`zero_grad → train_batch → backward → optimizer.step → scheduler.step`、loss 汇总 all_reduce、原子发布、早停轮询。
3. **launcher bash 统一为一份参数化模板** `scripts/brats_phase_launch_train.sh`（加入 `brats_phase_prep` / `brats_phase_run_contract` / `brats_phase_dcu_smoke` 的 phase 前缀族）：`PHASE=p1|p2|p3` 分派，公共骨架一份（env 校验、RUN_ROOT 布局、sidecar nohup、torchrun nohup、pid 落盘），阶段差异走 env var＋条件块（P1 replay/symlink；P2 `DM_SOURCE_CKPT` 硬前置/tblogs；P3 后续）。**幂等守卫内置模板**——漂移①守卫双侧化由此自动达成。P1/P2 旧 launcher 文件删除，受控布局文档随执行期更新。
4. **配方守卫（RecipeGuard）成为外壳一等钩子**：`PhaseHarness` 在 local_rank 0 训练启动前统一 `check()` 各阶段的 `RecipeSpec` 值对象。P2 现有 `P2RecipeGuard` 上移；**P1 按 ADR-0005 已钉配方补齐运行时 guard**（新增验证、不改任何配方值）；P3 落地时新写。
5. **P3 三件套图先行，直接按新外壳落地**（不等 P2 终验数据、不先抄再改）：`p3_finetune`（image-conditioned ControlNet 内核：四方法注入）、`p3_dev_eval`（配对误差领域件＋内核注入）、launcher 以 `PHASE=p3` 实例化模板。**P3 生成链轻复制族不属本 ADR 纳界**，另立 [钉板 P3 生成链轻复制族收编](https://github.com/ACautomata/NV-Generate-CTMR/issues/92)（#92）。
6. **已兑现漂移三条处置**：
   - ① 幂等守卫：模板内置（决定 3），双侧化自动达成；
   - ② dtype 修复**不回流** `train_controlnet.py:457`：该文件零生产引用（参照底本＋通用教程入口），修 vendor 文件只在对上游 diff 时增加本地噪音；`brats_p2_finetune` docstring 的 Deltas 节已如实记载该修复。本条注册「已知、不修、理由」，防止后人当 bug 再修；
   - ③ resilience 注释：watch 骨架入壳后 except 块唯一，注释只有一份，漂移类整体消失。
7. **收编与 import 迁移映射**：

   | 文件 | 处置 |
   |---|---|
   | `brats_p1_finetune.py` | 薄入口：`P1TrainDataCatalog` / `ScaleFactorPolicy` / `P1FinetuneJob` 领域逻辑重组为 P1 内核，注入 harness |
   | `brats_p2_finetune.py` | 薄入口：P2 内核（`P2DataCatalog` / 冻结 DM 挂接 / weighted loss）＋ `P2RecipeGuard` 上移为 RecipeSpec |
   | `brats_p1_dev_eval.py` | 薄入口：13 共享件迁 `harness/dev_eval.py`，P1 领域件（embedding 采样、spacing 源）留驻 |
   | `brats_p2_dev_eval.py` | 薄入口：13 符号 import 改自 `ctmr.harness`；`P2DevList` / `P2CandidateSampler` / `P2RoundTripDice` 等留驻 |
   | `brats_p2_holdout_generate.py:58-59` | 其对 `P2CandidateSampler` / `MODALITY_TOKENS` / `TARGET_MODALITIES` 的 import 随迁更新 |
   | `brats_p1_launch_train.sh` / `brats_p2_launch_train.sh` | 删除，并入 `brats_phase_launch_train.sh` 模板 |
   | `train_controlnet.py` | **不动**（上游参照底本） |

8. **验收门禁（两级）与时机硬条**：
   - **收敛门禁**（单元级，任意机器可跑）：收编前后，P1/P2 在相同 argv 下 argparse namespace 逐一相等；torchrun 校验 raise/no-raise 行为一致；provenance 字段集等价（`script` / `git_commit` 自指元字段除外）；checkpoint payload schema（key 集）各自保持；早停轮询语义（epoch 边界＋mid-epoch）、watch 的 pending/mark_done/should_stop 序列、`P2RecipeGuard` 行为（ADR-0007 值）不变。
   - **集成门禁**（sugon，执行期）：selftest 子命令全链＋`brats_phase_dcu_smoke` 通过（现有机制，不新造）。
   - **时机硬条**：#59 P2 候选训练与终验完成前，**不动 sugon 受控布局上的 P2 侧文件**（收编期间仓库演进不触碰运行中布局，避免 run contract 指纹与 provenance 追溯断裂）。合法落地窗口＝P2 完成后、P3 启动前。
9. **sugon 部署**：沿用 ADR-0008/0009 的 #55+ shim / src-tree 策略——薄入口绝对导入 `from ctmr.harness import …`，sugon 部署连同 `src/` 树同步、`src/` 上 `sys.path`；版本锚用 run contract 现有指纹（`git_commit`＋`script_sha256`）。

## 前置（硬阻塞）

- 本 ADR 只钉决策，**代码不动**；收编落地与两级门禁验证在执行期另行开始（新 effort / 新 issue），且遵守决定 8 时机硬条。

## 后果

- finetune/dev_eval/launcher 三对外壳复制收敛为单一定义；P3 不产生第三份拷贝（图先行收益）。
- P1 获得配方运行时守卫（ADR-0005 配方的验证强化，非配方变更）；launcher 重跑不再可能覆盖已指纹的 environment json（漂移①根除）。
- `train_controlnet.py` 与 P2 finetune 的 interpolate dtype 差异为**已注册的已知分歧**（零生产影响），不修。
- CLI 面（flag 名、子命令、用法说明）保持逐字不变，sugon 侧调用方式与文档命令不受影响；launcher 命令名变化（`brats_p{1,2}_launch_train.sh` → `brats_phase_launch_train.sh PHASE=…`）为唯一对外可见变化，受控布局文档随执行期更新。
- 新术语「**阶段脚本外壳 / PhaseHarness**」入 `CONTEXT.md`。
- 若需改 harness 纳界、内核 interface 形状、launcher 模板化决策、守卫钩子机制、门禁口径或时机硬条：**新 ADR 取代**，既有与本 ADR 不得混记为同一结论。
