# ADR-0014：P3 生成链轻复制族处置——历史运行器留驻注册与 ADR-0008 收编映射部分取代

- **状态**：已接受（2026-08-26；在任何重构执行之前预注册）
- **范围**：P3/P1 生成执行链轻复制族 11 件脚本的处置钉板（wayfinder 地图 #75 衍生票 #92；ADR-0011 决定 5 划出的界外族）。本 ADR 只钉决策；处置本身为零代码动作，无执行期。

## 背景

族内 11 件（file:line 级核实，2026-08-26）：

| 件 | 行数 | 职责 |
|---|---|---|
| `scripts/batch_generate_p1.sh` / `scripts/batch_generate_p1_gpu.sh` | 69/71 | #38 P1 波 v1 DM 批量生成（串行全表 / 单挑战×单 GPU 逐行近同双版本） |
| `scripts/make_p3_jobs.py` | 84 | #38 P3 波 jobs 生成器（4 锚轮 → 8 GPU 分片 jsonl；路径硬编码、无 argparse） |
| `scripts/img2img_batch.py` | 106 | jsonl 驱动 img2img 批量推理引擎（复用 `diff_model_setting.load_config + initialize_distributed` 外壳） |
| `scripts/p3_launch_shards.sh` | 17 | nohup 分片启动器 |
| `scripts/p1_prep_inputs.py` / `scripts/p3_prep_inputs.py` | 90/107 | #38 评估 nnU-Net 输入组装（resample 1mm + crop 240×240×155，`resample_to_1mm`/`crop_or_pad` 逐字同） |
| `scripts/wait_p1_generation.sh` / `wait_p1_generation_monitor.sh` / `wait_p3_generation_monitor.sh` / `wait_predict_chain.sh` | 13/13/14/9 | 本地 ssh 轮询监控器 |

票面前提已被事实解除：

- **「防第三份」未兑现为威胁**：#60/#61 落地走正式形状而非再抄——`brats_p3_stage0_generate.py`（311 行，PR #95）与 `brats_p3_controlnet_generate.py`（266 行，PR #97）argparse 完整（`--run/--manifest/--side/--shard/--num-shards`）、run contract 绑定（freeze 先于生成）、受控存储输出、L1/L2/L3 判定链直连（`samples.json` → `nnunet_l2_final_acceptance assemble --phase P3`）。stage-0 docstring 自证 "exactly the #38 img2img chain (`load_anchor_latent` + `run_img2img`) re-pointed at the P1-DM checkpoint"——#38 链被**正式化重做**，非复制延续。
- **真 seam 从来共享**：`img2img_batch.py` 与两新生成器同用 `diff_model_setting.load_config` / `diff_model_infer.load_models` / `img2img_infer.run_img2img`；族复制的只是 main() 布置薄皮，且新形状已把这层皮正式化。族内无可收编内核。
- **使命已完结**：11 件全部只服务 #38 受控执行（sugon `l2-synth-eval` 目录两波生成＋预测链），#38 已关闭、产物冻结；仓库内引用面为零（docs/ 无引用，仅 ADR-0008/0011 提及）。
- **与 ADR-0008 交叠**：prep 对在 [ADR-0008](0008-instrument-grid-geometry.md) 决定 4 的 8 份收编映射内（`p1_prep_inputs.py:28` / `p3_prep_inputs.py:29`，linear→B-spline、留驻改造为执行入口＋IO）。ADR-0008 立案时（2026-08-25）prep 对「死活」未判；本 ADR 即判定处。

## 决定

1. **改题：不收编**。不进 PhaseHarness（[ADR-0011](0011-phase-harness.md) 决定 5 划界维持）、不另立轻外壳。11 件定位**历史运行器**（新术语，见后果）：服务单一受控执行、使命完结后留驻原地仅作追溯锚。
2. **9 件留驻＋注册**（`batch_generate_p1.sh` / `batch_generate_p1_gpu.sh` / `make_p3_jobs.py` / `img2img_batch.py` / `p3_launch_shards.sh` / `wait_p1_generation.sh` / `wait_p1_generation_monitor.sh` / `wait_p3_generation_monitor.sh` / `wait_predict_chain.sh`）：退役事实＋「禁止再抄」禁令——同用途新需求（img2img / 条件生成批量）一律走 `brats_p3_{stage0,controlnet}_generate` 正式形状。处置与 ADR-0011 对 `train_controlnet.py` 同款（已知、退役、理由注册），执行成本零。
3. **wait 轮询器不参数化**：未来 ssh 轮询监控允许一次性现写（9–14 行粒度），不沉淀参数化模板——终态判定每次不同（文件数 352 vs 1056、wave 计数、进程名），模板抽象后每次仍要填 4 个正则，收益为负。run contract（init/select/freeze/attach/conclude/verify/selftest）无进度查询子命令，未来亦不为此扩展。
4. **prep 对退役不动＋部分取代 ADR-0008 决定 4**：`p1_prep_inputs.py:28` / `p3_prep_inputs.py:29` 两行移出收编映射（**8 份→6 份**），几何不收编、linear 口径原样保持。实质理由：
   - **#38 复现原样性**：#38 的 nnU-Net 输入是 linear 口径产物；照 ADR-0008 改 B-spline 后若复跑 #38 将得到不同产物——退役＋原样不动才是对 #38 历史最忠实的处置。ADR-0008 后果节「p1/p3 linear→B-spline 有意变更」注册对此二件不再兑现；其收敛门禁对象相应缩为 6 处（其余 6 处全为活脚本）。
   - 改永不复跑的死代码违反 simplicity；真要复跑 #38，从 git 历史取原样脚本（linear 口径）即是正确路径。
   - 程序合法：ADR-0008 后果节自规定「改收编映射须新 ADR 取代」，本条即该取代（仅决定 4 映射表中 prep 对两行；ADR-0008 其余全部维持）。
5. **票面两子问随改题消解**：`make_p3_jobs.py` 硬编码/无 argparse 不参数化（退役）；prep 对位物不归 `brats_phase_prep.py`（训练侧 manifest 驱动与评估侧采样目录驱动本不同域，且退役后两个原选项——归并/收编——均失去对象）。
6. **时机**：处置＝零代码动作（留驻不动），无执行期窗口；ADR-0008 执行期照新映射（6 份）执行。不受 ADR-0011 时机硬条约束（本族不在 sugon P2 训练布局）。

## 前置（硬阻塞）

- 无。本 ADR 决议即全部落地（文档钉板），代码零变动。

## 后果

- `scripts/` 零变动；#38 追溯链（sugon 受控目录副本 ← 仓库原件 ← git 历史）不变。
- ADR-0008 收编映射 8→6 份；其「有意变更」注册与收敛门禁对象相应缩减（取代声明仅在本 ADR，ADR-0008 原文不动）。
- 新术语「**历史运行器 / legacy run orchestrator**」入 `CONTEXT.md`；「阶段脚本外壳（PhaseHarness）」词条 `_Avoid_` 行由「另立 #92」更新为既定状态指向本 ADR。
- 未来同用途需求：批量生成走正式形状；ssh 监控一次性现写；均不从本族再抄。
- 若需改本处置（复活任一件、归档移动、删除）：新 ADR 取代。
