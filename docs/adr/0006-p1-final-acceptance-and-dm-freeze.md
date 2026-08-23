# ADR-0006：P1 完整验收与 DM 冻结执行决策（终验裁决与唯一 DM 来源）

- **状态**：已接受（2026-08-23；在任何 P1 完整终验运行之前预注册）
- **范围**：#58 P1 完整验收与冻结中，spec #51 留给执行期的落地方式：非补偿性终验裁决链、L2 报告入契约的绑定口径、通过候选作为 P2/P3 唯一 DM 来源的账本规则

## 背景

[Spec #51](https://github.com/ACautomata/NV-Generate-CTMR/issues/51) 与 [CONTEXT.md 数据划分角色](../../CONTEXT.md) 定死语义：最终验收采用非补偿性 AND，三层全部必算主判据通过才算完整 spec 终验，任一层失败即未通过，`undecided` 同样阻塞；通过候选的 DM 身份冻结为 P2/P3 的唯一 DM 来源，DM 再训使既有旁路失配。[ADR-0004](0004-l2-final-acceptance-pass-lines.md) 决定 L2 报告经 #53 run contract `attach --kind l2_report` 入 run record，但 #53 契约当时对 l2_report 附件无内容校验，也没有跨层终验裁决与 DM 来源机制。本 ADR 注册这些执行决策；不产生新测量数值。

## 决定

1. **终验裁决链（`brats_phase_run_contract conclude`）**。对 frozen run 裁决：三层正式附件（`l1_report` / `l2_report` / `l3_report`）必须各恰好一份且全部通过各自校验器——**缺失或非法附件拒绝裁决、不写裁决记录**，run 在证据补齐后仍可 conclude（与三层报告逐个生成、attach、齐备后裁决的工作流一致，也保证裁决记录只覆盖真实判定）；层读数取 L1 `summary.verdict`、L2 `overall_verdict`、L3 `verdict.overall`，三者全 `pass` 才 `pass`，任一 `fail` 或 L2 `undecided` 即 `blocked`。裁决记录（`<record-root>/final_acceptance/<run_id>.json`，schema `brats-final-acceptance/1`）不可变：重复 conclude 同一 run 拒绝；`blocked_reasons` 逐条可追溯（层 + 挑战/模态/方向 + 事实），按构造不可被其他层高分抵消；verify 重推三层 AND 并与记录核对，防裁决文件被手改。
2. **L2 报告绑定与正式口径**。`nnunet_l2_final_acceptance evaluate --run <run.json>` 把 #53 run record 身份（run_id、phase、manifest/checkpoint/samples SHA-256）写入报告 `binding`；契约 `attach --kind l2_report` 校验绑定匹配、`overall_verdict` 与逐挑战判定的非补偿一致性、round-trip 仅 P2 携带，以及**五挑战齐全且达到冻结持出配额**（`challenges_missing` / `provisional_challenges` 为空、`complete_coverage` 为 true）——provisional 运行只能留在受控目录作 smoke，不构成正式终验证据。L2 `undecided` 是合法的正式报告结论（仪器失效阻塞终验，修复方向是仪器/重跑），与 L1 的 undecided（测量未完成，不得作为正式报告）不同。
3. **DM source 账本（`<record-root>/dm_source.json`，schema `brats-dm-source/1`）**。终验 `pass` 的 P1 候选自动注册为当前 DM source：冻结其 DM 身份（selection checkpoint SHA-256）、配置（configs SHA-256）、provenance（manifest、base ckpt、code_version、run record 路径与 SHA-256）。注册为**取代制**：后续 P1 候选通过终验即取代旧 source（记录 `superseded_run_id`）；`blocked` 结论不触碰账本。
4. **旁路挂接与显式失配**。P2/P3 `init` 的 upstream 必须等于当前注册 DM source（run_id 与 checkpoint SHA-256 双匹配）；无 source 或失配即拒绝——P2/P3 只能挂在通过终验的 P1-DM 上。DM 再训练后的新候选通过终验并取代 source 后，挂旧 DM 的旁路在 `verify` 时显式失配（「DM was retrained; this bypass is pinned to a superseded DM」），旧 run record 与其历史结论保留不抹除，但不再构成当前实验链的可比结论。

## 后果

- `scripts/brats_phase_run_contract.py`（conclude 子命令、L2ReportValidator、DmSourceLedger、init/verify 接线）与 `scripts/nnunet_l2_final_acceptance.py`（evaluate `--run` 绑定）按本 ADR 实现，selftest 与五挑战全量口径的 CLI 集成冒烟覆盖；
- 终验期间仪器 checkpoint、plans、推理配置与校准包络仍不得更改（ADR-0004 decision 5）；verdict record、dm_source 账本与逐病例证据只在受控存储，不入 Git；
- 若需放宽五挑战正式口径或改变取代制语义：新 ADR 显式取代，既有裁决与账本不改写。
