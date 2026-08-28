# NV-Generate-CTMR

## 项目介绍

NVIDIA NV-Generate-CTMR(MAISI 3D 潜扩散 / Rectified Flow 医学影像生成)的 fork,在 BraTS2023 脑肿瘤数据上微调 `rflow-mr-brain v1`。工作按三个能力阶段推进:

- **P1** 仅以模态标签为条件的带肿瘤影像生成(全参续训 DM,VAE 冻结)
- **P2** 掩码→影像配对生成(ControlNet-only)
- **P3** 跨模态条件生成(源影像→目标模态,image-conditioned ControlNet)

每阶段经 L1 定量、L2 分布对齐、L3 专家目检三层验收,非补偿 AND——任一层不过即不过。

## 实验记录(`deploy/experiments/`)

在服务器(sugon 集群)上跑完实验后,把结果记录到仓库 `deploy/experiments/` 目录:一次实验一个 Markdown 文件,命名 `YYYYMMDD-<主题>.md`,内容含目的与关联 issue、运行配置或 run_id、关键读数与产物路径、结论与后续动作。实验以记录落盘为完成标志。(运维面职责边界见 docs/adr/0015-ddd-layered-endstate.md)

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues (ACautomata/NV-Generate-CTMR) via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default triage label vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Operational skills

操作型工作流 skill(模型下载/推理/训练)入住 `.claude/skills/<name>/SKILL.md`(Claude Code 项目级 skill,会话自动发现);**具有 live entry 的操作型 skill** 必须跟随 ADR-0015 终态(ctmr CLI 与 src.ctmr 包路径),不得指向退役入口。尚未迁入完整编排器的文件必须显式标为 reference-only,不得伪称可执行；终态守卫见 `tests/test_endstate_guards.py`。
