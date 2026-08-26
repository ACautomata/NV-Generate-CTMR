# NV-Generate-CTMR

## 项目介绍

NVIDIA NV-Generate-CTMR(MAISI 3D 潜扩散 / Rectified Flow 医学影像生成)的 fork,在 BraTS2023 脑肿瘤数据上微调 `rflow-mr-brain v1`。工作按三个能力阶段推进:

- **P1** 仅以模态标签为条件的带肿瘤影像生成(全参续训 DM,VAE 冻结)
- **P2** 掩码→影像配对生成(ControlNet-only)
- **P3** 跨模态条件生成(源影像→目标模态,image-conditioned ControlNet)

每阶段经 L1 定量、L2 分布对齐、L3 专家目检三层验收,非补偿 AND——任一层不过即不过。

## 实验记录(`Experiments/`)

在服务器(sugon 集群)上跑完实验后,把结果记录到仓库 `Experiments/` 目录:一次实验一个 Markdown 文件,命名 `YYYYMMDD-<主题>.md`,内容含目的与关联 issue、运行配置或 run_id、关键读数与产物路径、结论与后续动作。实验以记录落盘为完成标志。

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues (ACautomata/NV-Generate-CTMR) via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default triage label vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
