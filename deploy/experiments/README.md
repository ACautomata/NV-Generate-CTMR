# 实验记录（experiments）

实验是领域工件而非散放笔记（ADR-0015 决定 11）：在服务器上跑完实验后，把结果记到本目录——**一次实验一个 Markdown 文件，命名 `YYYYMMDD-<主题>.md`**，「记录落盘＝实验完成」。字段结构（目的与关联 issue、运行配置/run_id、关键读数与产物路径、结论与后续动作）以根 [`CLAUDE.md`](../../CLAUDE.md) 与 CONTEXT.md「实验记录」词条为准，此处不重复。

> 未来一切对实验记录的读取／汇总／检索须走统一 ExperimentRecord 接口（出现真实需求时实现并挂 `ctmr experiment` 子命令），不在仓库其他位置另开旁路爬文件。
