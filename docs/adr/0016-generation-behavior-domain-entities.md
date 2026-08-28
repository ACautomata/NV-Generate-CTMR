# ADR-0016：生成行为归位领域实体

- **状态**：已接受（2026-08-28；grill-with-docs 会话产出）
- **范围**：P1/P2/P3 的 live 训练与推理规则、VAE 的仓库自有目标函数、`maiisi_engine` 项目包名纠正及相关迁移边界。

## 决定

模型训练与推理的领域规则从 `ctmr.application` 与 `ctmr.infrastructure.maisi_engine` 收编到新的 `ctmr.domain.generation` 深模块；application 保留 batch/条件张量薄适配、epoch、loader、DDP/torchrun、早停、provenance 与工件编排，infrastructure 保留文件、checkpoint、MONAI/torch 装配、配置实例化与其他 framework adapter。此决定不改变 P1/P2/P3 的任何冻结配方值、验收口径或工件契约。

`ctmr.domain.generation` 的模块映射如下：

```text
ctmr/domain/generation/
├── model.py       # DiffusionModel
├── scheduler.py   # DiffusionScheduler
├── bypass.py      # ControlNetBypass
├── update.py      # GradientExecutor Protocol
└── objective.py   # VaeObjective、肿瘤加权、模态标签扰动
```

`DiffusionModel` 是不带 checkpoint 身份的充血行为实体，拥有 `train_step`、`sample` 与 `begin_sampling`。`WeightsRef` 继续是唯一的 checkpoint 血统身份，运行时模型对象与旁路对象不得替代它。`ControlNetBypass` 是 P2/P3 的独立旁路实体，以组合方式构成条件前向路径。`DiffusionScheduler` 是每次 `sample` 调用新建的一次去噪轨迹实体，持有该轨迹的 timestep 序列和推进位置；它采样完成即终结，不持久化、不进入 checkpoint，也不构成跨运行身份。

`DiffusionModel.train_step` 声明并驱动一次完整参数更新；`GradientExecutor` 由 application 注入，以承载 fp16、bf16 或非 AMP 的运行时反传与更新策略。VAE 不另建模型实体；仓库自有 KL 与损失聚合归入 `VaeObjective`，现有 `ctmr.domain.losses` 的业务自由函数随迁移删除。

未接入 CLI 或 live application 调用链的 anatomy-size→mask DDPM adapter `ctmr.infrastructure.dataio.sample_mask` 明确不属于本次迁移范围，继续留在 infrastructure，除非未来先建立正式 application 用例。项目自有 package `ctmr.infrastructure.maiisi_engine` 更正为 `ctmr.infrastructure.maisi_engine`，直接更新活代码、测试和活文档，不保留 re-export shim；MONAI 的外部 API 符号（例如 `AutoencoderKlMaisi`）保持原拼写。

## 迁移与门禁

迁移按 M0–M5 执行：M0 为本 ADR 与术语；M1 为 `maiisi_engine`→`maisi_engine` 的纯机械更名；M2 建立 generation 领域基座；M3 迁移 P1/P2/P3 与 VAE 训练；M4 迁移 live 推理并删除被替代的 MAISI 训练/推理数学；M5 删除 `ctmr.harness`、`ctmr.instrument` shim 与其余被替代死体，并清扫活文档。每批必须通过全量 pytest 与 pre-commit；数学、训练与采样迁移须以固定 seed、CPU fp32 对旧新实现执行逐张量/逐步 latent `assert_close`。DCU 字节级重跑仍是恢复受控执行窗口后的冻结·集成门禁。

## 取代关系

本 ADR 仅部分取代 ADR-0015：替代其 §2 中生成训练/推理业务逻辑归 application/infrastructure 的住址描述，以及 §10 中相应迁移批次。ADR-0015 关于 installable package、CLI、CheckpointRepository、deploy、验收、测试分级、其余 domain 模块映射和所有既有领域口径继续有效。
