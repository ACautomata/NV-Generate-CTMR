# ADR-0019：分层依赖守卫与并行拓扑——application 零 infrastructure import、组合根、周期验证与 VAE 复活

- **状态**：已接受（2026-08-30；grill-with-docs 会话产出，spec #267 B0）
- **范围**：整仓分层 import 依赖方向与守卫、组合根住址、domain 端口新增、训练拓扑口径、周期验证形态、VAE 训练复活与 DDP 化、早停时序修复、散件设备注入清理；迁移批次 B0–B3 与门禁。
- **取代性声明**：
  - **部分取代 ADR-0015**：§2 甲案分层包映射——顶层新增装配模块 `wiring`（与 `cli.py` 并列、三层之外）；§3 中 cli 职责细化为「纯分派，装配上收组合根」。其余全部维持。
  - **部分取代 ADR-0016**：决定中「`GradientExecutor` 由 application 注入」一句修订为「由组合根注入 application」——application 对精度执行器只经 domain 端口类型依赖，具体类选择与构造发生在组合根。其余全部维持。
  - **不改变**：ADR-0005/0007 训练配方值、ADR-0002/0004 冻结读数、ADR-0012 验收层注册、ADR-0017 共享词汇与诊断裁决、ADR-0015 §4 CheckpointRepository 辖区与 payload schema、maiisi_engine 冻结拷贝豁免、ADR-0015 §7 命名精简规则。

## 背景

- 痛点（2026-08-30 grilling 会话）：① application 层 46 条 import 边、21 个文件直接引用 infrastructure（精度执行器具体类、DM source 账本、引擎加载/推理链、checkpoint 仓储、日志/配置装配），违反 DDD 分层依赖规则——application 应只编排领域服务与聚合根；② 训练期验证由单卡 dev watch/select sidecar 串行执行、VAE GAN 训练循环无 live 入口且零分布式，多卡服务器资源未充分利用。
- 审计事实（2026-08-30 探查）：三族候选训练主路径（P1 DM 续训 / P2 掩码 ControlNet / P3 跨模态 ControlNet）**已是完整 torchrun+DDP 实现**并以 world_size=7 实跑（20260829 三份实验记录），不在迁移范围，仅加固；真正单卡路径 = dev-eval sidecar（设计上 reserved GPU）、vae_train（调用者随 scripts 层退役）、诊断/推理散件硬编码 `cuda:0`。另有早停 mid-epoch 路径跨 rank 时序缺陷（各 rank 到达时序不同时跳过 all_reduce 的挂起/报错风险）。
- 既有资产：domain 已有 `GradientExecutor` Protocol 但 application 从不 import 它（端口闲置，依赖倒置只做了一半）；终态守卫套件四道门均不守 import 方向；全仓库无 resume 能力、无梯度累积，per-GPU batch=1 固定。
- 决策树：grilling 会话 11 个决策（依赖倒置手法、组合根住址、守卫形态、批次顺序、并行化范围、批口径、验证形态 M1、验收口径、ADR 预注册），全量见 spec #267。

## 决定

### 1. 分层依赖方向与守卫（四方向 + 豁免清单）

- 唯一合法方向：application→domain 端口；infrastructure→domain（实现端口）；组合根→一切；接口层（cli）→组合根＋application。
- 禁止：application→infrastructure、domain→任何上层、infrastructure→application。
- 终态守卫套件新增第 5 道门：源码 import 扫描按四方向判定，豁免清单＝{`ctmr.wiring`, infrastructure 自身, tests}；正探针扫真实仓库必须干净，负探针种子违规必须检出（守不住违规的守卫不是守卫）。
- B1 迁移期内以**棘轮**护航：守卫携带当前 46 处违规的冻结清单——清单外新增违规立即红，迁移只许缩减清单（只降不升）；棘轮归零后拆除、守卫转纯终态。

### 2. 组合根 `ctmr.wiring`

- 顶层装配模块（与 `cli.py` 并列、三层之外），按子命令族组织装配函数；具体实现的知识唯一定居于此。
- `cli.py` 纯分派：不 import infrastructure，只调用装配函数；torchrun worker 入口复用同一装配函数。家族入口（各模块 `main`）为拿取运行时协作者直接咨询组合根装配——与 worker 复用同一条款，application→wiring 这一方向不计为跨层违规、守卫豁免之（B1 执行期 #274 澄清）。
- 46 条既有违规的构造全部上收此处；执行期搬移按 ADR-0015 惯例在 PR 描述申报 from→to 映射。

### 3. domain 端口与内容归位

- domain 新增端口（Protocol）：checkpoint 仓储端口（state_dict payload 存取、tmp 原子发布、latest.json 指针语义不变——ADR-0015 §4 辖区不动）、DM source 领域面（仅通过终验的 DM 可注册、旁路只能挂注册源、DM 重训即失配等规则与违规异常上浮 domain；json 读写留 infrastructure 适配器实现该端口）、引擎加载与推理端口（模型加载、配置解析、推理原语适配面）、日志端口。
- `GradientExecutor` 端口成为 application 唯一依赖形态（见取代性声明）；与 ADR-0015 §2 原则「纯度上浮、IO 下沉」一致。
- 纯张量逻辑（如掩码去肿瘤）上浮 domain；application 中 infrastructure 异常别名写法消除，契约层自持违规类型。

### 4. 训练拓扑口径

- per-GPU batch=1 固定（配方 ADR 钉值不动），全局有效 batch＝卡数；训练 world_size 7→8。
- 既往 world_size=7 的 run 不追溯重算，provenance 的 world_size 字段区分。

### 5. 周期验证（M1）

- 训练进程内每 `--val-every N` epoch（默认 10）进入验证阶段：全部 rank 分片 dev cohort、all_gather 特征、注入的 scorer、dev_trend ledger 落盘、早停边界评估；验证期模型置 eval，完成后续训。不引入暂停/续训、不扩展 checkpoint schema。
- 持续轮询 sidecar 退役：dev watch/select 保留**离线形态**（对任意 run 的已存 checkpoint 跑验证）；ledger 增量读、record 组装、argmin/argmax 选择合同与 dev-eval 候选选择语义不变（spec #51 决定 7）。
- 训练与验证的卡使用从「7 训＋1 验」变为「8 训＋周期 8 卡验证」；docs/training.md 的 reserved-GPU 描述随 B2 改写（活文档处置沿 ADR-0015 §9）。

### 6. 早停时序修复

- 早停 mid-epoch 路径跨 rank 统一收尾（barrier 或等价机制），消除 rank 到达时序不同导致的 all_reduce 挂起/报错。
- 落地机制（#277）：本地轮询（epoch 前 + 每 batch 边界）只置 sticky 标志，任何 rank 不得提前离开 epoch；epoch 尾每 rank 固定一次 MAX all_reduce 共识——任一 rank 的观测令全 rank 一致跳过 loss all_reduce 与 checkpoint 发布、`run` 一致 break。共识的 rendezvous 性质同时对齐每 epoch 的集合流，天然吸收 rank 间批内漂移。

### 7. VAE 复活与 DDP

- 新 live 入口 `ctmr data vae-train`（VAE 属潜空间数据基建，与 encode 同族）；组合根装配。
- DDP 形态：autoencoder＋discriminator 双网络 DDP 包装、per-rank 静态分片（沿三族既有模式）、paired GradScaler per-rank、validate_epoch 分片＋all_reduce、rank0 写盘、批口径 per-GPU 固定。

### 8. 散件设备注入清理

- 诊断/推理散件的 `cuda:0` 硬编码与裸 `.cuda()` 收敛为统一设备注入；功能行为不动。

### 9. 迁移批次与门禁

| 批 | 内容 | 门禁 |
|---|---|---|
| B0 | 本 ADR ＋ CONTEXT.md 词条更新 | 文档评审 |
| B1 | 分层收口：守卫＋棘轮、domain 端口、组合根、家族逐批迁移（contract/modality-label/mask/cross-modal/distribution）、棘轮归零 | pytest 绿 ＋ 守卫双探针过 |
| B2 | 早停修复、周期验证、world_size=8、watch/select 离线形态、散件清理、gpu 等价性测试（单/多卡 loss 浮点容差、checkpoint 往返、早停时序、周期验证口径） | pytest 绿 ＋ gpu 标记测试送服务器 |
| B3 | VAE 复活（`ctmr data vae-train` ＋ DDP） | pytest 绿 ＋ 出生即带执行级测试 |

硬排序：B0→B1→B2→B3；每批合入仓库全绿（pytest ＋ pre-commit）。

## 前置（硬阻塞）

- 无。

## 后果

- 守卫生效后，未来任何跨层 import 立即在 CI 失败；组合根成为具体实现知识的唯一居所，application 用例只认识 domain 端口。
- 早停与候选选择粒度从每 epoch 变为每 N epoch 边界（选择合同不变）；sidecar 概念退役。
- 有效全局 batch 随卡数伸缩的既有语义正式钉板；world_size=8 的新 run 与既往 7 的 run 并存，冻结读数不追溯。
- 验收推理并行化、通用 resume 能力、多节点分布式仍不在范围，留待未来立项。
- 若需修改本处置：新 ADR 取代本条。
