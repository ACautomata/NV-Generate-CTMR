# ADR-0012：run contract 内部集中（AcceptanceLayer / FrozenRunBinding）——kind 七处接线单点化、validator 三胞胎组合参数化与五键六份收编

- **状态**：已接受（2026-08-25；在任何重构执行之前预注册）
- **范围**：`scripts/brats_phase_run_contract.py`（2401 行、16 类）文件内结构集中与 binding 五键跨脚本共享的钉板（wayfinder 地图 #75 候选 5 / 票 #82）。deletion test 已判「拆 CLI 层只是搬家、不拆文件」——本 ADR 是**文件内**集中，不 src 化、不拆文件；只钉决策与执行 spec，代码不动。

## 背景

kind（l1/l2/l3_report）概念在 contract 脚本内的平行接线（file:line 级核对，票列 4 处 + 本次钉板新确认暗线 3 处，共 **7 处**）：

| # | 位置 | 形态 |
|---|---|---|
| 1 | `ATTACH_KINDS:100` | 模块常量 4 元组（含 `env`），兼作 CLI `--kind` choices（`2317-2319`） |
| 2 | attach if 链 `1109-1129` | `ReportAttacher.attach` 内 3 个同构 if 块（受控目录断言＋去重＋validator＋raise） |
| 3 | `verify_l1/l3/l2_reports:1422-1453` | `RunVerifier` 三胞胎方法，仅 kind 字符串不同（`verify:1539-1541` 逐个调用） |
| 4 | `LAYER_KINDS:1229` | `FinalAcceptanceJudge` 类常量（L1/L2/L3 → kind 映射） |
| 5 | `_collect_layers` validators dict `1278-1282` | 硬编码 3 个 validator 类实例 |
| 6 | `_layer_verdict:1305-1311` | kind → 层读数路径的三分支（L1 `summary.verdict` / L2 `overall_verdict` / L3 `verdict.overall`） |
| 7 | `_layer_reasons` 入口 `1313-1343` | layer_name 三分支（分支体为各层领域逻辑） |

validator 三胞胎（`L1:515-724` / `L3:726-968` / `L2:971-1086`）的纯参数化重复：`validate()` 序幕（读文件→JSON→root dict 检查，仅 L1/L2/L3 前缀不同）3×约 10 行、五键 binding 循环 3×17 行（`537-553` / `749-765` / `1001-1017`，仅 schema 常量与提示语不同）、`_challenges` 逐字同 2×7 行（L1 `555-561` / L3 `767-773`）、`_number` 微工具 2×2 行（L1 `719-720` / L3 `967-968`）——合计约 200 行可参数化。

binding 五键（`run_id` / `phase` / `manifest_sha256` / `candidate_checkpoint_sha256` / `samples_sha256`）现有 **6 份实现、跨 4 文件**（票面列 4 份，本钉板补齐生产侧 L2）：contract 内 3 份（上表）＋ `brats_l1_quantitative.py:250-283` `L1RunBinding`（frozen dataclass，`from_run_record:261-274` 内校验 status）＋ `nnunet_l2_final_acceptance.py:190-218` `L2RunBinding`（**票面漏列**；ADR-0006 决定 2 的 `evaluate --run` 绑定实现，`__init__` 自 run.json 路径构造并校验 frozen）＋ `brats_l3_blind_eval.py:300-306` `_binding` staticmethod（`:481` 报告组装调用；**无 frozen 校验**——该检查在 `BlindPackageBuilder.build:186-187` 入口另行做）。frozen 门禁三侧不一致（L1/L2 在 binding 件内、L3 在调用入口，错误消息先序不同）。

gate 常量镜像三对（ADR-0006 裁判独立性的刻意设计，**不合并**）：FID 2.5×（contract `:580` 协议记录校验＋`:621` 阈值复算 ↔ `brats_l1_quantitative.py:314/:334/:647`）、Turing 窗 (0.40, 0.60)（contract `:118` ↔ `brats_l3_blind_eval.py:79`）、Likert 线 4.0（contract `:119` ↔ `:80`）。

硬约束：**不改 conclude 裁决行为**（[ADR-0006](0006-p1-final-acceptance-and-dm-freeze.md) 语义：非补偿 AND、读数路径、blocked_reasons 逐条追溯、裁决不可变、DmSourceLedger 取代制）；**终验路径冻结读数不动**（ADR-0002/0004）；**gate 常量镜像不合并**（ADR-0006 裁判独立性，地图 #75 Out of scope 已列）。

## 决定

1. **落点与部署形态：留 contract 脚本内 + shim import**。kind 注册表与五键共享件定义于 `brats_phase_run_contract.py` 内（遵 deletion test「不拆文件」，contract 仍是自包含部署单元）；l1/l2/l3 生产脚本以既有 shim 范式导入（`nnunet_l2_final_acceptance_nifti.py:37-44` 先例：`sys.path.insert(0, parent)` + 兄弟脚本绝对导入）。不新建 `src/ctmr/contract/`，sugon 部署形态不变（scripts/ 目录整目录同步现状保持）。contract 脚本仅依赖标准库，生产侧 import 无新增重依赖。
2. **验收层注册表（AcceptanceLayer）单点全派生**：每层一个 frozen dataclass 条目（kind 字符串、层名、报告 schema 串、validator 工厂、层读数 reader、阻塞原因 builder），模块级 `ACCEPTANCE_LAYERS` 为唯一声明点。**派生而非并存**：`ATTACH_KINDS`（= 各层 kind ＋ `"env"`，env 为非层附件、无 validator 不参与 conclude）、`LAYER_KINDS`、三个 `LX_SCHEMA` 常量全部改为自注册表派生，原常量删除。上表 7 处接线全部收编：attach if 链 → 注册表查表单循环；verify 三胞胎 → 单一 `verify_layer_reports(record)` 遍历注册表；validators dict、`_layer_verdict`、`_layer_reasons` 分支入口 → 注册表条目字段。**加新验收层 = 注册表加一项 + 新层自有类组（Rules 与 reasons builder）**，一处接线；`env` 附件与既有 `l4_report` 未知-kind 拒绝负例（selftest `:1962`）语义不变。
3. **validator 参数化 = 组合注入，非实现继承**（Python 规则：不为实现复用而继承；票面「BaseReportValidator」术语按组合形状落地）。通用 `LayerReportValidator` 外壳（序幕：读文件→JSON→root dict＋binding 校验＋分派）组合注入共享件与层 Rules 对象：
   - 共享件：FrozenRunBinding 校验（决定 4）、`_challenges` 读取、`_number`/`_sha256`/`_interval` 等微工具；
   - 层 Rules（`L1Rules` / `L2Rules` / `L3Rules`）：protocol 校验、coverage、结果行校验、verdict 复算、gate 检查——**判定逻辑全部留层**，是裁判独立性（决定 5）的结构保证。
   纯参数化可消约 200 行 → 约 60 行（序幕 3×10 + binding 3×17 + challenges 2×7 + 微工具）。
4. **冻结候选绑定（FrozenRunBinding）：五键身份提取的单一构造点，`require_frozen` 门禁内置**（提取即校验 run 状态，统一三侧不一致现状）。六份收编映射：

   | 位置 | 现状 | 处置 |
   |---|---|---|
   | contract 内 `_binding` ×3（`537-553`/`749-765`/`1001-1017`） | 五键循环三份 | 改调 FrozenRunBinding；expected dict 单一定义 |
   | `brats_l1_quantitative.py:250-283` `L1RunBinding` | dataclass 重实现（from_record） | shim import 共享件（`from_record` 入口），原类删、不保留别名 |
   | `nnunet_l2_final_acceptance.py:190-218` `L2RunBinding` | 重实现（`__init__(run_record_path)`） | shim import 共享件（`from_path` 入口），原类删 |
   | `brats_l3_blind_eval.py:300-306` `_binding` | staticmethod 重实现（无 frozen 校验） | shim import 共享件；`BlindPackageBuilder.build:186-187` 前置 frozen 检查删（门禁已内置） |

   **已注册的消息面变更**：L3 侧 frozen 未过时不再先报 build 前置消息、改报共享件门禁消息（次序与文案微变）——属消息面非判定面，收敛门禁（决定 6）按新序冻结。
5. **裁判独立性边界清单化（ADR-0006 边界的本 ADR 明文）**：
   - **可共享 = 身份**：五键提取是 run record 既有字段路径的结构契约，漂移风险在双侧失同步（现 6 份手写即暴露），共享恰消除；
   - **不合并 = 判定**：gate 常量镜像三对（FID 2.5× / Turing 窗 / Likert 线）与 verdict 复算逻辑必须双侧不同源——生产侧 bug 不得被同源校验器放行，gate 检查留层 Rules 正是其结构保证；
   - **schema 串**：contract 侧以注册表为唯一来源（原 `LX_SCHEMA` 常量删除）；生产侧 `SCHEMA`/`REPORT_SCHEMA` 独立声明**不动**——标识非判定，独立声明使版本漂移必然在校验器暴露，双侧同源反而掩盖。
6. **验收门禁（两级）与 conclude 行为逐字不变**：
   - **收敛门禁**（单元级，任意机器可跑）：重构前后 ContractSelfTest 全链输出（含全部负例的 `ContractViolationError` 消息串）逐一相等——已注册的 L3 消息次序变更（决定 4）除外；另加注册表一致性断言（层集 = L1/L2/L3 + env、顺序稳定）。CLI 面（子命令、flag、choices、用法）逐字保持（ADR-0011 先例）。
   - **集成门禁**（sugon，执行期）：`selftest` 子命令 DCU 冒烟双模式（#53 既有机制，不新造）。
   - **conclude 裁决行为（ADR-0006 语义）逐字不变**：AND 重推、三层读数路径、blocked_reasons 逐条追溯、裁决不可变、DmSourceLedger 注册/取代制——由收敛门禁覆盖验证。本 ADR 无数值产出，不涉冻结读数重跑。
7. **时机硬条：沿用 [ADR-0011](0011-phase-harness.md) 决定 8**。#59 P2 候选训练与终验完成前，**不动 sugon 受控布局上的本 ADR 触碰面**（contract 脚本 + l1/l2/l3 报告脚本——`nnunet_l2_final_acceptance.py` 正是 P2 终验链核心件）；合法落地窗口 = P2 完成后、P3 启动前。仓库侧 ADR 与代码可先备。
8. **ContractSelfTest 留驻 + 与 #83 的衔接声明**：`ContractSelfTest`（`1553-2277`，约 725 行）是 `selftest` CLI 子命令的自包含冒烟件（stdlib-only、sugon 部署单元的一部分），**留在 contract 脚本内、不迁移**；仓库 pytest 测试面（tests/ 目录、pytest 收编 selftest、CI）的钉板归 [#83](https://github.com/ACautomata/NV-Generate-CTMR/issues/83)（候选 6）另议，本 ADR 不预判其结论。

## 前置（硬阻塞）

- 本 ADR 只钉决策，**代码不动**；收编落地与两级门禁验证在执行期另行开始（新 effort / 新 issue），且遵守决定 7 时机硬条。

## 后果

- 加新验收层的接线面 7 处 → 1 处（注册表一项 + 新层自有类组）；validator 三胞胎约 200 行参数化重复收敛至约 60 行共享件；五键 6 份实现收敛为单一构造点。
- L3 生产侧 frozen 门禁消息次序微变（已注册）；其余 CLI 输出与错误消息面逐字保持。
- 生产侧 schema 串独立声明成为**已注册的允许镜像**（与 gate 镜像同属裁判独立性家族，但方向相反：gate 是刻意独立、schema 是独立声明防版本漂移）。
- 新术语「**验收层注册表 / AcceptanceLayer**」「**冻结候选绑定 / FrozenRunBinding**」入 `CONTEXT.md`。
- 若需改注册表形状、共享件落点、参数化范围、裁判独立性边界、门禁口径或时机硬条：**新 ADR 取代**，既有与本 ADR 不得混记为同一结论。
