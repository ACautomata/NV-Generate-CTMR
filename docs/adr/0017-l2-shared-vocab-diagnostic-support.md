# ADR-0017:L2 共享词汇分家与诊断作业支撑——stdlib-only 注册、词汇叶与统一种子注册表

- **状态**:已接受(2026-08-30;架构评审 grilling 会话产出,先于实现预注册)
- **范围**:`final_acceptance` 判官的共享词汇拆件与 interface 收缩、stdlib-only 约束的显式注册、REGIONS/Wilson 冻结词汇的单一定义点住址修订(住址级修订 ADR-0010 决定 4,单一定义点语义不变)、L2 统一种子注册表、诊断读数支撑件与诊断面分离的显式化。
- **来源**:2026-08-29 架构评审(三走查代理)候选 4/5 的 grilling 会话;接口 diff 表、冻结工件位置、gate 先例等事实核查随收编 issue spec 落盘。

## 背景

- `final_acceptance.py`(1112 行)兼任 L2 共享词汇宿主:5 个 importer 拉 15+ 符号——测量表 CSV 协议、五挑战/名额/种子注册、cluster bootstrap、统计读出寄宿判官;诊断作业(zcrop_compensation/et_discrimination/measurement_run/html_report)被迫 import 判官,判官 interface 随消费者无界增长。
- stdlib-only 仅以判官 docstring 口头纪律存在,未注册;它迫使判官持有第二份 REGIONS 字面量(`final_acceptance.py:84-85`,与 `domain/measurement/regions.py` 值全同)与私有 Wilson 复本(`:730-738`,公式/Z95 与 canonical 逐字同)。
- 事实核查:REGIONS 在三调用点+判官拆分形式共四份、值全同;canonical `ctmr.domain.measurement` 整包模块级依赖 numpy/scipy,stdlib 判官不可 import;判官 `FailureGate.wilson_upper` 函数内无守卫、调用点以 None 兜底(`metrics.py:19-21` 已注册该分工)。
- 种子两套并存:判官 bootstrap 种子注册(`final_acceptance.py:93-95`)与诊断种子命名空间(基 900,000,000、每挑战 1000 宽带,作业 A 占 0/1/100/101、作业 B 占 200)靠跨文件散文注释维持防串染,无代码不变量;诊断作业 C/P3 复用(#205 序列③)在途,复用是预期而非假设。

## 决定

1. **共享词汇拆件,住 distribution 包内**(CONTEXT.md「L2 共享词汇」词条):`measurement_table.py`(wide 27 列/long 24 列 CSV 协议)、`statistics.py`(纯统计原语 rel_diff/分位分布读出 + cluster bootstrap)、`challenge_registry.py`(五挑战/名额/统一种子注册表)。不上浮 `application/acceptance/` 层级(L1/L3 无消费者,上浮属 interface 先于调用者的投机)。
2. **stdlib-only 注册为收敛门禁性质**:判官与共享词汇零三方依赖、任意机器可裁;以 import 面测试守护,不再是 docstring 纪律。
3. **词汇叶 `ctmr.domain.vocabulary`(stdlib-only)**:REGIONS/REGION_NAMES/LABEL_DOMAIN/WilsonUpper/Z95 唯一定义于此;`ctmr.domain.measurement` 再导出,既有消费面不变。住址级修订 ADR-0010 决定 4「REGIONS/Wilson 在 measurement 唯一定义」为「唯一定义于词汇叶,measurement 再导出」——单一定义点语义保持,只是上移一层以同时服务 numpy 侧与 stdlib 侧。gate 常量仍归 ADR-0006 镜像纪律,不入词汇叶。
4. **判官收缩为纯判定链**:包络、gate、TOST、裁决、CLI;relative 分支与 Wilson 私有复本收敛引用共享件,「终验调用点自己守卫 None」的既有分工保持(已注册于 `metrics.py`)。诊断件改 import 共享词汇,不再 import 判官;判官不 import 任何诊断件(判定不同源的方向不变)。
5. **统一种子注册表**:判官 bootstrap 种子带与诊断种子命名空间同入 `challenge_registry`;「L2 全域种子无碰撞」为单测不变量(升格自「诊断互不相交」);诊断作业种子经分配器取用,禁散写裸常量;既有槽位(判官带、A/B 已用槽)以常量钉住单测逐字节复现。
6. **诊断读数支撑件 `diagnostic_support.py`**:DiagnosticError 唯一化、`variant=diagnostic` 报告写出器、P3 复用钩子(#205 序列③)落点;诊断面不进 `ctmr accept` 动词面(既有分离显式化,见 CONTEXT.md「诊断读数」词条)。

## 门禁与兑付

- 涉判官收敛(决定 4/5)以「冻结聚合 JSON 重跑逐字节一致」为兑付 gate(ADR-0010 决定 6 同款;冻结对象是聚合 JSON,逐病例 CSV 的 nan↔None/`hier_viol`→`case_usable` 改名差异已注册、不属冻结面)。gate 为**纯 CPU 测量重跑**(不重演推理),工件在 sugon 受控目录,以 sugon 纯 CPU 作业兑现(先例:诊断作业 A/B 的 sugon CPU 作业;#108 冻结验证推迟 #114 的同模式)。与 L2 测量收编(ADR-0010 决定 5 执行)共用同一集成窗口清单,含 #114 既有欠账一并清账。
- 共享件/词汇叶/支撑件本体以收敛门禁(unit 级、任意机器)验收;stdlib-only 以 import 面断言守护。

## 取代关系

- 住址级修订 ADR-0010 决定 4 的单一定义点表述(决定 3);ADR-0010 其余内容(口径数值、门禁机制、注册改名、收编清单)不动。
- 不触 ADR-0006(gate 常量镜像不合并)、ADR-0012(验收层注册表/组合注入)、ADR-0015(住址与 CLI 面)。
