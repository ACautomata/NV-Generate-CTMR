# ADR-0013：仓库测试面钉板——pytest canonical、selftest 双薄入口、CI 轻科学栈与出生即带测试

- **状态**：已接受（2026-08-26；在执行期落地之前预注册）
- **范围**：仓库测试面的组织与边界（wayfinder 地图 #75 候选 6 / 票 #83）——tests/ 结构、既有 selftest 的 pytest 收编方式、CI 测试 step 范围、新深模块测试要求、pytest 最小配置。只钉决策与执行 spec，代码不动。

## 背景

现状（file:line 级核对，两处票面修正）：

- **零 pytest、零 conftest、零 test_\*.py**；CI（`.github/workflows/ci.yml`）仅 pre-commit（ruff，ubuntu-latest + Python 3.11）单 job。
- **内嵌 selftest 实为 13 处**（票面记 9，系钉板时点后新增）：12 个脚本带 `selftest` 子命令 + `brats_phase_dcu_smoke.py` 本身即冒烟（SmokeFixture 内嵌）。按依赖分三档：
  - **stdlib-only 6 处**：`brats_phase_run_contract.py:2524`（ContractSelfTest 约 725 行，ADR-0012 决定 8 已判留驻不迁移）、`brats_l3_blind_eval.py:703`、`nnunet_l2_final_acceptance.py:1307`（判定链）、`brats_p1_replay_prep.py:603`、`brats_p3_stage0_manifest.py:411`、`brats_p3_controlnet_manifest.py:478`；
  - **stdlib+Pillow 1 处**：`brats_p1_l2_html.py:694`（渲染器，自称「no numpy/NIfTI」）；
  - **科学栈/重型 6 处**：`brats_l1_quantitative.py:1085`（nibabel/numpy/skimage，无 torch/monai——**票面把 l1 计入 stdlib 判定链，实带三个非 stdlib 顶层依赖**）、`brats_p1/p2/p3_dev_eval.py`（import 级依赖 torch/monai，执行不用 GPU）、`brats_phase_prep.py:787`（monai）、`brats_phase_dcu_smoke.py`（需真 GPU）。
- **两级门禁已在 ADR-0008~0012 逐份存在**，每份的收敛门禁均自称「单元级，任意机器可跑」（ADR-0008 点名需 SimpleITK）——天然的 pytest/CI 承载物；集成门禁（ADR-0011/0012）已定义为 sugon 上 selftest 子命令全链 + dcu_smoke，现有机制不新造。
- **sugon 部署模型**：受控目录只复制 scripts/ 子集，集群无 pytest，tests/ 不上集群——selftest 子命令是集群侧唯一自检入口。
- `pyproject.toml` 首行注释「no installable package」，仅 ruff 配置、无 pytest 段；`requirements.txt` 漏 SimpleITK/cc3d/Pillow/nnunetv2；`scripts/__init__.py` 存在（仓库根可直接 `import scripts.x`）。
- 硬约束：**不触生产代码**、sugon 零改动、**不改变终验路径冻结读数**（ADR-0002/0004）。

## 决定

1. **pytest 为唯一 canonical 测试形态（增量规则）**：新增测试一律只以 pytest 形态写进 tests/，不再新增任何 selftest 模式。**唯一例外**：依赖 GPU/集群执行环境的冒烟（dcu_smoke 类、nnUNet 推理链）仍以 selftest/冒烟子命令形态新增，归 sugon 集成门禁——分界线是「能否在任意机器跑」，与两级门禁的收敛（任意机器）/集成（sugon）切分同构。
2. **存量 selftest 收编 = 双薄入口、单一驻留实现**：13 处 selftest 的实现原地不动（ADR-0012 决定 8 对 ContractSelfTest 已判）；tests/ 内每处一个薄 wrapper 直接调用驻留实现；`selftest` 子命令照旧。子命令**不转发 pytest**——sugon 无 pytest、tests/ 不上集群，转发即坏集成门禁。随 ADR-0008~0011 执行期收编，selftest 逻辑自然迁入 src/ctmr 模块并被 pytest 面承接，双轨收敛。
3. **tests/ 两层结构，第二层按域模块**（非代码存放位置）：

   | tests/ 第二层 | 承接 |
   |---|---|
   | `tests/contract/` | contract selftest wrapper + ADR-0012 收敛门禁断言 |
   | `tests/final_acceptance/` | final_acc 判定链 selftest wrapper |
   | `tests/l1/` `tests/l3/` | l1/l3 selftest wrapper |
   | `tests/replay/` | p1_replay_prep wrapper |
   | `tests/report/` | p1_l2_html 渲染器 wrapper |
   | `tests/dev_eval/` | p1/p2/p3_dev_eval 三兄弟 wrapper（torch 级，CI skip） |
   | `tests/phase/` | phase_prep wrapper（torch 级）；dcu_smoke 不收编（GPU 级） |
   | `tests/p3/` | stage0/controlnet manifest wrapper |
   | `tests/grid/` `tests/instrument/` `tests/measure/` `tests/harness/` | src/ctmr 四模块收敛门禁测试（ADR-0008~0011，出生即带） |

   命名随 pytest 默认（`test_*.py`）；wrapper 粒度为**一 selftest 一 test 函数**（粗粒度，拆细执行期按需）；合成 fixture（SmokeFixture 式：合成非 subject id、内存合成、不依赖外部数据）初始各测试自含，不先建共享 conftest 库，待重复出现再提炼。
4. **CI 测试 job = 轻科学栈档**：现有 pre-commit job 旁新增 test job（ubuntu-latest、Python 3.11，显式 `pip install pytest numpy nibabel scikit-image Pillow SimpleITK`——不用 requirements.txt，它含 torch 全家桶）。覆盖：6 处 stdlib-only wrapper + l1 + report 渲染器 + src/ctmr 收敛门禁测试（ADR-0008 需 sitk）。torch 级 wrapper（dev_eval/phase）以 `pytest.importorskip` 在无 torch 环境 skip，不靠 CI 装包。GPU/集群链不进 CI：本地手动 + sugon 集成门禁（ADR-0011/0012 既有归属，不新造）。
5. **新深模块出生即带测试（硬 gate）**：`src/ctmr/` 四模块（grid/instrument/measure/harness）首个落地 PR 必须同时交付其收敛门禁测试，CI 即刻生效，无测试不合入——「事后补测试」已被 10 个零 selftest 漂移文件证明从不发生。
6. **pyproject 最小配置，维持 no installable package**：新增 `[tool.pytest.ini_options]`（`testpaths=["tests"]`、`pythonpath=["src", "."]`——pytest≥7 原生支持，免 conftest sys.path hack、免 install；src/ 落地前该路径空置无害）、markers 声明。CI 依赖用 ci.yml 显式列表，不动 package 声明。`requirements.txt` 只补 CI 档用到且生产确实 import 的缺口（Pillow、SimpleITK）；cc3d/nnunetv2 留 sugon 侧不动，不在本 ADR 展开。
7. **范围边界：漂移重文件不逐文件钉测试清单**。几何/测量/命令构造三族已由 ADR-0008/0009/0010 的收编+收敛门禁承载；未收编遗留（`compute_fid_2-5d_ct.py` 归一化族纯函数、sugon 执行壳、shell 编排）是否补测试 = 执行期自便，本 ADR 不列清单、不设要求。

## 前置（硬阻塞）

- 本 ADR 只钉决策，代码不动；tests/ 落地与 CI 接线在执行期另行开始（新 effort / 新 issue）。落地物全部为新增文件（tests/、ci.yml step、pyproject 段、requirements 两行）——不触生产代码、sugon 零改动、不涉冻结读数，无 ADR-0011 式时机硬条；若与 ADR-0008~0011 执行期收编相交（dev_eval 族、contract），wrapper 随收编一并迁移即可。

## 后果

- 仓库测试面从「13 处内嵌 selftest、CI 仅 lint」变为「pytest canonical + CI 轻科学栈档」；selftest 子命令降级为 sugon 集群兼容入口，存量实现零迁移成本。
- 收敛门禁（任意机器可跑）获得 CI 常驻承载；冻结/集成门禁维持 sugon 侧不变。
- 新术语「**canonical 测试面**」「**selftest（集群兼容入口）**」「**两级门禁**」入 `CONTEXT.md`。
- 若需改 pytest 形态边界、tests/ 分层原则、CI 依赖档位或出生规则：**新 ADR 取代**，既有与本 ADR 不得混记为同一结论。
