# Deploy 运维面

`deploy/` 是本仓库的运维面（ADR-0015 §5）：部署手册、集群作业提交配方、数据获取向导与实验记录的分层住址。独立于 Python 包结构，不受 ruff 分层与 pytest 管束；launcher/nohup/torchrun 等业务编排能力属应用层，永不回栖此处（CONTEXT.md「运维面（deploy）」词条）。**`.sh` 仅存于本目录的 `jobs/` 与 `data/`，在此之外新产任何 `.sh` 都违反终态守卫。**

## 目录结构

```text
deploy/
├── README.md                       # 本手册：安装、DTK 环境、作业提交模式
├── jobs/                           # 集群作业提交配方（仅收 .sh）
│   ├── l2_calibration_predict.sh   #   #36 校准推理编排：5 子挑战 × 3 次重复，4 卡工作槽消费任务队列
│   ├── run_l2_synth_domain_eval.sh #   #38 合成域适用性评估全链：生成→组装→冻结仪器推理→指标报告
│   ├── p1_predict_all.sh           #   #38 五挑战并行仪器预测（每挑战一卡）
│   ├── run_zcrop_compensation_a.sh      #   #206 诊断作业 A：z-crop 补偿重算（测量轴归因，只读、不入 git）
│   └── run_et_discrimination_b.sh       #   #207 诊断作业 B：冻结仪器 ET 甄别（ET 缺失定量化，只读、不入 git）
├── data/
│   └── synapse_download_wizard.sh  # Synapse 数据集下载交互式向导（曙光 login 节点运行）
└── experiments/                    # 实验记录住址（书写约定见 experiments/README.md）
```

## 安装（Python 环境）

- Python **≥ 3.11**；PyPI 侧组件版本锁由仓库根 [`requirements.txt`](../requirements.txt) 承担（nnunetv2 等非 PyPI 组件不在其中，见下节版本锚表）：

  ```bash
  pip install -r requirements.txt
  ```

- 开发链路另装 pre-commit（ruff lint/format + markdownlint，合入前必绿）：

  ```bash
  pip install pre-commit && pre-commit install
  ```

- ⚠️ **DCU 集群特例**：PyPI 上的 `torch` 是 CUDA 构建，装到曙光实例会崩或静默错算——torch 及 flash_attn/triton 必须用 DCU 构建（sourcefind 渠道，见下节），其余依赖照常 pip。

## 环境（曙光 DCU · DTK）

目标集群为中科曙光 / SothisAI 曙光云（加速卡为海光 DCU，走 DTK 栈，非 NVIDIA）。任何 python/torch/GPU 操作前先做**双 source**（缺一不可）：

```bash
source /opt/dtk/env.sh                             # DTK（算）：LD_LIBRARY_PATH 等；不 source 则 import torch 直接报 libxxx.so not found
source /root/private_data/.ai_user_info/ai_proxy   # 平台 squid 代理（网）：不 source 则一切外网访问 Connection timed out
```

两条可一次性幂等写入 `~/.bashrc` 最顶部持久化（系统盘易失，实例重置后需重做）；非交互 `ssh sugon '<cmd>'` 时记得显式挂上。SSH 连接、四挂载点持久性、bashrc 注入的完整步骤见用户级 **sugon-bootstrap** skill。

### 版本锚（执行期环境锁）

| 组件 | 版本 |
|------|------|
| DTK | 26.04（`/opt/dtk/env.sh`）|
| torch-dcu | 2.9.0+das.opt1.dtk2604（py3.11，sourcefind 渠道）|
| MONAI | 1.6.0 |
| nnunetv2 | 2.8.1 |

来源：`/root/private_data/ENVIRONMENT_LOCK.md`（冻结口径见 ADR-0002）。DCU 兼容性分析与踩坑清单见 [docs/research/dcu-compatibility.md](../docs/research/dcu-compatibility.md)。

常见坑速查：

- pip 安装 ML 依赖后 numpy 可能被顶到 2.x，冲坏按 1.x 编的 DCU torch——装完立刻 `python -c "import torch"`，坏了用 `pip install "numpy==1.26.4"` 修回。
- GPU 监控用 **`hy-smi`**（等效 nvidia-smi），无 `nvidia-smi`。
- 对 sourcefind 补装 wheel 一律直链 URL ＋ `pip install --no-deps --no-index <wheel-url>`，防止 pip 回退 PyPI 拉 CUDA 包。

## 作业提交模式

SothisAI 实例独占整机、无调度器：配方 shell 直接在 DCU 计算节点上执行，两种形态：

```bash
# 前台单发（长任务建议 tmux/screen 承载）
bash deploy/jobs/run_l2_synth_domain_eval.sh all

# nohup 后台（断开 ssh 不中断；pid 可复用于状态查询）
nohup bash deploy/jobs/p1_predict_all.sh p1 > /root/private_data/<run_dir>/predict.log 2>&1 &
echo $! > /root/private_data/<run_dir>/predict.pid
```

执行前置三类（各配方的具体值以脚本头部注释为准，手册不复制集群侧绝对路径——单一事实源在脚本内）：

1. **选卡**：`HIP_VISIBLE_DEVICES=<k>` 控制该进程可见的 DCU 卡号。
2. **PYTHONPATH shim**（installable package 化完成前的过渡）：repo 形态 `export PYTHONPATH=$REPO/src`；平铺部署形态 `export PYTHONPATH=$BASE/src`。部分配方已内置此行。
3. **nnU-Net 变量组**（涉及仪器推理的配方自带）：`nnUNet_raw`／`nnUNet_preprocessed`／`nnUNet_results`／`nnUNet_compile=f`。

### 配方一览

| 配方 | 用途 | 用法 | 运行侧要点 |
|------|------|------|-----------|
| `l2_calibration_predict.sh` | #36 L2 仪器校准推理编排：15 个 (挑战×重复) 任务经 flock 文件队列被 4 卡工作槽消费，输出齐全即跳过（幂等可断点续跑） | `CALIB_BASE=<...> REPO_COMMIT=<sha> bash deploy/jobs/l2_calibration_predict.sh` | 两个环境变量必填；进度看 `$CALIB_BASE/logs/predict-status.txt` |
| `run_l2_synth_domain_eval.sh` | #38 合成域适用性评估全链：病例列表→v1 DM 直出样本→nnU-Net 输入组装→冻结仪器推理→R_fail_synth 指标报告 | `bash deploy/jobs/run_l2_synth_domain_eval.sh [p1\|p3\|all]` | 报告落 `$EVAL_ROOT/report_<mode>/`，脚本末尾自动汇总打印 |
| `p1_predict_all.sh` | #38 五挑战并行预测（每挑战一卡，TTA 保持开启），通常作为上一条 Step 3 的替代入口 | `bash deploy/jobs/p1_predict_all.sh [p1\|p3]` | 进度看 `$BASE/logs/predict-status.txt` |
| `run_zcrop_compensation_a.sh` | #206 诊断作业 A（父 #205）：L2 终验逐 case z-crop 补偿重算 `vol_wt_rel`/质心 z，产出「测量轴 vs 候选缺陷」归因读数（variant=diagnostic，不产生验收判定） | `bash deploy/jobs/run_zcrop_compensation_a.sh`（路径可用 `L2_RUN_TREE`/`MEASUREMENTS_CSV`/`PREDICT_DIR` 覆写） | 纯 CPU 重算，无需 DCU 卡；报告落运行树 `diagnostics/zcrop_compensation/`（工件区，不入 git），核心统计为带单测的纯函数 |
| `run_et_discrimination_b.sh` | #207 诊断作业 B（父 #205）：holdout 530 例生成伪四模态体已产出的逐观测仪器读数重算为逐挑战 ET 检出率、ET 体积分布 vs real、空 pred 计数（#38 读数口径同族；variant=diagnostic，不产生验收判定） | `bash deploy/jobs/run_et_discrimination_b.sh`（路径可用 `L2_RUN_TREE`/`MEASUREMENTS_CSV`/`OUTPUT_DIR` 覆写） | 零推理纯 CPU 读数，无需 DCU 卡；报告落运行树 `diagnostics/et_discrimination/`（工件区，不入 git），核心统计为带单测的纯函数 |

三个配方的仪器调用全部走 canonical 入口 `ctmr measure predict`（ADR-0009 收编，#140 迁至 `src/ctmr/infrastructure/nnunet_runner.py`），逐挑战 dataset/plans/config 由收编门禁测试钉死与 `INSTRUMENT_SPECS` 逐字一致——改 spec 请改 `src/ctmr/domain/instrument_spec.py`，勿手调配方参数。

## 数据获取

```bash
bash deploy/data/synapse_download_wizard.sh   # 在曙光 login 节点运行，交互式 5 stage
```

向导流程：网络/依赖自检 → Synapse 账号与数据集授权（如 BraTS2023 任务数据 `syn64377310`）→ Personal Access Token 写入 `~/.synapseConfig` → 下载参数（实体 ID、落地目录，默认持久盘 `/root/private_data/`）→ 生成断点续传下载脚本并 nohup 启动。中断后重跑同一向导即续传。

## 实验记录

实验结果记入 `experiments/`：一次实验一个 Markdown 文件（`YYYYMMDD-<主题>.md`），**记录落盘＝实验完成**。字段结构、命名与通行约定见 [experiments/README.md](experiments/README.md)（权威定义在根 [`CLAUDE.md`](../CLAUDE.md) 与 CONTEXT.md「实验记录」词条）。
