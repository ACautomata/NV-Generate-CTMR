# deploy/ — 运维面手册

部署与集群运维的分层住址（ADR-0015 §5）：独立于 Python 包与 ruff/pytest 分层管束之外。launcher/nohup/torchrun 派生等编排能力属应用层（`src/ctmr`），永不回栖此处。

## 目录结构

| 路径 | 内容 |
|---|---|
| `deploy/README.md` | 本手册：pip 安装、DTK 环境、作业提交/nohup 模式 |
| `deploy/jobs/` | 集群作业提交配方（`.sh` 仅允许居此） |
| `deploy/data/` | 数据获取运维（下载向导） |
| `deploy/experiments/` | 实验记录：一次实验一个 `YYYYMMDD-<主题>.md`，记录落盘即实验完成 |

## 1. pip 安装

前置 Python 3.11+，仓库 checkout 根目录下执行：

```bash
pip install -r requirements.txt
```

- 依赖清单与逐包说明见 [docs/setup.md](../docs/setup.md)。
- GPU 支持需另装对应硬件的 PyTorch 构建：本地 NVIDIA 参照 docs/setup.md；sugon DCU 参照下节。

## 2. DTK 环境（sugon DCU 集群）

DCU 不直接复用 PyPI 安装的 torch，环境三要点（完整调研见 docs/research/dcu-compatibility.md）：

1. torch 必须装 sourcefind 提供的 DCU 构建（如 `torch 2.9.0+das.opt1.dtk2604`），不能从 PyPI 安装。
2. 集群侧所有 `pip install` 一律加 `--no-deps`，并把 numpy 钉在 1.x（1.26.4）——否则 numpy 被升至 2.x 会当场顶坏按 numpy 1.x 编译的 DCU torch。
3. 计算会话先完成「双 source」：DTK 算力环境与平台代理网。出网（synapse、pip 外源）均依赖代理生效。

nnUNet/仪器侧注意：DTK 26.04 存在已知 compile 缺陷，保持 `export nnUNet_compile=f`（deploy/jobs 各配方已内置该开关）。

## 3. 作业提交 / nohup 模式

集群上从 checkout 根目录提交 jobs 配方：

```bash
bash deploy/jobs/synth_domain_eval.sh all       # 合成域适用性评估（#38）
bash deploy/jobs/calibration_predict.sh          # 校准推理编排（#36）
bash deploy/jobs/predict_all.sh                  # 五挑战并行仪器预测（[p1|p3]，默认 p1）
```

长任务一律 nohup 后台化，断开 ssh 不受影响：

```bash
nohup bash deploy/jobs/<配方>.sh > <日志路径> 2>&1 &
echo $! > <pid 文件>
tail -f <日志路径>
```

数据集下载使用交互向导（在 sugon login 节点运行；末段以 nohup 启动后台下载，支持断点续传）：

```bash
bash deploy/data/synapse_download_wizard.sh
```

各配方的 env 前置（如 `CALIB_BASE`、`REPO_COMMIT`）见脚本头部注释。
