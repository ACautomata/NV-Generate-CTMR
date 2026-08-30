# tests/ — 测试环境纪律

## 完整 venv 安装(跑测试的硬性前置)

测试跑 full-dependency tier(ADR-0015 §6,与 CI 同配方):torch/monai/nnunetv2 层测试真实执行、不 skip。缺包的后果不是跳过,而是收集/导入失败——先装满,再跑。

在仓库根(worktree 根)执行:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # 全量安装;torch 下载慢属正常,等它装完
pip install -e .                  # ctmr 包 editable 安装 = 唯一 import 轨道(ADR-0015 §3)
```

**完成判据**——探针通过才算装好:

```bash
python -c "import torch, monai, nnunetv2, cc3d, ctmr" && echo ok
```

- 每个 worktree 都是独立路径:`.venv` 与 editable 安装不跨 worktree 共享,进入新 worktree 后按上面步骤重建。
- 复用已有 `.venv` 前,先跑探针;不过关就重跑上面两条安装命令补齐。
- 本地装的是 CPU 版 torch(PyPI 默认),够跑全部本地测试;GPU 版 torch 只在 DCU/集群上需要。

## 运行与判读

```bash
pytest
```

- `torch` 标记的测试真实执行;`gpu` 标记的自动 skip 属预期,不是故障。
- `pytest --run-gpu`:仅限 GPU/DCU 主机(opt-in 机制见 `tests/conftest.py`)。
- 验收线:pytest 退出码 0。
