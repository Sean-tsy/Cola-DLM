# 本地 ↔ 服务器闭环 (环节六)

GPU / 权重 / 模型采样 / 评测**只在服务器**执行；本地只**改码 + lint + 纯 Python
单测 + 提交**，并把服务器要用的代码 / 脚本 / 配置**准备好但不在本地运行**。

## 职责分工

| 环节 | 本地 | 服务器 |
| --- | --- | --- |
| 改码 / lint / 单测 | ✅ `ruff` + `black` + `pytest`（验证器/生成器/指标必过） | （CI 同样跑此门禁） |
| 数据物化（纯 CPU） | ✅ `python -m research.scripts.gen_data <cfg>` | ✅ 同脚本 |
| 权重获取 | ❌ 只写脚本不运行、不下载 | ✅ `fetch_weights.sh`（校验 sha256） |
| 模型采样 / 评测 | ❌ 不跑模型 / 不加载权重 | ✅ `submit_job.sh` → `run_experiment.sh` |
| 提交 / 版本控制 | ✅ 代码走 git | — |
| 大文件同步 | — | ✅ `sync_results.sh`（权重/数据/trace 走文件同步） |

> 本地**不跑**模型推理 / 采样 / smoke，也**不下载或加载权重**。CI 是统一门禁：
> 只 lint + 纯 Python 单测，绝不安装 torch / 加载权重 / 跑模型。

## 闭环流程

1. **本地**：改码 → `ruff check . && black --check . && pytest -q` 通过 → `git commit`。
   记录该 commit SHA（`git rev-parse HEAD`），用于服务器核对。
2. **同步**：
   - 代码：服务器 `git fetch && git checkout <sha>`。
   - 权重 / 数据：走文件同步（对象存储 / rsync），**不进 git**。
   - 核对一致：服务器 `bash research/scripts/check_sync.sh <sha>`（HEAD 与本地
     commit 一致且工作树干净才放行）。
3. **权重（服务器）**：`WEIGHTS_URL=... bash research/scripts/fetch_weights.sh`——
   下载到 `hf_models/` 并按 `weights.sha256` 校验哈希（模板见
   `weights.sha256.example`，服务器侧填真实哈希；权重与该文件均不入库）。
4. **起作业（服务器）**：用配置启动（多卡 / 集群调度）：
   ```bash
   NUM_GPUS=8 EXPECTED_SHA=<sha> LAUNCHER=srun \
     bash research/scripts/submit_job.sh research/configs/experiments/<cfg>.yaml
   ```
5. **最小先行**：先用最小配置
   [`smoke_dyck1.yaml`](../configs/experiments/smoke_dyck1.yaml)（最短 Dyck-1、
   `timestep_num=2`、`block_size=1`、8 样本）在服务器跑通整条「采样 → 评测」链路，
   确认 samples / traces / metrics 正常产出后，再逐级放大 L / D / 块长与规模
   （改实验配置 = 新实验 = 新 commit）。

## 复现 / 可回滚

每个实验 = 一份配置 + 一个 commit；产物按 `run_id` 归档，`manifest.json` 记录
`git_sha` + seed 集合 + 配置快照（见 [`../experiment/README.md`](../experiment/README.md)）。
`check_sync.sh` 保证服务器运行点与提交点一致，使结果可追溯、可回滚。
