# 运行脚本 (`research/scripts/`)

> ⚠️ **凡涉及 GPU / 权重的脚本（取权重、采样、评测）一律只在服务器执行。**
> 本目录在本地仅负责把脚本准备好；只有纯 CPU 的数据物化脚本可本地运行。

约定：**一切实验 = 一份配置 + 一个 git commit**，脚本以 `research/configs/experiments/`
下的实验配置为唯一输入，产物按 `run_id` 落到 `research/results/`。本地 ↔ 服务器
闭环说明见 [`../docs/server_workflow.md`](../docs/server_workflow.md)。

## 脚本清单

| 脚本 | 环境 | 用途 |
| --- | --- | --- |
| `gen_data.py` | 本地 / 服务器（纯 CPU） | 读实验配置 → 逐 seed 物化探针数据到 `results/<run_id>/data/` + 写 manifest |
| `fetch_weights.sh` | **仅服务器** | 拉取 / 放置 Cola 权重到 `hf_models/` 并按 `weights.sha256` 校验哈希（权重不入库） |
| `check_sync.sh` | 服务器 | 核对服务器 HEAD 与本地提交一致且工作树干净，再放行作业 |
| `submit_job.sh` | **仅服务器（GPU）** | 多卡 / 集群调度（local/torchrun/srun）启动一个实验，先 `check_sync` 再 `run_experiment` |
| `run_experiment.sh` | **仅服务器（GPU）** | 端到端：物化数据 → 逐 seed 注入 checkpoint/seed/trace → 模型采样 → samples + traces 入档（骨架，含 TODO 挂载点） |
| `sync_results.sh` | 服务器 | 同步产物：大文件（data/samples/traces/logs）入对象存储，小结果（manifest/metrics/plots）入 git |

## 权重获取（服务器侧）

- 来源：`WEIGHTS_URL`（HF 仓库 / 内部镜像 / 对象存储；不写进版本控制）。
- 目标：`hf_models/`（对齐上游 `scripts/run_benchmark.sh` 默认路径）。
- 校验：`weights.sha256`（服务器侧由 `weights.sha256.example` 复制并填真实哈希）；
  `fetch_weights.sh` 用 `sha256sum -c` 校验后才算就绪。
- 权重与 `weights.sha256` 均**不进 git**（见仓库根 `.gitignore`）。

## 用法

```bash
# 本地（纯 CPU）：物化数据 + 写 manifest（不跑模型）
python -m research.scripts.gen_data research/configs/experiments/dyck_L32_D6_k2_blk4.yaml

# 服务器：取权重
WEIGHTS_URL=... bash research/scripts/fetch_weights.sh

# 服务器：最小冒烟先跑通链路，再放大
NUM_GPUS=1 EXPECTED_SHA=<local-sha> \
  bash research/scripts/submit_job.sh research/configs/experiments/smoke_dyck1.yaml

# 服务器：正式实验（多卡）
NUM_GPUS=8 EXPECTED_SHA=<local-sha> LAUNCHER=srun \
  bash research/scripts/submit_job.sh research/configs/experiments/dyck_L32_D6_k2_blk4.yaml

# 服务器：同步产物
RESULTS_REMOTE=s3://bucket/cola-diag bash research/scripts/sync_results.sh dyck_L32_D6_k2_blk4 push
```

每个脚本顶部注明：是否需要 GPU、所需权重、可复现 seed。服务器侧采样请在**干净、
已提交**的 git 状态下运行（`check_sync.sh` 把关），使 manifest 的 `git_sha`
可追溯、可回滚。
