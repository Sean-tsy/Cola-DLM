# 实验层 (`research/experiment/`)

约定：**一切实验 = 一份配置 + 一个 git commit**。`research/configs/experiments/`
下每个 YAML **唯一描述**一个实验；产物按 `run_id` 归档到 `research/results/`。

纯 Python、无模型：本层负责**配置 + 数据物化（CPU）+ 产物归位契约**；模型采样 /
评测在服务器侧执行。

## 1. `ExperimentConfig` —— `config.py`

一个实验的唯一真相，字段覆盖完成要求点名的全部内容：

- `run_id`：唯一、可追溯的实验标识（须等于配置文件名 stem，便于对齐）。
- `task` + `data`：探针类型与其数据生成旋钮（Dyck 的 `length` L / `max_depth` D /
  `k` / `max_pairing_distance` / `mode`；structured 的 `fmt` / `max_depth` /
  `max_props`；tool_call 无 data）。
- `block_size` / `timestep_num`（+ 解码旋钮）：采样配置，`sweep_for(seed)` 组合成
  环节四的 `SweepConfig`。
- `checkpoint`：DiT / VAE / tokenizer 路径，用 `${VAR}` 占位（服务器侧注入，
  不把绝对路径 / 大文件写进版本控制）。
- `seeds`：随机种子**集合**，实验在其上复现。

关键方法：`run_id_for(seed)`（=`<run_id>.s<seed>`，归档单元）、`sweep_for(seed)`、
`materialize_data(seed)`（纯 CPU 物化探针数据，相同 `(配置, seed)` → 相同数据）、
`to_dict()` / `load_experiment(path)`。未知键显式报错。

## 2. 产物归档 —— `manifest.py`

`ResultsArchive(run_id, root="research/results")` 定义每个实验产物的落位契约：

```
research/results/<run_id>/
    manifest.json   配置快照 + provenance（git sha / 时间 / 额外信息）
    data/           生成的探针数据 (jsonl, gitignored)
    samples/        服务器侧模型输出 (jsonl, gitignored)
    traces/         环节四诊断 JSONL trace (gitignored)
    metrics/        结构一致性指标 (入库)
    plots/          图 (小图入库)
    logs/           作业日志 (gitignored)
```

`ensure()` 建目录、`write_manifest(config)` 写带 git sha 的清单、`read_manifest()`
回读供评测聚合。大产物由仓库根 `.gitignore` 排除，只入库 manifest / metrics / plots。

## 配套

- 配置示例：[`../configs/experiments/`](../configs/experiments)（Dyck / JSON / tool-call）。
- 数据物化驱动（本地 CPU 可跑）：`python -m research.scripts.gen_data <config.yaml>`。
- 服务器作业 / 同步骨架：[`../scripts/`](../scripts)（`run_experiment.sh` / `sync_results.sh`）。
- 单测：[`../tests/`](../tests)（`test_experiment_config.py` / `test_results_archive.py`）。
