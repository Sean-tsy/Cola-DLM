# 结果归档 (`research/results/`)

约定：**一切实验 = 一份配置 + 一个 git commit**，产物按 `run_id` 归档。每个实验在
`research/results/<run_id>/` 下有一份 `manifest.json`（配置快照 + provenance）与
分类产物子目录（由 `research.experiment.ResultsArchive` 管理）。

## 目录契约

```
research/results/<run_id>/
    manifest.json   配置快照 + git_sha + 时间 + 额外 provenance（入库）
    data/           生成的探针数据 (jsonl，gitignored)
    samples/        服务器侧模型输出 (jsonl，gitignored)
    traces/         环节四诊断 JSONL trace (gitignored)
    metrics/        结构一致性指标 CSV/JSON（入库）
    plots/          图（小图入库）
    logs/           作业日志 (gitignored)
```

## 入库 vs 不入库

- **入库**（小体积、可追溯）：`manifest.json`、`metrics/`、`plots/`。
- **不入库**（大文件）：`data/`、`samples/`、`traces/`、`logs/`——放服务器侧或
  对象存储（见仓库根 `.gitignore` 与 `research/scripts/sync_results.sh`）。

每份结果通过 `manifest.json` 关联其配置、`git_sha`、seed 集合与生成时间，保证
可追溯、可复现、可回滚。
