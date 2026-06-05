# Cola DLM 诊断研究层 (`research/`)

本目录是叠加在上游 [Sean-tsy/Cola-DLM](https://github.com/Sean-tsy/Cola-DLM)
之上的**研究层**。所有本研究新增的代码 / 配置 / 脚本都集中在此目录下，与上游
`cola_dlm/` 核心代码隔离，便于与上游对齐、整体回滚。

## 目标

围绕 Cola DLM（连续隐空间扩散语言模型）搭建一套
「数据生成 → 验证器 → 评测」流水线，量化 Cola 在**精确离散结构一致性**
（括号匹配 / Dyck → 严格格式 → 工具调用）上的短板并归因，为后续 RLVR
后训练改进打基础。

## 范围与约束

- **当前只做诊断与评测**，暂不涉及后训练（SFT/DPO/RL）。本目录**不含**后训练相关子目录。
- **本地开发环境为纯 Python，不安装 torch、不加载模型权重、不跑模型 smoke。**
  本地只做 lint 与 `research/` 的单元测试（验证器 / 生成器 / 指标），秒级跑通。
- **凡涉及 GPU / 权重的工作（获取权重、采样、评测）一律只在服务器执行。**
  本目录中的 `scripts/` 仅负责准备好可在服务器执行的脚本，不在本地运行。

## 本地纯 Python 测试环境

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-test.txt   # pytest、ruff、black、jsonschema —— 不含 torch
pytest                                 # research/tests 秒级跑通；torch 模型测试自动跳过
```

依赖分工：

- `requirements-test.txt` —— 纯 Python 测试/lint 依赖（本地 + CI 门禁，无 torch）。
- `requirements.lock` —— 模型/GPU 锁定闭包，仅供**服务器 & CI 模型作业**复现，**本地不安装**。

研究层单测位于 [`tests/`](./tests/)（纯 Python，不导入模型）。

## 目录结构

| 子目录 | 用途 |
| --- | --- |
| `configs/` | 实验配置（`experiments/` 一实验一 YAML；`sweep/` 诊断 sweep 网格） |
| `data_gen/` | 数据生成器（Dyck/括号匹配、严格格式、工具调用） |
| `validators/` | 验证器 / verifier（对生成结果做精确结构一致性校验） |
| `instrument/` | 配置层 + 采样循环插桩 harness（环节四，默认关闭、零开销） |
| `experiment/` | 实验层：一实验=一配置（`ExperimentConfig`）+ 按 run_id 归档（`ResultsArchive`） |
| `eval/` | 评测 harness（聚合指标 + 绘图、归因分析） |
| `scripts/` | 运行脚本（数据物化纯 CPU；采样/评测/同步**仅服务器侧 GPU**执行） |
| `results/` | 结果归档（按 `run_id`：manifest + 产物；大文件不入库） |
| `docs/` | 研究文档：源码改动地图 `change_map.md`、本地↔服务器闭环 `server_workflow.md` |

## 复现性原则

每个实验都要求 **可配置、可复现、可回滚**：

- 配置集中在 `configs/`，通过版本控制追踪；
- 随机性通过显式 seed 固定（参见上游 `COLA_INFER_PER_SAMPLE_NOISE_SEED`）；
- 依赖版本分两份锁定：纯 Python 测试用 `requirements-test.txt`（本地 + CI），
  模型/GPU 运行用 `requirements.lock`（服务器 & CI 模型作业），互不混淆；
- 大权重 / 大数据不进版本控制（见根 `.gitignore`），仅放服务器侧或大文件存储。
