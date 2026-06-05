# AGENTS.md

面向在本仓库工作的编码 agent 的执行指南。与 `Desktop/CLAUDE.md` 的通用行为准则
合并使用：本文件给**项目专属**约定，`CLAUDE.md` 给**通用**防错准则（见文末「行为准则」）。

## 项目总述

围绕 **Cola DLM**（连续隐空间扩散语言模型）的「诊断 → 改进」研究工程。
在上游 Cola 代码库上搭建一套「**数据生成 → 验证器 → 评测**」流水线，量化 Cola 在
**精确离散结构一致性**（括号匹配 / Dyck → 严格格式 → 工具调用）上的短板并归因，
为后续 RLVR 后训练改进打基础。

- **角色**：在本地修改 Cola 核心代码的工程 agent。
- **目标**：把「数据生成 → 验证 → 评测 → 服务器运行」链路搭好，让每个实验
  **可配置、可复现、可回滚**。

## 范围与硬约束

- ✅ **只做诊断与评测**；❌ 暂不涉及后训练（SFT/DPO/RL）。
  **不要创建任何后训练相关目录或配置**（`research/tests/test_scaffold.py` 有守卫断言）。
- ✅ **本地只有一个极简纯 Python 环境**，用于跑验证器 / 生成器 / 指标这类
  **不加载模型**的 lint 与单测。
- ❌ **本地绝不运行模型**：凡涉及 GPU、权重、模型采样、评测的工作**一律只在服务器执行**。
  本地只负责把对应代码 / 脚本 / 配置准备好。
- 🦴 **骨架优先**：先产出符合接口与结构约定的**骨架与说明**，**暂不写具体实现代码**。
- 🔁 **来源唯一 / 可回滚**：上游代码从 github.com/Sean-tsy/Cola-DLM 获取；研究新增
  全部以 `research/` 子目录 + 独立分支叠加，便于与上游对齐、整体回滚。
  **不改动上游模型 / 核心算法逻辑**（`cola_dlm/`、`examples/`、`scripts/run_benchmark.sh`、`docs/`）。

## 环境分工（关键）

| 位置 | 装什么 | 跑什么 |
| --- | --- | --- |
| **本地** | `requirements-test.txt`（pytest/ruff/black/jsonschema/pyyaml，**无 torch**） | 改码 · lint · `research/` 纯 Python 单测 · 提交 |
| **CI** | 同上（纯 Python） | 统一门禁：lint + 纯 Python 单测（**不导入/不跑模型**） |
| **服务器** | `requirements.lock`（含 torch 的固定闭包，GPU） | 取权重 · 模型采样 · 评测 |

**本地不要 `pip install -e .`**（会引入 torch，把本地变回模型环境）。

```bash
# 本地纯 Python 环境
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-test.txt
pytest                 # research/tests 秒级跑通；torch 模型测试自动跳过（见 conftest.py）
ruff check . && black --check .
```

依赖文件分工：

- `requirements-test.txt` —— 纯 Python 测试/lint 依赖（本地 + CI，无 torch）。
- `requirements.txt` —— 模型运行依赖（下界）。
- `requirements.lock` —— 模型/GPU 固定闭包，**仅服务器 & CI 模型作业复现，本地不安装**。

## 仓库结构（研究层）

```
research/                # 诊断研究层（叠加在上游 cola_dlm/ 之上）
├── configs/             # 实验配置（YAML，可配置/可复现/可回滚）
├── data_gen/            # 数据生成器（Dyck/括号、严格格式、工具调用）—— 纯 CPU
├── validators/          # 验证器/verifier —— 对生成结果做精确结构一致性校验
├── eval/                # 评测 harness —— 聚合指标、归因
├── scripts/             # 运行脚本（涉 GPU/权重者仅服务器执行）
├── results/             # 结果归档（小体积结构化结果；大文件不入库）
├── docs/change_map.md   # 源码改动地图：采样循环/block/VAE/DiT/解码器定位
└── tests/               # 纯 Python 单测（本地 + CI，不导入模型）
```

上游源码定位详见 [`research/docs/change_map.md`](research/docs/change_map.md)。

## 工作约定

- **数据契约**：生成数据落 `generate_task_data/*.jsonl`（已 gitignore），字段对齐
  `cola_dlm.inference` 输入（`id`/`prompt`/`question`/`context`/`choices`/`ground_truth`）。
- **新任务 prompt 模板**：需在 `cola_dlm/inference.py::apply_prompt_template` 加分支
  （属上游核心改动，谨慎、单独进行）。
- **可复现**：随机性用显式 seed 固定（参见 `COLA_INFER_PER_SAMPLE_NOISE_SEED`）。
- **大文件**：权重 / 大数据 / 本地 venv 不进版本控制（见 `.gitignore`）。
- **提交**：语义清晰、可回滚；commit 末尾附
  `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`。
- **新增纯 Python 单测**放 `research/tests/`；不要让本地单测依赖 torch / 模型。

## 完成门禁

每次改动收尾必须本地通过：

```bash
ruff check . && black --check . && pytest -q
```

## 行为准则（与 `Desktop/CLAUDE.md` 合并）

1. **先想后写**：明说假设，不确定就问；有多种解读先列出，不要默默选。
2. **简单优先**：解决问题的最小代码，不做未要求的抽象 / 配置 / 防御。
3. **外科式改动**：只动必须动的；不顺手「改进」相邻代码；匹配既有风格。
4. **目标驱动**：把任务转成可验证目标，循环到通过门禁为止。

> 本研究当前阶段尤其强调**骨架优先**与**简单优先**：按环节顺序推进，每个环节先给
> 骨架与说明，不要提前写实现。
