# Cola DLM 诊断工程 · 服务器操作手册

本手册面向**服务器侧执行 agent**，按环节给出从零跑通「权重 → 采样 → 评测 → 归档」
整条链路的**指令**与**完成要求**，**不含具体代码**（脚本与参数以仓库内既有脚本、
`change_map.md`、配置文件为准）；末尾附本工程的实验设计总结。

> 适用范围：**凡涉及 GPU、权重、模型采样、评测的工作都在服务器执行。** 本地只负责
> 改码 + lint + 纯 Python 单测 + 提交（见 [`server_workflow.md`](./server_workflow.md)）。

- 代码仓：`github.com/Sean-tsy/Cola-DLM`
- 研究分支：`research/diagnostics-scaffold`
- 研究层根目录：`research/`（叠加在上游 `cola_dlm/` 之上，不改核心算法逻辑）
- 已确认服务器：单节点 **8×A100-SXM4-40GB**，**CUDA 12.8**（驱动 570.158.01）
- 存储：home 盘小不放产物；工作主盘 **`/data0/users/siyuan`**，备用/溢出盘
  **`/data1/users/siyuan`**

---

## 环节〇 · 存储布局与环境搭建

- **【指令】** 在数据盘上建立隔离的 GPU 运行环境，并把所有大体积产物（含 torch 的
  venv、权重、results、各类缓存）从容量小的 home 盘重定向到数据盘；克隆代码、切到
  研究分支并锁定到与本地一致的提交。
- **【完成要求】**
    - **工作根落 data0**：代码仓克隆到 `/data0/users/siyuan` 下，venv 建在仓库目录内；
      data0 为工作主盘，data1 作备用/溢出。**home 盘不落任何产物。**
    - **缓存重定向**：pip、HuggingFace、临时文件（`TMPDIR`）均指向 data0，避免默认写
      `$HOME` / `/tmp` 把 home 写满；每个新会话开头先设好这些环境变量。
    - **GPU 依赖锁定**：安装 `requirements.lock`（含 torch 的固定闭包，仅服务器 & CI
      模型作业用）；`requirements-test.txt`（纯 Python，无 torch）只供本地/CI 门禁，
      服务器无需安装；**不要 `pip install -e .`**。
    - **CUDA 匹配**：本节点 CUDA 12.8，与 PyTorch `cu124`/`cu121` 轮子兼容；若锁文件
      torch 与本机 CUDA 不匹配，按本机 CUDA 装对应 torch，其余依赖保持锁定。
    - **提交锁定**：切到 `research/diagnostics-scaffold` 并记录 `EXPECTED_SHA`（或由本地
      告知的具体 SHA），后续一致性核对与实验 manifest 均以此为准。
    - **环境自检**：确认 `torch.cuda.is_available()` 为真、可见卡数符合预期、
      `nvidia-smi` 正常（自检**不加载权重**）。

---

## 环节一 · 权重获取（服务器侧）

- **【指令】** 用脚本目录下的取权重脚本拉取并校验 Cola 权重；权重不进版本控制。
- **【完成要求】**
    - **校验清单先行**：由 `weights.sha256.example` 复制出 `weights.sha256` 并填入发布方
      公布的真实哈希；下载后**必须 `sha256sum -c` 校验通过才算就绪**。
    - **目标布局**：权重落到对齐上游 `scripts/run_benchmark.sh` 的默认布局
      （`cola_dlm/cola_dit`、`cola_dlm/cola_vae`、`tokenizer.json`）；仓库在 data0 内，
      故权重天然落 data0。如需独立于仓库另放，用脚本的 `WEIGHTS_DEST` 指到数据盘。
    - **来源可换**：取权重脚本的下载命令是骨架（`huggingface-cli` / `aws s3` 等），按
      实际权重源替换 `TODO(server)` 行；`WEIGHTS_URL` 指向真实来源。
    - **路径导出**：导出 `DIT_PATH` / `VAE_PATH` / `TOKENIZER_PATH` 供后续环节与配置占位符
      注入。
    - **不入库**：权重已被 `.gitignore` 排除，确认不会被提交。

---

## 环节二 · 代码同步与一致性核对

- **【指令】** 代码走 git、大权重/数据走文件同步；起任何作业前先核对服务器运行点与
  本地提交一致。
- **【完成要求】**
    - 服务器切到本地告知的 `EXPECTED_SHA`，用 `check_sync.sh` 核对：**HEAD 与期望 SHA
      不一致、或工作树有未提交改动时必须非零退出**，阻止在不可复现状态下起作业。
    - 核对通过后，实验 `manifest.json` 里记录的 `git_sha` 才可信、可回滚。

---

## 环节三 · 数据物化（纯 CPU）

- **【指令】** 用实验配置生成探针数据并写 manifest，按 `run_id` 归档；采样前完成数据
  与上游输入契约的对接。
- **【完成要求】**
    - **物化与归档**：用 `gen_data` 按配置物化全部 seed 的数据，产物落
      `research/results/<run_id>/`（`data/seed*.jsonl` + `manifest.json`，含配置快照与
      `git_sha`）；同一 `(配置, seed)` 必须产出完全相同的数据（固定随机种子、可复现）。
    - **⚠️ prompt 字段对接（采样前必须处理）**：上游 `apply_prompt_template` 对未登记的
      `task_name` 走兜底分支，从输入记录的 **`question`** 字段取提示词；而本工程数据把
      提示词放在 **`prompt`** 字段。**采样前须把每行 `prompt` 复制到 `question`**
      （推荐，不改核心）；或在 `apply_prompt_template` 为 `dyck`/`structured`/`tool_call`
      增分支（属上游核心改动，需单独评审）。否则提示词为空。
    - 数据记录字段为 `id` / `prompt` / `ground_truth` / `meta`，与
      `cola_dlm.inference` 输入契约对齐。

---

## 环节四 · 最小冒烟：先跑通整条链路

- **【指令】** 先用最小配置 `smoke_dyck1`（最短 Dyck-1、`timestep_num=2`、
  `block_size=1`、8 样本、单 seed）在**单卡**跑通「采样 → 评测」链路，确认无误再放大。
- **【完成要求】**
    - **单卡采样**：用上游推理入口对 `smoke_dyck1` 数据采样，输出落
      `results/smoke_dyck1/samples/`（每行含 `prompt`/`generate`/`ground_truth`）。注意
      使用真实参数名（`--dit_path`/`--vae_path`/`--tokenizer_path`/`--input_jsonl`/
      `--output_dir`/`--task_name`/`--batch_size`/`--max_new_tokens`/`--timestep_num` 等）。
    - **可复现噪声**：固定 `COLA_INFER_PER_SAMPLE_NOISE_SEED`。
    - **诊断插桩按需开启**：`COLA_DIAG_TRACE` 依赖采样循环里已落地 `research/instrument`
      的守卫式 hook（默认关闭、零开销）。**若服务器代码尚未应用插入点，先不要设该环境
      变量**，按 [`change_map.md` 环节四](./change_map.md) 的「默认关闭 patch 计划」
      （A–E，含 `try/finally` 异常安全清理与首块裁剪）落地后再开启。
    - **一键封装**：也可用 `submit_job.sh`（先核对 commit 再调 `run_experiment.sh`）端到端
      跑；`run_experiment.sh` 内模型调用处的 `TODO(server)` 需填入真实推理命令，并按配置
      应用 `model_overrides`。
    - **验收**：`samples` 行数 == 输入样本数且每行 `generate` 非空；开 trace 时
      `traces/` 含 `block_start`/`ode_step`/`block_decoded`/`mismatch`/`request_finish`；
      **不开 trace 时输出应与上游基线逐行一致（无行为改变）**。

---

## 环节五 · 正式实验与放大（多卡数据并行）

- **【指令】** 冒烟通过后逐级放大 L/D/块长与规模；改实验配置 = 新实验 = 新 commit。
- **【完成要求】**
    - **仅数据并行**：本节点 8×A100-40GB，每卡放一份完整模型副本，**不启用模型/张量
      并行**。用上游原生 `--rank`/`--world_size` 做 stride 分片
      （`raw_data[rank::world_size]`），每卡输出 `<task>_rank<N>.jsonl`，**全部 rank 结束
      后合并为 `<task>.jsonl`**；`NUM_GPUS` 设为实际可用卡数（满节点=8）。
    - **多卡 trace 隔离**：每个 rank 用独立的 `COLA_DIAG_TRACE_PATH`（带 rank 后缀），
      避免 JSONL 交错。
    - **多 seed**：对配置 `seeds` 列表逐个重复（切换噪声 seed 与对应 `data/seed<seed>.jsonl`）。
    - **集群调度**：需要时用 `submit_job.sh` 的 `LAUNCHER`（Slurm/torchrun 等）提交。
    - **sweep**：H3 块长 / 步数 / H5 压缩的网格见 `research/configs/sweep/`，用
      `research.instrument.load_sweep_grid` 读取，对每个点应用其 `model_overrides` /
      `sampling` 后采样（注意 `block_size`、`patch_size` 的覆盖约束见「注意事项」）。

---

## 环节六 · 评测与产物归档 / 同步

- **【指令】** 对采样结果做结构一致性评测与归因，归档小结果、同步大产物。
- **【完成要求】**
    - **结构评测（纯 CPU 后处理）**：对 `samples/*.jsonl` 的 `generate` 用
      [验证器](../validators)做精确结构判定（Dyck→`check_brackets`、JSON→`check_json`、
      tool-call→`check_tool_call`）；对 `traces/*.jsonl` 做失配位置 / 块归因聚合，指标
      落 `metrics/`。上游通用任务准确率仍可用 `scripts/acc_calc.py`，与本工程结构指标独立。
    - **归档分级**：小结果（`manifest.json`/`metrics/`/`plots/`）入 git；大产物
      （`data/`/`samples/`/`traces/`/`logs/`）走文件同步、不入库（用 `sync_results.sh`）。
    - **一致性**：提交小结果时确保仍在与采样一致的 commit 上（必要时新建结果 commit 并
      记录关系）。

---

## 重要注意事项（采样前必读）

1. **prompt 字段对接**：未登记任务提示词取自 `question`，须先把数据的 `prompt` 复制到
   `question`，否则提示词为空（见环节三）。
2. **`block_size` 覆盖**：DiT 与 VAE 都建块因果 mask，须**同时**设 `dit.block_size` 与
   `vae.block_size` 并断言一致（见 [`change_map.md`](./change_map.md)）。
3. **`patch_size` 覆盖**：`patch_size` 已烘焙进 VAE 的 Conv/Linear 结构，**不能**靠运行时
   属性赋值切换；H5 压缩 sweep 须为每个 `patch_size` 加载匹配的权重对。
4. **插桩异常安全**：诊断 hook 须包在 `try/finally` 内，异常路径也要清理全局 probe 与
   关闭 writer，避免泄漏到下次请求。
5. **首块裁剪**：首块 `one_block_ids` 含会被裁掉的 prompt-overlap token，喂 locator 前按
   首块 prompt token 数裁剪，保证字符位置对齐最终 `generate`（见 change_map patch D）。
6. **多卡 trace**：每个 rank 用独立 `COLA_DIAG_TRACE_PATH`，避免 JSONL 交错。
7. **默认关闭**：不设 `COLA_DIAG_TRACE` 时上游逐字节不变、零开销；插桩仅只读，不引入
   任何后训练损失/奖励。
8. **可复现**：固定 `COLA_INFER_PER_SAMPLE_NOISE_SEED`；采样在干净、已提交的 git 状态下
   运行（`check_sync.sh` 把关）。
9. **磁盘溢出**：data0 将满时，把 `research/results/` 或 `hf_models/` 迁到
   `/data1/users/siyuan` 并软链接回原位（命令/脚本无需改）；随时 `df -h` 核对两盘余量。

---

## 实验设计总结

### 探针与数据集

本工程**不使用外部数据集**，全部探针数据由 `research/data_gen/` **程序化合成**、固定
随机种子可复现（同 `(seed, 参数, n)` → 同数据），并自带可由验证器交叉校验的真值。
三级探针对应「精确离散结构一致性」由易到难：

| 级别 | 探针 | 生成器 | 任务形式 | 真值 / 校验 |
| --- | --- | --- | --- | --- |
| L0 | Dyck-k 括号匹配 | `generate_dyck` | 无条件生成 / 条件补全（给未闭合前缀） | 平衡括号串；`check_brackets` |
| L1 | 严格格式 JSON / XML | `generate_structured` | 按随机 JSON Schema 产出符合实例 | schema 校验；`check_json` |
| L2 | 工具调用 tool-call | `generate_tool_calls` | 函数签名 + 意图 → 期望调用 | 函数名 / 必填参数 / 嵌套；`check_tool_call` |

可控旋钮：
- **Dyck-k**：长度 `length`(L)、最大嵌套深度 `max_depth`(D)、括号类型数 `k`、配对距离
  上界 `max_pairing_distance`、模式 `unconditional|completion`。
- **JSON/XML**：`fmt`、`max_depth`、`max_props`。
- **tool-call**：内置函数库（`get_weather` / `send_email` / `set_timer`，含必填/选填参数）；
  意图文本由采样的参数值生成。

### 诊断信号（供指标使用）

到最近合法串的编辑距离、最长合法前缀长度，以及采样循环插桩给出的：是否合法、首个
结构错误的全局字符位置、所在**生成块**、错误落在**块内 vs 块边界**（H3 归因）。

### 实验配置（`research/configs/experiments/`，一实验 = 一配置 = 一 commit）

| run_id | 任务 | 关键数据旋钮 | 块长 / 步数 | seeds | 样本数 | 用途 |
| --- | --- | --- | --- | --- | --- | --- |
| `smoke_dyck1` | dyck | L=4, D=2, k=1, 无条件 | blk=1 / steps=2 | [0] | 8 | 链路冒烟 |
| `dyck_L32_D6_k2_blk4` | dyck | L=32, D=6, k=2, 条件补全 | blk=4 / steps=16 | [1234,2025,7] | 256 | L0 诊断 |
| `json_strict_blk4` | structured | fmt=json, depth=2, props=4 | blk=4 / steps=16 | [1234,2025,7] | 256 | L1 诊断 |
| `tool_call_blk4` | tool_call | —（内置函数库） | blk=4 / steps=16 | [1234,2025,7] | 256 | L2 诊断 |

每个配置唯一描述：任务、L/D/k 等、块长、采样步数、checkpoint 占位符、seed 集合、
run_id；`checkpoint` 用 `${DIT_PATH}` / `${VAE_PATH}` / `${TOKENIZER_PATH}` 占位，服务器
侧注入。

### Sweep（`research/configs/sweep/`，诊断假设）

| 文件 | 扫描旋钮 | 取值 | 假设 |
| --- | --- | --- | --- |
| `block_sweep.yaml` | `block_size` | 1 / 2 / 4 / 8 | H3：块内 vs 块边界失配归因 |
| `step_sweep.yaml` | `timestep_num` | 4 / 8 / 16 / 32 / 64 | 步数对结构一致性的影响 |
| `vae_compression.yaml` | `patch_size` | 1 / 2 / 4 | H5：VAE / latent 压缩率影响（须换匹配权重） |

### 算力与产物

- **算力**：只做推理诊断、无训练，需求轻。**最低 1×A100-40GB 即可跑通全部诊断任务**；
  为缩短全量 sweep 墙钟时间，推荐用本节点 **8×A100-40GB 做数据并行**。诊断序列短
  （L≤32、`max_new_tokens`≤64）、batch 小，单份 DiT+VAE 副本远低于 40 GB，**不需切分**。
- **产物与复现**：每个实验产物按 `run_id` 归档到 `research/results/<run_id>/`：
  `manifest.json`（配置快照 + `git_sha` + 时间 + seed 集合）+
  `data/samples/traces/metrics/plots/logs`。小结果入库、大产物走文件同步——保证每个
  实验可由「配置 + commit」完整复现、可回滚。
