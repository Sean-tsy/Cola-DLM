# Cola DLM 诊断工程 · 服务器操作手册

本手册整理在**服务器**上从零开始、按环节跑通「权重 → 采样 → 评测 → 归档」整条
链路所需的全部操作与具体命令，并在末尾总结本工程设计的实验（探针 / 数据集 /
配置 / sweep）。

> 适用范围：**凡涉及 GPU、权重、模型采样、评测的工作都在服务器执行。** 本地只负责
> 改码 + lint + 纯 Python 单测 + 提交（见 [`server_workflow.md`](./server_workflow.md)）。
> 本手册假设你已拿到一台带 NVIDIA GPU、可访问代码仓与权重源的服务器。

- 代码仓：`github.com/Sean-tsy/Cola-DLM`
- 研究分支：`research/diagnostics-scaffold`
- 基线提交（本手册编写时）：`3c710d4`
- 研究层根目录：`research/`（叠加在上游 `cola_dlm/` 之上，不改核心算法逻辑）

---

## 环节〇 · 服务器环境从零搭建

```bash
# 1) 取代码（HTTPS 或 SSH 二选一）
git clone https://github.com/Sean-tsy/Cola-DLM.git
cd Cola-DLM

# 2) 切到研究分支并锁定到与本地一致的提交（见“环节二”核对）
git fetch --all
git checkout research/diagnostics-scaffold
EXPECTED_SHA="$(git rev-parse HEAD)"   # 或由本地告知的具体 SHA
echo "server at ${EXPECTED_SHA}"

# 3) 建立 GPU 运行环境（与本地纯 Python 测试环境相互独立）
python -m venv .venv-gpu
. .venv-gpu/bin/activate
python -m pip install --upgrade pip

# 4) 安装“模型 / GPU 运行依赖”锁定闭包（仅服务器 & CI 模型作业用）
#    requirements.lock 含 torch 等重依赖；本地不安装。
pip install -r requirements.lock
```

> 注意：
> - `requirements.lock` 顶部已标注「SERVER & CI MODEL-JOB reproduction ONLY」。
> - `requirements-test.txt`（纯 Python，无 torch）只供本地 / CI 门禁，服务器**无需**用它跑模型。
> - 如服务器 CUDA 版本与锁文件中的 torch 轮子不匹配，按集群实际 CUDA 重装对应
>   torch 版本，其余依赖保持锁定。本节点为 **CUDA 12.8**（驱动 570.158.01），
>   与 PyTorch `cu124`/`cu121` 轮子向后兼容；如锁文件 torch 不匹配，装对应
>   `--index-url https://download.pytorch.org/whl/cu124` 的 torch 即可。

环境自检（不加载权重）：

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'n_gpu', torch.cuda.device_count())"
nvidia-smi
```

---

## 算力资源（本工程需求）

> 本工程**只做诊断与评测（推理 / 采样），不做任何训练**，因此算力需求**很轻**：
> 每张卡放一份完整模型副本即可，**无需模型并行 / 张量并行**，多卡仅用于**数据并行**
> 提升吞吐。

**已确认的服务器配置**（来自 `nvidia-smi`）：

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA **A100-SXM4-40GB** ×8（单节点，NVLink/SXM4） |
| 单卡显存 | 40 GB |
| 驱动 / CUDA | 570.158.01 / **CUDA 12.8** |
| 状态 | 全部空闲（0% util、0 MiB 占用） |

**本工程实际需求**：

| 用途 | GPU 需求 | 说明 |
| --- | --- | --- |
| 最小冒烟（`smoke_dyck1`，8 样本、steps=2） | **1×A100-40GB** | 几分钟级；验证链路用 |
| 单个正式实验（L0/L1/L2，256 样本、steps≤16、L≤32） | **1×A100-40GB 可跑**；**8× 推荐** | 单卡可完成；8 卡数据并行 ~8× 提速 |
| 全量 L0–L2 × 多 seed(3) × sweep(块长/步数/压缩) | **8×A100-40GB** | 纯吞吐扩展，stride 分片到 8 卡 |

要点：

- **显存充裕**：诊断序列短（L≤32、`max_new_tokens`≤64）、batch 小，单份
  DiT+VAE 副本远低于 40 GB；**40 GB 单卡即可承载完整模型**，不需切分。
- **并行方式**：仅 `--rank` / `--world_size` 数据并行（`raw_data[rank::world_size]`
  stride 分片，见环节五），**不启用任何模型并行**。给定 8 卡，`NUM_GPUS=8`。
- **CPU / 内存 / 磁盘**：数据物化、`prompt→question` 适配、验证器评测均为纯 CPU、
  内存友好；磁盘主要用于**权重**（DiT+VAE+tokenizer，按发布大小预留，建议 ≥50 GB
  空间）与 `research/results/` 产物。
- **结论**：**最低 1×A100-40GB 即可跑通全部诊断任务**；为缩短全量 sweep 墙钟时间，
  推荐用本节点的 **8×A100-40GB 做数据并行**。本手册多卡示例默认 `NUM_GPUS=8`，
  按实际可用卡数调整即可。

---

## 环节一 · 权重获取（服务器侧）

权重**不进版本控制**；由脚本拉取到 `hf_models/` 并校验哈希。

```bash
# 1) 准备 sha256 校验清单：复制模板并填入发布方公布的真实哈希
cp research/scripts/weights.sha256.example research/scripts/weights.sha256
$EDITOR research/scripts/weights.sha256     # 填 cola_dit / cola_vae / tokenizer.json 的 sha256

# 2) 拉取并校验（WEIGHTS_URL 指向 HF 仓 / 内部镜像 / 对象存储）
WEIGHTS_URL="<weight-source>" bash research/scripts/fetch_weights.sh
```

`fetch_weights.sh` 会把权重放到默认布局（对齐上游 `scripts/run_benchmark.sh`）：

```
hf_models/
  cola_dlm/cola_dit/      # DiT 先验权重目录
  cola_dlm/cola_vae/      # VAE 权重目录
  tokenizer.json          # 分词器
```

并执行 `sha256sum -c research/scripts/weights.sha256`，**校验通过才算就绪**。
脚本内的下载命令是骨架（`huggingface-cli download` / `aws s3 sync` 等），按你的
权重源替换其中的 `TODO(server)` 行。

导出后续步骤要用的路径：

```bash
export DIT_PATH="$PWD/hf_models/cola_dlm/cola_dit"
export VAE_PATH="$PWD/hf_models/cola_dlm/cola_vae"
export TOKENIZER_PATH="$PWD/hf_models/tokenizer.json"
```

---

## 环节二 · 代码同步与一致性核对

代码走 git，大权重 / 数据走文件同步；同步后**核对服务器运行点与本地提交一致**，
这样每次实验 `manifest.json` 里的 `git_sha` 才可信、可回滚。

```bash
# 服务器切到本地告知的提交，并确认工作树干净
git fetch && git checkout "${EXPECTED_SHA}"
bash research/scripts/check_sync.sh "${EXPECTED_SHA}"
# 期望输出：[check_sync] OK: server at <sha>, clean tree, matches <sha>
```

`check_sync.sh` 会在 HEAD 与期望 SHA 不一致、或工作树有未提交改动时**非零退出**，
阻止在不可复现的状态下起作业。

---

## 环节三 · 数据物化（纯 CPU，可在服务器或本地）

用实验配置生成探针数据并写 `manifest.json`，产物按 `run_id` 归档到
`research/results/<run_id>/`。

```bash
# 单个实验：物化全部 seed 的数据 + 写 manifest
python -m research.scripts.gen_data research/configs/experiments/smoke_dyck1.yaml

# 产物：
#   research/results/smoke_dyck1/data/seed0.jsonl   (探针数据，gitignored)
#   research/results/smoke_dyck1/manifest.json      (配置快照 + git_sha)
```

数据记录字段（每行一个 JSON）：`id` / `prompt` / `ground_truth` / `meta`。
同一 `(配置, seed)` → 完全相同的数据（固定随机种子，可复现）。

> ⚠️ **重要集成点（采样前必须处理）**：上游 `cola_dlm.inference` 的
> `apply_prompt_template` 对未登记的 `task_name` 走兜底分支 `return question`，
> 即它从输入记录的 **`question`** 字段取提示词；而本工程数据记录把提示词放在
> **`prompt`** 字段。两种对接方式任选其一：
>
> 1)（推荐，不改核心）采样前把每行的 `prompt` 复制到 `question`：
> ```bash
> IN=research/results/smoke_dyck1/data/seed0.jsonl
> python - "$IN" <<'PY'
> import json, sys
> p = sys.argv[1]
> rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
> for r in rows:
>     r["question"] = r.get("prompt", "")
> with open(p, "w", encoding="utf-8") as f:
>     for r in rows:
>         f.write(json.dumps(r, ensure_ascii=False) + "\n")
> print(f"patched {len(rows)} rows: prompt -> question")
> PY
> ```
> 2)（可选，改上游）在 `apply_prompt_template` 增加 `dyck` / `structured` /
>    `tool_call` 分支返回 `question`。注意这属于核心代码改动，需单独评审。

---

## 环节四 · 最小冒烟：先跑通整条链路

**先用最小配置** `smoke_dyck1.yaml`（最短 Dyck-1、`timestep_num=2`、
`block_size=1`、8 样本、1 个 seed）验证「采样 → 评测」链路无误，再放大。

### 4.1 直接调用上游推理 CLI（真实可运行命令）

```bash
RUN_ID=smoke_dyck1
SEED=0
BASE="research/results/${RUN_ID}"
mkdir -p "${BASE}/samples" "${BASE}/traces"

# 数据物化 + prompt->question 适配（见环节三）
python -m research.scripts.gen_data research/configs/experiments/${RUN_ID}.yaml
# ...（执行环节三的 prompt->question 适配片段）

# 开启可复现噪声 + 环节四诊断插桩
export COLA_INFER_PER_SAMPLE_NOISE_SEED="${SEED}"
export COLA_DIAG_TRACE=1
export COLA_DIAG_TRACE_PATH="${BASE}/traces/seed${SEED}.jsonl"

# 单卡冒烟（注意 flag 名：--dit_path/--vae_path/--tokenizer_path/--input_jsonl）
python -m cola_dlm.inference \
  --dit_path "${DIT_PATH}" \
  --vae_path "${VAE_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --input_jsonl "${BASE}/data/seed${SEED}.jsonl" \
  --output_dir "${BASE}/samples" \
  --task_name "dyck" \
  --batch_size 8 \
  --max_samples 0 \
  --max_new_tokens 16 \
  --timestep_num 2 \
  --guidance_scale 7.0 \
  --temperature 0.0
# 输出：research/results/smoke_dyck1/samples/dyck.jsonl
#       （字段 prompt / generate / ground_truth）
# 诊断：research/results/smoke_dyck1/traces/seed0.jsonl
```

> 诊断插桩（`COLA_DIAG_TRACE`）需要上游采样循环里已落地 `research/instrument`
> 的守卫式 hook（默认关闭、零开销）。若服务器代码尚未应用这些插入点，请按
> [`change_map.md` 环节四](./change_map.md) 的「默认关闭 patch 计划」（A–E 片段，
> 含 `try/finally` 异常安全清理与首块裁剪）落地后再开 `COLA_DIAG_TRACE`；
> **未应用前先不设该环境变量**，链路其余部分照常可跑。

### 4.2 用封装脚本一键跑（推荐）

```bash
# submit_job.sh 先核对 commit 再调 run_experiment.sh（端到端：数据→采样→trace）
NUM_GPUS=1 EXPECTED_SHA="${EXPECTED_SHA}" LAUNCHER=local \
  bash research/scripts/submit_job.sh research/configs/experiments/smoke_dyck1.yaml
```

`run_experiment.sh` 内模型调用处标了 `TODO(server)`：把上面 4.1 的真实
`python -m cola_dlm.inference ...` 命令填入，并按配置应用
`model_overrides`（`block_size` / `patch_size`，见下「注意事项」）。

### 4.3 冒烟验收

- `samples/dyck.jsonl` 行数 == 输入样本数，且每行含非空 `generate`。
- 设了 `COLA_DIAG_TRACE` 时，`traces/seed0.jsonl` 含
  `block_start` / `ode_step` / `block_decoded` / `mismatch` / `request_finish`。
- 不设 `COLA_DIAG_TRACE` 时，输出应与上游基线逐行一致（无行为改变）。

---

## 环节五 · 正式实验与放大（多卡）

冒烟通过后，按需逐级放大 L / D / 块长与规模。改实验配置 = 新实验 = 新 commit。

### 5.1 多卡数据并行（上游原生 stride 分片）

> 本节点 8×A100-40GB，**仅做数据并行**（每卡一份完整模型副本，无模型并行）。
> `NUM_GPUS` 设为实际可用卡数（满节点 = 8）。

上游 `cola_dlm.inference` 支持 `--rank` / `--world_size` 做数据并行：每卡跑
`raw_data[rank::world_size]`，每卡输出 `<task>_rank<rank>.jsonl`，最后合并。模式
同 `scripts/run_benchmark.sh`：

```bash
RUN_ID=dyck_L32_D6_k2_blk4
SEED=1234
BASE="research/results/${RUN_ID}"
NUM_GPUS=8

export COLA_INFER_PER_SAMPLE_NOISE_SEED="${SEED}"
# 多卡时每卡 trace 落到独立文件，避免 JSONL 交错
export COLA_DIAG_TRACE=1

PIDS=()
for GPU_ID in $(seq 0 $((NUM_GPUS-1))); do
  COLA_DIAG_TRACE_PATH="${BASE}/traces/seed${SEED}_rank${GPU_ID}.jsonl" \
  CUDA_VISIBLE_DEVICES=${GPU_ID} python -m cola_dlm.inference \
    --dit_path "${DIT_PATH}" --vae_path "${VAE_PATH}" --tokenizer_path "${TOKENIZER_PATH}" \
    --input_jsonl "${BASE}/data/seed${SEED}.jsonl" \
    --output_dir "${BASE}/samples" \
    --task_name "dyck" \
    --batch_size 20 --max_samples 0 \
    --max_new_tokens 64 --timestep_num 16 --guidance_scale 7.0 --temperature 0.0 \
    --rank ${GPU_ID} --world_size ${NUM_GPUS} &
  PIDS+=($!)
done
for PID in "${PIDS[@]}"; do wait "$PID" || { echo "rank failed"; exit 1; }; done

# 合并分片
cat "${BASE}/samples/dyck_rank"*.jsonl > "${BASE}/samples/dyck.jsonl"
rm -f "${BASE}/samples/dyck_rank"*.jsonl
```

集群调度（Slurm / torchrun）用 `submit_job.sh` 的 `LAUNCHER`：

```bash
NUM_GPUS=8 EXPECTED_SHA="${EXPECTED_SHA}" LAUNCHER=srun \
  bash research/scripts/submit_job.sh research/configs/experiments/dyck_L32_D6_k2_blk4.yaml
```

### 5.2 多 seed

对 `seeds` 列表中每个 seed 重复 5.1（`COLA_INFER_PER_SAMPLE_NOISE_SEED=<seed>`，
数据用 `data/seed<seed>.jsonl`，trace 落 `seed<seed>_rank*.jsonl`）。

### 5.3 sweep（H3 块长 / 步数 / H5 压缩）

sweep 网格见 `research/configs/sweep/`，用 `research.instrument.load_sweep_grid`
读取，对每个点应用其 `model_overrides` / `sampling` 再采样：

```python
from research.instrument import load_sweep_grid
for sw in load_sweep_grid("research/configs/sweep/block_sweep.yaml"):
    print(sw.name, sw.to_model_overrides(), sw.to_inference_kwargs(), sw.to_env())
    # 服务器侧：应用 model_overrides 到 dit/vae，按 to_inference_kwargs 调推理
```

---

## 环节六 · 评测与产物归档 / 同步

### 6.1 结构一致性评测（纯 CPU 后处理）

对 `samples/*.jsonl` 的 `generate` 字段用[验证器](../validators)做精确结构判定，
对 `traces/*.jsonl` 做失配位置 / 块归因聚合，指标落 `metrics/`。

> `research/eval/` 的指标聚合在后续环节实现；当前可直接用验证器快速核验：

```python
import json
from research.validators.stack import check_brackets
rows = [json.loads(l) for l in open("research/results/smoke_dyck1/samples/dyck.jsonl", encoding="utf-8") if l.strip()]
ok = sum(check_brackets(r["generate"]).valid for r in rows)
print(f"valid Dyck: {ok}/{len(rows)}")
```

上游通用任务的准确率打分仍可用 `scripts/acc_calc.py`（针对 benchmark 任务），
本工程的结构一致性指标独立于此。

### 6.2 归档与同步

- 小结果（`manifest.json` / `metrics/` / `plots/`）入 git。
- 大产物（`data/` / `samples/` / `traces/` / `logs/`）走文件同步、不入库：

```bash
RESULTS_REMOTE=s3://bucket/cola-diag \
  bash research/scripts/sync_results.sh dyck_L32_D6_k2_blk4 push
```

提交小结果时确保仍在与采样一致的 commit 上（必要时新建结果 commit 并记录关系）。

---

## 重要注意事项（采样前必读）

1. **prompt 字段对接**：未登记任务的提示词取自 `question` 字段，须先把数据的
   `prompt` 复制到 `question`（见环节三片段），否则提示词为空。
2. **`block_size` 覆盖**：不是仅改 `dit.block_size`——DiT 与 VAE 都建块因果 mask，
   须**同时**设 `dit.block_size` 与 `vae.block_size` 并断言一致（见
   [`change_map.md`](./change_map.md)）。
3. **`patch_size` 覆盖**：`patch_size` 已烘焙进 VAE 的 Conv/Linear 结构，**不能**
   靠运行时属性赋值切换；H5 压缩 sweep 须为每个 `patch_size` **加载匹配的权重对**。
4. **插桩异常安全**：诊断 hook 须包在 `try/finally` 内，异常路径也要
   `clear_probe()` + 关闭 writer，避免全局 probe 泄漏到下次请求。
5. **首块裁剪**：首块 `one_block_ids` 含会被裁掉的 prompt-overlap token，喂
   locator 前按 `first_block_prompt_token_counts` 裁剪，保证字符位置对齐最终
   `generate`（见 change_map patch D）。
6. **多卡 trace**：每个 rank 用独立 `COLA_DIAG_TRACE_PATH`（带 `rank` 后缀），
   避免 JSONL 交错。
7. **默认关闭**：不设 `COLA_DIAG_TRACE` 时上游逐字节不变、零开销；插桩仅只读，
   不引入任何后训练损失 / 奖励。
8. **可复现**：固定 `COLA_INFER_PER_SAMPLE_NOISE_SEED`；采样在干净、已提交的 git
   状态下运行（`check_sync.sh` 把关）。

---

## 实验设计总结

### 探针与数据集

本工程**不使用外部数据集**，全部探针数据由 `research/data_gen/` **程序化合成**、
固定随机种子可复现（同 `(seed, 参数, n)` → 同数据），并自带可由验证器交叉校验的
真值。三级探针对应「精确离散结构一致性」由易到难：

| 级别 | 探针 | 生成器 | 任务形式 | 真值 / 校验 |
| --- | --- | --- | --- | --- |
| L0 | Dyck-k 括号匹配 | `generate_dyck` | 无条件生成 / 条件补全（给未闭合前缀） | 平衡括号串；`check_brackets` |
| L1 | 严格格式 JSON / XML | `generate_structured` | 按随机 JSON Schema 产出符合实例 | schema 校验；`check_json` |
| L2 | 工具调用 tool-call | `generate_tool_calls` | 函数签名 + 意图 → 期望调用 | 函数名 / 必填参数 / 嵌套；`check_tool_call` |

可控旋钮：
- **Dyck-k**：长度 `length`(L)、最大嵌套深度 `max_depth`(D)、括号类型数 `k`、
  配对距离上界 `max_pairing_distance`、模式 `unconditional|completion`。
- **JSON/XML**：`fmt`、`max_depth`、`max_props`。
- **tool-call**：内置函数库（`get_weather` / `send_email` / `set_timer`，含必填 /
  选填参数）；意图文本由采样的参数值生成。

### 诊断信号（供指标使用）

到最近合法串的编辑距离、最长合法前缀长度，以及采样循环插桩给出的：是否合法、
首个结构错误的全局字符位置、所在**生成块**、错误落在**块内 vs 块边界**（H3 归因）。

### 实验配置（`research/configs/experiments/`，一实验 = 一配置 = 一 commit）

| run_id | 任务 | 关键数据旋钮 | 块长 / 步数 | seeds | 样本数 | 用途 |
| --- | --- | --- | --- | --- | --- | --- |
| `smoke_dyck1` | dyck | L=4, D=2, k=1, 无条件 | blk=1 / steps=2 | [0] | 8 | 链路冒烟 |
| `dyck_L32_D6_k2_blk4` | dyck | L=32, D=6, k=2, 条件补全 | blk=4 / steps=16 | [1234,2025,7] | 256 | L0 诊断 |
| `json_strict_blk4` | structured | fmt=json, depth=2, props=4 | blk=4 / steps=16 | [1234,2025,7] | 256 | L1 诊断 |
| `tool_call_blk4` | tool_call | —（内置函数库） | blk=4 / steps=16 | [1234,2025,7] | 256 | L2 诊断 |

每个配置唯一描述：任务、L/D/k 等、块长、采样步数、checkpoint 占位符、seed 集合、
run_id；`checkpoint` 用 `${DIT_PATH}` / `${VAE_PATH}` / `${TOKENIZER_PATH}` 占位，
服务器侧注入。

### Sweep（`research/configs/sweep/`，诊断假设）

| 文件 | 扫描旋钮 | 取值 | 假设 |
| --- | --- | --- | --- |
| `block_sweep.yaml` | `block_size` | 1 / 2 / 4 / 8 | H3：块内 vs 块边界失配归因 |
| `step_sweep.yaml` | `timestep_num` | 4 / 8 / 16 / 32 / 64 | 步数对结构一致性的影响 |
| `vae_compression.yaml` | `patch_size` | 1 / 2 / 4 | H5：VAE / latent 压缩率影响（须换匹配权重） |

### 产物与复现

每个实验产物按 `run_id` 归档到 `research/results/<run_id>/`：`manifest.json`
（配置快照 + `git_sha` + 时间 + seed 集合）+ `data/samples/traces/metrics/plots/logs`。
小结果入库、大产物走文件同步——保证每个实验可由「配置 + commit」完整复现、可回滚。

---

## 附：一页速查（冒烟最短路径）

```bash
git checkout research/diagnostics-scaffold && EXPECTED_SHA=$(git rev-parse HEAD)
python -m venv .venv-gpu && . .venv-gpu/bin/activate && pip install -r requirements.lock
cp research/scripts/weights.sha256.example research/scripts/weights.sha256   # 填真实哈希
WEIGHTS_URL=<src> bash research/scripts/fetch_weights.sh
export DIT_PATH=$PWD/hf_models/cola_dlm/cola_dit VAE_PATH=$PWD/hf_models/cola_dlm/cola_vae TOKENIZER_PATH=$PWD/hf_models/tokenizer.json
bash research/scripts/check_sync.sh "$EXPECTED_SHA"
NUM_GPUS=1 EXPECTED_SHA="$EXPECTED_SHA" LAUNCHER=local \
  bash research/scripts/submit_job.sh research/configs/experiments/smoke_dyck1.yaml
# （run_experiment.sh 内填入真实 python -m cola_dlm.inference 命令 + prompt->question 适配）
```
