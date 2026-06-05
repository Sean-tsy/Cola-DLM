# Cola DLM 源码改动地图 (change map)

为后续环节（核心改动 7.4）定位上游 `cola_dlm/` 中采样循环、block 划分、VAE、
DiT 先验、解码器的源码位置。行号以基线提交为准（克隆自上游 `main`，HEAD
`8d02058`），后续如上游变动需重新校准。

## 三步推理算法 (arXiv:2605.06548, Eq. 2.2.4–2.2.6)

入口: `cola_dlm/inference.py::generate_task_repaint_inference` (约 L285–745)

1. **Prefix encode** `z^pre ~ q_phi(z^pre | x^pre)` (Eq. 2.2.4)
   - `ColaTextVAEModel.encode` — `cola_dlm/modeling_cola_vae.py:580`
   - 调用点: `inference.py` Step 2 (约 L406–408)
2. **Block-wise latent prior transport** `hat z_0^(b) = Phi^psi_{0<-1}(...)` (Eq. 2.2.5)
   - 生成主循环: `inference.py` Step 5 `while not stop_flag` (约 L571+)
   - 每步 Euler 更新 + CFG 组合; 扩散步长 `_diffusion_dt` (L356)
   - DiT 向量场前向: `ColaDiTModel.forward` — `modeling_cola_dit.py:594`
3. **Conditional decode** `hat x^res ~ p_theta(x^res | ...)` (Eq. 2.2.6)
   - `ColaTextVAEModel.decode` — `modeling_cola_vae.py:646`
   - token 采样策略: `inference.py::sample_with_strategies` (L211)
     - 支持 greedy / temperature / top-k / top-p / repetition_penalty

## Block 划分

- `block_size`: `cola_dlm/configuration_cola_dit.py:69` (默认 4)
  - 定义 block-causal 注意力可见集 `V_b` 与推理循环每步推进的 chunk
- `patch_size`: VAE 端 1-D patch 因子 (`configuration_cola_vae.py`, 默认 1)
- `chunk = patch_size * block_size`: `inference.py:367` — 每样本 prompt
  按 chunk 对齐 pad（无 batch padding，NA / flatten-concat 形式）
- 首生成块布局 / prefix 切分: `inference.py` Step 3 (约 L411–473)

## 模块清单

| 角色 | 类 / 函数 | 文件:行 |
| --- | --- | --- |
| DiT 先验配置 | `ColaDiTConfig` | `configuration_cola_dit.py:18` |
| DiT 先验模型 | `ColaDiTModel` | `modeling_cola_dit.py:536` (forward L594) |
| DiT block | `ColaDiTBlock` | `modeling_cola_dit.py:471` |
| DiT 注意力 | `ColaDiTAttention` | `modeling_cola_dit.py:360` |
| VAE 配置 | `ColaTextVAEConfig` | `configuration_cola_vae.py` |
| VAE 模型 | `ColaTextVAEModel` | `modeling_cola_vae.py:470` |
| VAE encoder `q_phi` | `ColaTextVAEModel.encode` | `modeling_cola_vae.py:580` |
| VAE decoder `p_theta` | `ColaTextVAEModel.decode` | `modeling_cola_vae.py:646` |
| 后验高斯分布 | `DiagonalGaussianDistribution` | `modeling_cola_vae.py:76` |
| 注意力 NA mask | `create_na_block_causal_mask` | `attention_utils.py` |

## I/O 契约 (评测流水线对接点)

- 推理 CLI 入口: `inference.py::main` (L746)
  - 输入 JSONL 每行字段: `id`, `prompt`, `question`, `context`,
    `choices`, `ground_truth` (经 `apply_prompt_template` 组装，L80)
  - 输出 JSONL 每行字段: `prompt`, `generate`, `ground_truth`, ...
- prompt 模板: `inference.py::apply_prompt_template` (L80) — 按 task_name 分支
- 评测打分: `scripts/acc_calc.py` (process_single_file L321, main L591)
  - 扫描 `eval_output/tasks_*/` 下各 task jsonl，输出 `accuracy_summary.csv`

## 后续环节挂载点

- 新任务数据（Dyck/括号、严格格式、工具调用）: 由 `research/data_gen/` 生成
  → 落 `generate_task_data/*.jsonl`（已 gitignore）→ 喂 `inference.main`
- 新任务 prompt 模板: 需在 `apply_prompt_template` 增加分支（后续环节）
- 结构一致性验证: `research/validators/` 对 `generate` 字段做精确校验
- 新评测指标: `research/eval/` 在 `acc_calc` 之外补充结构一致性指标

---

# 环节四 · 核心改动地图（可配置开关 + 可插桩，默认关闭 patch 计划）

本环节**不直接改 `cola_dlm/` 核心代码**（本地不跑模型，核心改动无法验证）。
采用 **overlay 优先**：

- 研究侧已建完整、可单测的 harness —— `research/instrument/`
  （配置层 `config.py` + 无副作用 probe 契约 `probe.py` + 失配定位
  `locate.py` + JSONL trace `trace.py`），以及 `research/configs/sweep/` 的
  三份 sweep 配置（block / step / VAE 压缩）。
- 下面给出**精确插入点 + 默认关闭（no-op）的 patch 片段**，到服务器侧再落地。
  所有插桩以 `COLA_DIAG_TRACE` 环境开关 + `research.instrument.get_probe()` 守卫：
  **未开启时上游循环逐字节不变、零开销**。
- **不引入任何后训练相关的损失 / 奖励**：插桩只读不写，仅记录诊断信号。

## 改动点对照表（要理解 / 可配置 / 可插桩）

| 改动点 | 位置（文件:行） | 旋钮 / 插桩 | 服务于 | 落地方式 |
| --- | --- | --- | --- | --- |
| **block 划分 / 长度** | `dit.block_size`（`configuration_cola_dit.py:69`，默认 4）；用于 `inference.py:354,367,569`；VAE 块因果 mask 亦用 `vae.block_size`（`modeling_cola_vae.py:603,668`） | 配置开关 `SweepConfig.block_size` → `model_overrides` | H3 block sweep | 服务器侧须**同时**设 `dit.block_size` 与 `vae.block_size` 并断言相等；模型读 `self.block_size`，仅改 `config` 不够 |
| **采样循环（插桩 / 日志）** | 主循环 `inference.py:571+`；ODE 内层 `inference.py:608`；解码 `inference.py:668` | `probe.on_block_start` / `on_ode_step` / `on_block_decoded` | 步数 sweep；失配位置定位 | 守卫式 hook 调用（下方 patch） |
| **采样步数** | `timestep_num`（入参，默认 16）→ `timesteps`（`inference.py:478`） | 配置开关 `SweepConfig.timestep_num` → `to_inference_kwargs()` | 步数 sweep | 直接透传入参，无需改 core |
| **VAE encode/decode** | `ColaTextVAEModel.encode`（`modeling_cola_vae.py:580`）/ `decode`（:646）；`patch_size` 烘焙进 Conv/Linear（`modeling_cola_vae.py:514,540,687`） | 配置开关 `SweepConfig.patch_size` → `model_overrides` | H5 VAE-only、latent 压缩 sweep | **不可仅改 `vae.patch_size` 属性**（结构已烘焙）；须为每个 patch_size **加载/重建匹配的 VAE/DiT 权重对** |
| **DiT 先验（速度场前向）** | `ColaDiTModel.forward`（`modeling_cola_dit.py:594`）；调用点 `inference.py:622,633,693` | `on_ode_step` 记录 `drift_norm`/`txt_norm` | H4 似然-结构错配诊断 | 在 CFG 组合后取范数（`.float().norm().item()`）传 hook |
| **解码器（条件解码头）** | `ColaTextVAEModel.decode` → `sample_with_strategies`（`inference.py:668`） | `on_block_decoded` 喂每块解码文本 → `BlockMismatchLocator` | H3 块内 vs 块边界归因 | 解码后 `tokenizer.decode` 出每样本块文本传 hook |
| **验证 / 评测（新增）** | `research/validators/` + `research/instrument/locate.py` | 失配位置 / 块归因 / 诊断信号 | L0–L2 评测 | 已在 overlay 完成、单测覆盖 |

## 默认关闭的插入点（patch 计划，服务器侧落地）

> 约定：所有 hook 以 `_diag_probe = get_probe()` 取活动 probe，`None` 即跳过。
> 开关由 `COLA_DIAG_TRACE` 环境变量在入口处装配 `TracingProbe` 决定。
>
> **重要（服务器侧落地）**：
> - `model_overrides` 的 `block_size` / `patch_size` **不是**简单属性赋值：
>   `patch_size` 烘焙进 VAE 的 Conv/Linear 结构，须按值**加载匹配权重对**；
>   `block_size` 须**同时**改 `dit` 与 `vae` 并断言一致（二者都建块因果 mask）。
> - 插桩须 **exception-safe**：把 hook 调用与收尾放进 `try/finally`，
>   异常路径也要 `clear_probe()` + 关闭 writer，避免全局 probe 泄漏到下一次请求。
> - 首块 `one_block_ids` 含会被裁掉的 prompt-overlap token（`inference.py:717-721`）：
>   snippet D 在 `step == 0` 须按 `first_block_prompt_token_counts[b]` 先裁剪再解码，
>   使 locator 消费的文本流与最终 `generate` 一致（否则字符位置偏移）。

**A. 入口装配（`generate_task_repaint_inference` 顶部，`dit.eval()` 之前）**

```python
# --- diagnostics (default-off): only active when COLA_DIAG_TRACE is set ---
_diag_probe = None
if os.environ.get("COLA_DIAG_TRACE", "").strip():
    from research.instrument import JsonlTraceWriter, TracingProbe, set_probe
    _trace_path = os.environ.get("COLA_DIAG_TRACE_PATH", "diag_trace.jsonl")
    _diag_writer = JsonlTraceWriter(_trace_path)
    _diag_probe = TracingProbe(_diag_writer, probe_kind=task_name)
    set_probe(_diag_probe)
    _diag_probe.on_request_start(
        {"task_name": task_name, "block_size": dit.block_size,
         "patch_size": vae.patch_size, "timestep_num": timestep_num,
         "n_samples": len(prompts)}
    )
```

**B. 块开始（`while not stop_flag:` 体首，`inference.py:574` 附近）**

```python
if _diag_probe is not None:
    _diag_probe.on_block_start(int(step), txt_shape_cum.detach().cpu().tolist())
```

**C. ODE 步（`for t_curr, t_next ...` 体内，`txt = txt_next` 之后，`inference.py:654` 附近）**

```python
if _diag_probe is not None:
    _diag_probe.on_ode_step(
        int(step), int(_ode_index), float(t_curr), float(t_next),
        drift_norm=float(drift.float().norm().item()),
        txt_norm=float(txt.float().norm().item()),
    )
```
（`_ode_index` 由 `enumerate(zip(timesteps[:-1], timesteps[1:]))` 提供。）

**D. 块解码后（`sample_with_strategies(...)` 之后，`inference.py:675` 附近）**

```python
if _diag_probe is not None:
    _ids = one_block_ids.detach().cpu().tolist()
    if step == 0:  # 裁掉首块的 prompt-overlap，使文本流对齐最终 generate
        _trim = first_block_prompt_token_counts.detach().cpu().tolist()
        _ids = [row[int(t):] for row, t in zip(_ids, _trim)]
    _block_texts = tokenizer.decode_batch(_ids, skip_special_tokens=False)
    _diag_probe.on_block_decoded(int(step), list(_block_texts))
```

**E. 收尾（exception-safe，整个生成体包在 try / finally 内）**

```python
try:
    ...  # Step 5 生成循环 + Step 6 裁剪 + 组装 results
    if _diag_probe is not None:
        _diag_probe.on_request_finish(results)
    return results
finally:
    if _diag_probe is not None:
        _diag_writer.close()
        from research.instrument import clear_probe
        clear_probe()  # 全局注册表，异常路径也必须清理，避免泄漏到下次请求
```

落地校验（服务器侧）：未设 `COLA_DIAG_TRACE` 时输出须与基线逐行一致（diff 为空）；
设开关后产出 `diag_trace.jsonl`，含 `block_start` / `ode_step` / `block_decoded` /
`mismatch` / `request_finish` 记录，由 `research/eval/`（环节后续）聚合为指标。

## 留给环节五（指标 / 评测）的注意点

- **trace 身份**：聚合多 sweep 点前，建议在 `on_request_start` 的 `meta` 里补
  `run_id` / `config_name` / `seed` / 权重 id / git sha；多 rank / 多进程时
  `COLA_DIAG_TRACE_PATH` 应带 rank 后缀，避免 JSONL 交错。
- **tokenizer 可组合性**：块边界归因依赖 `decode(b1)+decode(b2) == decode(b1+b2)`；
  部分 tokenizer（byte-level / 空格清理）不严格成立。建议同时落 `block_token_ids`
  以便下游核对拼接一致性，并把「边界」表述为**解码块边界**而非 token 边界。
- **失配分类**：`tool_call` / JSON schema 违例无字符位置（`block_index` 为空），
  下游应区分「有位置的语法失败 / schema-路径失败 / 无位置语义失败」三类，
  不要把「无 `block_index`」当缺失数据。
- **逐块 vs 收尾**：`TracingProbe` 仅在收尾输出最终 `mismatch`，但每块 `block_text`
  已落盘，下游可据此重算「首个失配块」；注意 JSON 部分前缀的 EOF 解析错误属预期，
  应与已提交的语法错误区分。
- **性能**：`on_ode_step` 的 `drift.norm().item()` 会触发逐步 device sync；
  开启 trace 时不要用其测得的端到端时延作为性能指标。
