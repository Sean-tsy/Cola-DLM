# 插桩 / 配置 harness (`research/instrument/`)

环节四的 **overlay 侧**交付：把核心采样路径上的「可配置开关 + 可插桩点」做成
纯 Python、可单测、**默认零开销**的 harness。**本目录不导入也不运行模型**；
对 `cola_dlm/` 核心的实际插入点（默认关闭的 patch 计划）见
[`../docs/change_map.md`](../docs/change_map.md) 的「环节四」一节。

> 不引入任何后训练（SFT/DPO/RL）相关的损失 / 奖励：插桩只读不写，仅记录诊断信号。

## 1. 配置层 —— `config.py`

`SweepConfig` 把一次 sweep 点的旋钮按**作用位置**分三类：

- `sampling`：直接透传给 `generate_task_repaint_inference` 的入参
  （`timestep_num` 步数、`guidance_scale`、解码温度等）。`to_inference_kwargs()`。
- `model_overrides`：作用于**已加载模型配置**的属性，而非函数入参——
  `block_size`（`dit.block_size`，H3）与 `patch_size`（`vae.patch_size` / latent
  压缩，H5）。`to_model_overrides()`，服务器侧在调用前应用到 `dit` / `vae`。
- `env`：上游已支持的环境开关（`COLA_INFER_PER_SAMPLE_NOISE_SEED`，由 `seed` 派生）
  + 诊断开关 `COLA_DIAG_TRACE`（由 `trace` 派生）。`to_env()`。

加载：`load_sweep(path)`（单点）/ `load_sweep_grid(path)`（带 `defaults:` + `sweep:`
列表的网格）。未知键会**显式报错**，避免配置笔误被静默忽略。

## 2. probe 契约 —— `probe.py`

`SamplingProbe` 定义采样循环的插桩契约（全部 no-op 默认实现）：

- `on_request_start(meta)` / `on_block_start(step, txt_shape_cum)`
- `on_ode_step(step, ode_index, t_curr, t_next, drift_norm, txt_norm)`
- `on_block_decoded(step, block_text)` —— 驱动失配定位的关键 hook
- `on_request_finish(results)`

全局注册表 `set_probe / get_probe / clear_probe`：核心循环通过 `get_probe()` 取活动
probe，**未注册时返回 `None`、循环逐字节同上游、零开销**。所有 hook 入参均为纯
Python 值（核心在调用前把张量转成标量 / 列表 / 字符串）。

`TracingProbe` 是落地实现：写 JSONL trace，并为每个样本维护一个
`BlockMismatchLocator`，收尾时输出每样本失配定位。

## 3. 失配定位 —— `locate.py`

`BlockMismatchLocator` 逐块累积解码文本，用[环节二验证器](../validators)给出：

- 是否结构合法、首个结构错误的**全局字符位置**；
- 该错误落在**第几个生成块**；
- 错误是否位于**块边界**（块首/块尾字符）还是**块内部**。

块边界 vs 块内归因即 **H3**（块内 vs 块边界）的信号：多在块缝处失败 → 指向
block-causal 隐先验传输；多在块内失败 → 指向块内条件解码。

## 4. trace 落盘 —— `trace.py`

`JsonlTraceWriter`：append-only JSONL sink（每条记录一行），可作上下文管理器。

## 单测

见 [`../tests/`](../tests)：`test_instrument_config.py` / `test_instrument_locate.py`
/ `test_instrument_probe.py`，覆盖配置三分桶与校验、网格加载、块归因（边界/内部/
未闭合）、probe 注册表 no-op 与完整 trace 流程。本地秒级跑通，不加载模型。
