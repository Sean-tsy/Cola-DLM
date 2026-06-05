# 实验配置 (`research/configs/`)

集中存放实验配置，保证每个实验**可配置、可复现、可回滚**。

约定（后续环节填充）：

- 每个诊断实验一个 YAML，记录任务、数据生成参数、推理超参（`timestep_num`、
  `guidance_scale`、`block_size` 等）、随机种子、权重路径占位符。
- 权重 / 数据路径用变量占位（如 `${DIT_PATH}`），实际值在服务器侧注入，
  不把绝对路径或大文件写入版本控制。
- 配置纳入 git 追踪，改动可 diff、可回滚。

> 本环节仅建立骨架；具体配置文件在后续「数据生成 / 评测」环节加入。

## experiments/ —— 环节五实验配置（一实验 = 一配置 = 一 commit）

`experiments/` 下每个 YAML **唯一描述**一个实验（`run_id` / 任务 / L·D·k 等数据旋钮 /
块长 / 采样步数 / checkpoint 占位符 / 随机种子集合），由 `research.experiment`
（`load_experiment`）消费，产物按 `run_id` 归档到 `research/results/`。示例：

- `dyck_L32_D6_k2_blk4.yaml` —— Dyck-k 括号匹配（L0）。
- `json_strict_blk4.yaml` —— 严格 JSON 格式（L1）。
- `tool_call_blk4.yaml` —— 工具调用（L2）。

`run_id` 须等于文件名 stem；`checkpoint` 用 `${VAR}` 占位、服务器侧注入。
详见 [`../experiment/README.md`](../experiment/README.md)。

## sweep/ —— 环节四诊断 sweep 配置

`sweep/` 下三份网格配置，配合 `research/instrument`（`load_sweep_grid`）使用，
**仅服务器侧落地**（本地只校验加载/解析）：

- `block_sweep.yaml` —— 扫 `block_size`（`dit.block_size`，H3 块内 vs 块边界）。
- `step_sweep.yaml` —— 扫 `timestep_num`（ODE / 扩散积分步数）。
- `vae_compression.yaml` —— 扫 `patch_size`（VAE / latent 压缩率，H5）。

每份用顶层 `defaults:` 共享旋钮 + `sweep:` 列表逐点覆盖；`block_size` / `patch_size`
是模型配置属性（`model_overrides`，服务器侧应用），`timestep_num` 等为推理入参
（直接透传）。详见 [`../instrument/README.md`](../instrument/README.md)。
