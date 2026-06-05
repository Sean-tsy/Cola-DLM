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
