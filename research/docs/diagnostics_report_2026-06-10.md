# Cola DLM 精确离散结构一致性诊断报告

日期：2026-06-10 · 分支：`research/diagnostics-scaffold` · 权重：官方发布
cola_dit + cola_vae（指纹见 `research/results/weights.sha256`）

## 1. 目标与问题定义

Cola DLM（连续隐空间扩散语言模型）在通用基准上有合理表现，但其在**精确离散
结构一致性**任务上的能力未知。本诊断回答三个问题：

1. Cola 能否产出结构精确合法的输出（括号匹配 → 严格 JSON → 工具调用）？
2. 若不能，失败发生在管线哪一环：VAE 编解码、扩散先验、块状采样、还是条件链路？
3. 采样超参（block 长度、ODE 步数）能否缓解？

方法：三级合成/外部探针 + 精确验证器 + 采样循环插桩归因，全部实验由
「配置 + commit + seed + checkpoint hash」唯一确定（索引见
`research/results/reproducibility_index.json`，19 个 run）。

## 2. 实验设置

- **L0 探针**：合成 Dyck-1/k 括号串（可控长度 L、深度 D、模式无条件/条件补全），
  验证器 `check_brackets` 给出合法性、首错位置、错误类型、repair distance。
- **L1/L2 外部基准**：JSONSchemaBench（Github_easy/test）、BFCL v3（simple）；
  验证器分级 parse → schema → 函数名/必填参数。格式与语义严格分开，本报告只测格式。
- **归因工具**：采样循环 probe（块开始/ODE 步/块解码事件 + 失配定位，默认关闭、
  对上游零扰动——重跑验证非 trace 指标逐位一致）；VAE-only GT roundtrip 对照。
- **设置**：guidance 7.0、greedy（temperature 0）、3 seeds（1234/2025/3407）、
  确定性逐样本噪声种子；8×A100-40GB 数据并行。

## 3. 校准（9.2）：复现基线成立

官方 8 任务平均 26.83 vs 论文 26.75（delta 0.08 < 容差 3.0，PASS）。
后续所有负面结果不是「跑错了模型」。

## 4. 主结果

### 4.1 结构合法率全景

| 任务 | 设置 | valid rate | 95% CI |
| --- | --- | ---: | --- |
| Dyck-1 无条件生成 | L=32 | **0.953** | [0.914, 0.984] |
| Dyck-1 条件补全 | L=32 | **0.031** | [0.008, 0.063] |
| Dyck-1 条件补全 | L=64, D≤8（主网格中心） | 0.009 | [0.003, 0.017] |
| DyckLanguage（HF, n=768） | 条件补全 | 0.0 | — |
| JSON（合成 smoke） | schema 提示 | 0.0 | — |
| JSONSchemaBench | Github_easy/test, n=1536 | **0.0** | parse_error 100% |
| BFCL v3 simple | n=1200 | **0.0** | parse_error 100% |
| **VAE-only GT roundtrip** | L=32 | **1.0**（ppl≈1.0002） | — |
| **VAE-only GT roundtrip** | L=64 主网格 | **0.723**（repair 0.28） | [0.690, 0.755] |

核心对比：**无条件 0.95 vs 条件补全 0.03**（同分布、同长度）。条件化本身
就是断崖。外部真实任务上输出甚至不是可解析 JSON，呈「损坏的 prompt 续写」
形态（复读 schema/工具描述文本、漂移到预训练语料风格片段）。

### 4.2 跨模型对照：同量级 AR 基线（Qwen2.5-1.5B base）

Cola 系统约 1.5–2B（DiT 2048×24 + VAE）。对照模型 Qwen2.5-1.5B **base**
（非指令版，与 Cola 同为无 SFT 的 base 模型），**同一份探针数据、prompt
逐字相同（不加 chat template）、同 greedy 解码与 token 预算、同一套验证器**：

| 探针 | 口径 | Cola | Qwen2.5-1.5B base |
| --- | --- | ---: | ---: |
| Dyck L64 补全 | 严格合法率 | 0.009 [0.003, 0.017] | **0.057** [0.040, 0.073] |
| JSONSchemaBench | 严格合法率 | 0.0 | 0.014 |
| JSONSchemaBench | 宽松：可解析¹ | 0.190 | **0.752** |
| JSONSchemaBench | 宽松：过 schema¹ | 0.031 | **0.229** |
| BFCL simple | 严格合法率 | 0.0 | 0.0 |
| BFCL simple | 宽松：完整合法调用¹ | **0.0** | **0.583** |

¹ 宽松口径：先从输出提取首个 JSON 片段（markdown 围栏或首个配平 `{...}`），
再走同一验证器；补充性分析（seed 1234），提取规则如上可复现。

定性差异比数字更大：Qwen 的失败是**包装问题**——JSON 写在围栏/客套话里，
BFCL 给出了正确 envelope 但混在 few-shot 风格续写中（base 模型的典型行为）；
Cola 的失败是**根本不产生尝试**——损坏的 prompt 复读。两个 base 模型严格
口径都低是「未指令调优」的共性，但宽松口径下 58.3% vs 0.0%（BFCL）、
22.9% vs 3.1%（JSON schema）、以及不需要任何指令理解的 Dyck 上 6×差距
（CI 不重叠），说明**结构条件化缺陷是 Cola 特有的，不能归咎于缺少 SFT**。

### 4.3 采样超参 sweep（9.4，Dyck L64 D≤8 补全，n=768/点）

| 旋钮 | 取值 | valid rate |
| --- | --- | --- |
| block 长度（steps=16） | 1 / 4 / 8 / 16 / 32 | 0.012 / 0.005 / 0.010 / 0.009 / 0.008（CI 全重叠） |
| ODE 步数（block=16） | 8 / 16 / 32 / 64 | 0.005 / 0.009 / 0.007 / 0.008（CI 全重叠） |

两个旋钮都救不了：失败与采样精度/块粒度基本无关。

### 4.4 失配位置归因（trace 插桩）

错误落在解码块边界（块首/末字符）的比例 vs 均匀分布基线 `2/块字符长`：

| block | 观测 | 均匀基线 |
| ---: | ---: | ---: |
| 4 | 0.272 | 0.250 |
| 8 | 0.113 | 0.125 |
| 16 | 0.065 | 0.062 |
| 32 | 0.031 | 0.031 |

逐档吻合 → **H3（块缝合假设）不成立**：错误不在块边界富集。错误类型以
`unclosed`（开括号不闭合）为主（59–90%），即全局栈深度/计数约束缺失，
而非局部语法噪声。（注：unclosed 的 error_position 指向未闭合开括号，
block_index 直方图的 b0 前倾部分源于该报告约定，不能直接读作「首块即错」。）

## 5. 归因结论（逐环节排除）

| 环节 | 证据 | 判定 |
| --- | --- | --- |
| VAE 编解码 | GT roundtrip：L32 完美；L64 合法率 0.723、repair 仅 0.28 | 有随长度增长的**小噪声底**，量级不足以解释失败（repair 0.28 vs 模型 20–37） |
| 块状采样拼接（H3） | boundary rate ≈ 均匀基线；block 1–32 无差异 | **排除** |
| ODE 离散化 | 8→64 步无改善 | **排除** |
| 扩散先验（无条件） | 无条件 Dyck 0.95 | 基本胜任短串结构 |
| **条件补全/指令链路** | 条件化即断崖（0.95→0.03）；外部任务退化为 prompt 复读；unclosed 主导（前缀栈状态未被尊重） | **主因** |
| 「缺 SFT」解释 | 同量级未调优 AR base（Qwen2.5-1.5B）同条件下：Dyck 6×、BFCL 宽松 0.583 vs 0.0 | **排除**（缺陷为 Cola 特有，非 base 模型共性） |

**一句话结论**：Cola 的失败不是「写不出结构」，而是「prompt/前缀的结构约束
没有进入扩散先验的条件生成」。clean-guidance 前缀机制保住了 KV 上下文的表面
连续性，但没有传递「还有 N 个括号没闭合」「接下来必须输出 JSON」这类约束。

## 6. 对后续改进（RLVR 后训练）的输入

本阶段不做训练，但诊断直接给出改进抓手：

1. **奖励信号现成**：三级验证器（brackets/json/tool_call）就是 RLVR verifier，
   分级错误（parse/schema/参数）支持塑形。
2. **训练分布应以条件补全为主**：无条件已 0.95，提升空间在条件链路。
3. **难度阶梯可控**：L/D/k/配对距离/格式深度全部参数化，可做课程式数据。
4. **评测护栏已就位**：校准脚本防回退通用能力；可复现索引防实验漂移。

## 7. 工程产出与已知限制

- 采样循环插桩落地（change_map Phase-4，commit `2aa7277`）：默认关闭零开销，
  开启时 trace 含块/ODE/失配事件，多卡按 sample_id 对齐。
- 外部基准接入：JSONSchemaBench（`datasets`）、BFCL v3（裸 JSONL 走
  `hf_hub_download` + Python 风格类型清洗，commit `044ae63`）。
- 已知坑修复：JSONSchemaBench default 档巨型 schema OOM（`c24c38b`）、
  VAE-only completion 评测双前缀伪影（`ed5605c`）、sync --delete 误删未拉取
  结果（`400c2fe`）。
- **Blocked**：latent 压缩率（patch_size）sweep——patch_size 烘焙进权重结构，
  仅有单权重对，不可执行。
- **未跑**：StructEval（L1 可选）；BFCL 语义比对（possible_answer 已存入
  ground_truth 备用）。
- **公开榜单不直接引用**：BFCL leaderboard / JSONSchemaBench 论文数字基于
  chat template、约束解码或 AST 比对等不同协议，与本报告的格式合法率口径
  不可比，故跨模型对照只用同管线实测（§4.2）。
- 9.4/9.5 详表分别见 `research/results/dyck_9_4_summary.md`、
  `research/results/l1_l2_9_5_summary.md`。

## 8. 可复现性（7.10）

`research/results/reproducibility_index.json`（由
`research/scripts/build_repro_index.py` 重建）覆盖全部 19 个归档 run：
git sha、created_at、seeds、配置 sha256、headline 指标、错误分布，以及
checkpoint 指纹（DiT 两分片 + VAE + tokenizer 的 sha256）。任一 run 可由
`git checkout <sha>` + 对应配置 + seeds + 同一权重对完整复现。
