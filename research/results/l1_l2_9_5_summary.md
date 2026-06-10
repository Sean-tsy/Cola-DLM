# 9.5 L1/L2 格式评测汇总（2026-06-10）

外部基准上的格式合法率 baseline。评测只测**结构/格式**（parse → schema →
函数名/参数分级），不与基准自带的语义答案比对；语义指标留待后续，
两者严格分开。BFCL 的 possible_answer 已随数据 join 进 `ground_truth`
字段，供将来语义评分使用。

## 结果

| run | 基准 | n | format valid | 错误分布 |
| --- | --- | ---: | ---: | --- |
| `jsonschemabench_cola` | JSONSchemaBench `Github_easy`/test | 1536 | **0.0** | parse_error 100% |
| `bfcl_cola` | BFCL v3 `simple` | 1200 | **0.0** | parse_error 100% |

采样设置：block 16、steps 16、guidance 7.0、greedy、max_new_tokens 256、
3 seeds。验证器分级（parse_error / schema_error / unknown_function /
missing_argument / invalid_argument）本可区分「会写 JSON 但不守 schema」
等中间状态——实际全部止步于第一级：**输出根本不是可解析的 JSON**。

## 失败形态

两个基准的生成样本呈同一形态：模型输出**损坏的 prompt 续写**——复读
schema/工具描述文本（夹杂错乱 token，如 `"Produce a the document
conforming to the the Schema:"`），或漂移到预训练语料风格的无关文本
（GitHub issue 片段），没有任何按指令产出 JSON envelope 的迹象。

这与 9.2–9.4 的归因一致并将其外推到真实任务：Cola 的条件生成链路
（prompt → 续写）没有把指令/结构约束注入扩散先验，prompt 越长越复杂，
退化越彻底（Dyck 补全 valid ~1% → JSON/tool-call 0%）。

## 工程记录

- 数据接入：JSONSchemaBench 经 `datasets` 正常加载（列 `json_schema`/
  `unique_id`）；BFCL 仓库是裸 JSONL，`datasets` 报 DataFilesNotFoundError，
  改走 `hf_hub_download`（`load_bfcl_records`），并把 BFCL 的 Python 风格
  类型（dict/tuple/float/any）清洗成 JSON Schema 原语，否则 Draft 2020-12
  校验直接抛 UnknownType（commit `044ae63`）。
- OOM：JSONSchemaBench `default` 混档含 Kubernetes/JsonSchemaStore 巨型
  schema，prompt 全文使 VAE prefix attention 申请 12–20 GiB 而 OOM；
  改用 `Github_easy` 档 + `INFER_BATCH_SIZE=4`（commit `c24c38b`）。
- 复现注记：`bfcl_cola` 采样发生在 commit `044ae63`；其 manifest/metrics
  因 sync `--delete` 事故在服务器上丢失，由原 samples 在 `c24c38b` 重建
  （samples 未动，eval 纯 CPU 后处理）。sync.sh 已加 exclude 防再发
  （commit `400c2fe`）。
- StructEval（L1 可选）未跑：必选基线已给出 0.0 的明确信号，增量信息有限，
  留作后续可选项。
