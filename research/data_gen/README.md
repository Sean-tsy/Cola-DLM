# 数据生成器 (`research/data_gen/`)

三级探针的样本生成器。纯 Python、无模型、确定性（固定随机种子可复现），
本地可秒级单测。每个生成器接受 `seed`，相同 `(seed, 参数, n)` 产出**完全相同**的样本列表。

每个样本都提供 `to_record()`，渲染为推理用的 JSONL 记录
（`id` / `prompt` / `ground_truth` / `meta`），可直接喂给服务器侧采样与
[`research/eval`](../eval) 评测；`meta` 携带探针类型与可量化的标注信息。

生成器与[验证器](../validators)**解耦**：生成器自身不依赖验证器，
但单测会用验证器交叉校验「每个生成样本都合法」。

## 1. Dyck-k 生成器 —— `dyck.py`

- `generate_dyck(n, seed, *, length, max_depth=4, k=1, max_pairing_distance=None, mode="unconditional") -> list[DyckSample]`
- 可控旋钮：
  - `length`：完整合法词长度（偶数且 >= 2）
  - `max_depth`：最大嵌套深度（>= 1）
  - `k`：括号类型数（1..4，取 `()[]{}<>` 前 k 对）
  - `max_pairing_distance`：每对括号 `闭合下标 - 开启下标` 的上界（可选，>= 1）
- 两种模式：
  - `"unconditional"`：样本即完整合法 Dyck 词（`prefix` 为空）。
  - `"completion"`：在仍有未闭合左括号处切分，`prefix`（未闭合）+ `completion`（补全）
    始终拼成完整合法词，用于「条件补全」探针。
- `DyckSample` 字段含**实测** `max_depth` / `max_pairing_distance`（诚实上报：
  旋钮为约束，字段为实际测量值）。

## 2. JSON / XML 生成器 —— `structured.py`

- `generate_structured(n, seed, *, fmt="json", max_depth=2, max_props=4) -> list[StructuredSample]`
- 随机生成一份 JSON Schema（带类型、可嵌套、可必填的 object）及其**符合实例**，
  按 `fmt` 序列化为 JSON 或 XML。
- `fmt="json"`：实例经 `check_json(text, schema)` 校验为 `VALID`。
- `fmt="xml"`：良构元素树（根标签 `object`），必填字段对应元素均存在。
- `StructuredSample` 携带 `schema` / `value` / `text`。

## 3. tool-call 生成器 —— `tool_call.py`

- `generate_tool_calls(n, seed) -> list[ToolCallSample]`
- 以「函数签名 + 意图 → 期望 tool-call」成对产出。内置函数库
  （`get_weather` / `send_email` / `set_timer`），必填参数始终出现，可选参数随机。
- 每个样本携带单工具 `registry`（`函数名 -> arguments JSON Schema`），
  其 `text` 经 `check_tool_call(text, registry)` 校验为 `VALID`。

## 复现性

每个生成器内部仅用一个 `random.Random(seed)`。固定种子即固定输出；
不同种子产出不同样本。单测见 [`research/tests/`](../tests)
（`test_dyck.py` / `test_structured.py` / `test_gen_tool_call.py`），
覆盖复现性、与验证器交叉校验、旋钮约束与输入校验。
