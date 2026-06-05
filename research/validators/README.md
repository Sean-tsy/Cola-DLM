# 验证器 (`research/validators/`)

三类**精确离散结构一致性**判定器。纯 Python、确定性、纯 CPU，可本地秒级单测。
**本阶段只用于评测与诊断，不作为后训练奖励。**

每个判定器都是纯函数，返回一个 frozen dataclass 结果；并额外产出可量化诊断信号，
供 [`research/eval`](../eval) 的指标使用。

## 1. 栈 / 括号 (Dyck) 判定器 —— `stack.py`

- `check_brackets(s, pairs=DEFAULT_PAIRS) -> BracketResult`
  - `valid`：是否合法
  - `error_position`：首次出错位置（不合法时）
  - `error_type`：不匹配类型 `BracketErrorType`
    - `UNEXPECTED_CLOSING`（多余右括号）/ `MISMATCHED`（左右不配）/ `UNCLOSED`（有未闭合左括号）
- 诊断信号：
  - `longest_valid_prefix_len(s)`：最长合法前缀长度
  - `repair_distance(s)`：到最近合法串的编辑距离（单括号类型精确；多类型为紧上界）

## 2. JSON / Schema 校验器 —— `json_schema.py`

- `check_json(text, schema=None) -> JsonResult`，三态 `JsonStatus`：
  - `PARSE_ERROR`：根本不是合法 JSON（带 `error_position` 字符偏移）
  - `SCHEMA_ERROR`：JSON 合法但违反 schema（带 `error_path` JSON 路径）
  - `VALID`：合法（且在给定 schema 时符合）

## 3. tool-call 校验器 —— `tool_call.py`

在前两者之上，再判函数名、必填参数、嵌套结构。

- `check_tool_call(text, tools) -> ToolCallResult`
  - `tools`：注册表 `函数名 -> arguments 的 JSON Schema`
  - 状态 `ToolCallStatus`：
    `PARSE_ERROR` / `SCHEMA_ERROR`（信封 `{"name","arguments"}` 不合规）/
    `UNKNOWN_FUNCTION` / `MISSING_ARGUMENT`（含嵌套必填）/
    `INVALID_ARGUMENT`（类型/嵌套结构错误）/ `VALID`

单测见 [`research/tests/`](../tests)（`test_stack.py` / `test_json_schema.py` /
`test_tool_call.py`），含随机串与边界用例。
