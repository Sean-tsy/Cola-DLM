"""验证器 / verifier (Cola DLM 诊断研究层)。

对模型生成结果（推理输出 JSONL 的 ``generate`` 字段）做**精确结构一致性**
校验，作为后续 RLVR 的可验证奖励信号来源：

* Dyck / 括号匹配是否闭合、嵌套正确  —— :mod:`research.validators.stack`
* 严格格式 (JSON / Schema) 是否合法    —— :mod:`research.validators.json_schema`
* 工具调用是否为合法、可解析的调用     —— :mod:`research.validators.tool_call`

验证器为纯函数式、确定性、纯 CPU，可在本地单测。**本阶段只用于评测与诊断，
不作为后训练奖励。** 除合法/非法判定外，还产出可量化诊断信号（最长合法前缀长度、
到最近合法串的编辑距离等）供评测指标 (:mod:`research.eval`) 使用。
"""

from __future__ import annotations

from .json_schema import JsonResult, JsonStatus, check_json
from .stack import (
    DEFAULT_PAIRS,
    BracketErrorType,
    BracketResult,
    check_brackets,
    longest_valid_prefix_len,
    repair_distance,
)
from .tool_call import ToolCallResult, ToolCallStatus, check_tool_call

__all__ = [
    # bracket / Dyck
    "BracketErrorType",
    "BracketResult",
    "DEFAULT_PAIRS",
    "check_brackets",
    "longest_valid_prefix_len",
    "repair_distance",
    # json / schema
    "JsonResult",
    "JsonStatus",
    "check_json",
    # tool-call
    "ToolCallResult",
    "ToolCallStatus",
    "check_tool_call",
]
