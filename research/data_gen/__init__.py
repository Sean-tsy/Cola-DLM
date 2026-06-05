"""数据生成器 (Cola DLM 诊断研究层)。

生成用于诊断 Cola 在精确离散结构一致性上短板的三级探针任务数据：

* 括号匹配 / Dyck 语言     —— :func:`generate_dyck`
* 严格格式 (JSON / XML)     —— :func:`generate_structured`
* 工具调用 (tool-call)      —— :func:`generate_tool_calls`

每个 sample 都带 ``to_record()``，渲染为对齐 :mod:`cola_dlm.inference` 输入契约的
JSONL 记录（``id`` / ``prompt`` / ``ground_truth`` + ``meta``）；落地写入仓库根
``generate_task_data/``（已 gitignore）。

所有生成器**纯 CPU、无模型、固定随机种子可复现**（相同 ``(seed, 参数, n)`` →
相同输出）。
"""

from __future__ import annotations

from .dyck import PAIRS, DyckSample, generate_dyck
from .structured import StructuredSample, generate_structured
from .tool_call import ToolCallSample, generate_tool_calls

__all__ = [
    "PAIRS",
    "DyckSample",
    "generate_dyck",
    "StructuredSample",
    "generate_structured",
    "ToolCallSample",
    "generate_tool_calls",
]
