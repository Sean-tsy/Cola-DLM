"""数据生成器 (Cola DLM 诊断研究层)。

生成用于诊断 Cola 在精确离散结构一致性上短板的任务数据：

* 括号匹配 / Dyck 语言
* 严格格式 (strict-format)
* 工具调用 (tool-call)

生成结果落地为 JSONL（写入仓库根 ``generate_task_data/``，已 gitignore），
字段对齐 :mod:`cola_dlm.inference` 的输入契约
（``id`` / ``prompt`` / ``question`` / ``ground_truth`` 等）。

注意：本模块只负责**生成数据文件**（纯 CPU、无模型），不在本地跑模型。
具体生成器在后续环节实现。
"""
