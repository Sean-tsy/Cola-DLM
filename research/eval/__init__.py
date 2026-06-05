"""评测 harness (Cola DLM 诊断研究层)。

在上游 ``scripts/acc_calc.py`` 之外，补充面向**结构一致性**的评测：聚合
验证器（:mod:`research.validators`）的逐样本判定，产出指标与归因分析，
并把结果归档到 ``research/results/``。

评测对**已生成的输出 JSONL** 做后处理（纯 CPU），不触发模型推理；模型采样
本身只在服务器侧执行。具体实现在后续环节加入。
"""
