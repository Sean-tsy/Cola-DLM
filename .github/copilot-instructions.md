# Copilot 指令（转发至 AGENTS.md）

本仓库的协作约定以根目录 [`AGENTS.md`](../AGENTS.md) 为**单一真相源**，本文件仅指向并
摘录其要点；细则、环境分工、目录结构与行为准则一律以 `AGENTS.md` 为准。

要点摘录（完整内容见 `AGENTS.md`）：

- **项目**：围绕 Cola DLM 的「诊断 → 评测」研究；搭「数据生成 → 验证器 → 评测」流水线，
  量化精确离散结构一致性短板。**只做诊断与评测，不涉及后训练（SFT/DPO/RL）。**
- **环境分工**：本地仅纯 Python（`requirements-test.txt`，**无 torch**）跑 lint + 单测；
  **本地绝不加载/运行模型**；模型运行 / 采样 / 评测 / 取权重**一律只在服务器**
  （`requirements.lock`）。本地不要 `pip install -e .`。
- **来源唯一 / 可回滚**：研究新增全部以 `research/` 子目录叠加，不改上游核心算法逻辑。
- **完成门禁**：`ruff check . && black --check . && pytest -q` 必须通过。
- **远程纪律（硬约束）**：服务器作业必须 detached 后台跑、免密 ssh、绝不前台阻塞。
- **提交**：语义清晰、可回滚，commit 末尾附
  `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`。
