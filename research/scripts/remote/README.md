# research/scripts/remote — 本地触发、服务器执行（B 模式）

把「本地准备、服务器执行」落地为**可调用脚本**：本地 agent 只发**立刻返回**的 ssh
命令触发作业，真正的 GPU 作业在服务器**后台 detached** 运行。本地只「**提交 → 轮询
→ 拉回**」，**绝不前台阻塞**。

## 前置：免密 ssh

依赖 `~/.ssh/config` 里的别名 + key，脚本内**不出现任何密码 / token**：

```sshconfig
# ~/.ssh/config
Host cola-gpu
    HostName <server-ip>
    User siyuan
    IdentityFile ~/.ssh/id_ed25519
```

之后所有脚本用 `REMOTE=cola-gpu` 引用该别名。

## 四件套

| 脚本 | 作用 | 用法 |
| --- | --- | --- |
| `sync.sh` | git push + 服务器 pull + **核对 commit 一致** | `REMOTE=cola-gpu bash research/scripts/remote/sync.sh [branch]` |
| `submit.sh` | **detached** 启动作业（命令秒回），日志 → `logs/<run_id>.log` | `REMOTE=cola-gpu bash research/scripts/remote/submit.sh <config> <run_id>` |
| `status.sh` | tail 日志 + 判断 RUNNING/DONE/FAILED | `REMOTE=cola-gpu bash research/scripts/remote/status.sh <run_id> [tail]` |
| `fetch.sh` | rsync 拉回 `results/<run_id>/` | `REMOTE=cola-gpu bash research/scripts/remote/fetch.sh <run_id>` |

`status.sh` 退出码：`0`=DONE(exit 0)、`1`=FAILED、`2`=RUNNING、`3`=NOT_FOUND。

## 配置（环境变量，均无密码）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `REMOTE` | （必填） | `~/.ssh/config` 的 ssh 别名（key-based） |
| `REMOTE_REPO` | `/data0/users/siyuan/projects/Cola-DLM` | 服务器仓库路径 |
| `RESULTS_ROOT` | `research/results` | 结果目录（本地 + 服务器） |
| `LOG_DIR` | `logs` | 服务器侧 detached 作业日志目录（相对仓库） |
| `LAUNCHER` | `nohup` | `nohup` \| `tmux` \| `sbatch` |
| `JOB_CMD` | `bash research/scripts/submit_job.sh <config>` | 作业命令；可换纯 Python 假作业做联调 |
| `SSH` / `RSYNC` | `ssh` / `rsync -az` | 可注入（仅供测试） |

## 典型闭环

```bash
export REMOTE=cola-gpu
bash research/scripts/remote/sync.sh                                   # 代码一致
bash research/scripts/remote/submit.sh \
  research/configs/experiments/smoke_dyck1.yaml smoke_dyck1            # 秒回
bash research/scripts/remote/status.sh smoke_dyck1                     # 轮询直到 DONE
bash research/scripts/remote/fetch.sh smoke_dyck1                      # 拉回产物
```

## 端到端自检（纯 Python 假作业，不接模型）

`selftest.sh` 用注入的假 ssh/rsync + 纯 Python 假作业（`_fake_job.py`，不加载模型）
验证四脚本链路通畅，无需真实服务器：

```bash
bash research/scripts/remote/selftest.sh
# => [selftest] OK: sync/submit/status/fetch chain verified with fake job
```

确认链路无误后，把 `JOB_CMD` 换成真实作业（默认即 `submit_job.sh`）接服务器即可。
