# 实验记录

这个目录放三类东西，别混在一起：

| 文件 | 来源 | 能不能当实验结果 |
|---|---|---|
| `offline-demo.json` | `python homework.py demo`，假模型 | **不能**。只证明机制跑得通 |
| `learn-<时间戳>.json` | `python homework.py learn`，真实模型 | 能，属于训练阶段记录 |
| `evaluate-<时间戳>.json` | `python homework.py evaluate`，真实模型 | 能，是最终对照数据 |
| `summary-<时间戳>.md` | `evaluate` 结束时自动生成 | 能，直接贴进 README 的表格 |

`learn` 和 `evaluate` 每跑完一道题就落盘一次，中途中断也保留已完成部分，`complete` 字段会如实标记是否跑完。

`results/private/` 是经验库（`memory.json`）和逐次运行记录，已被 `.gitignore` 排除，不要提交。

## 一次完整实验的做法

1. 把 `ROUTER_DB`、`RSL_OUTPUT` 指到新的文件 / 目录（相当于清空经验和历史）。
2. 启动服务：`python homework.py serve`。
3. 另开终端：`python homework.py learn`。
4. 再跑：`python homework.py evaluate`。
5. 把 `summary-<时间戳>.md` 里的表格和 `evaluate` 的 `complete` 状态一起贴进 README。

## 记录里应该看什么

- `stop_reason`：`passed`（通过提前停）、`call_limit`（用满 3 次仍未通过）、`upstream_error`（接口故障，**不计入答错**）。
- `feedback_kind`：`labeled_training` 表示这题的反馈带标准答案数值，只能出现在 train 划分。
- `learn` 记录的 `reason`：`accepted` / `validation_rejected` / `no_successful_repair` / `memory_full` / `invalid_or_duplicate_rule`。
- `evaluate` 记录里的 `memory`：20 道题必须完全一致，否则说明测试期间经验被改动，这组对照无效（`homework.py` 会直接中止并标记 `memory_changed_during_test`）。
