"""RSL（Recursive Self-Learning，递归自我学习）核心。

一次完整的 RSL 循环：

    作答 → 程序批改 → 有限轮修订 → 提炼经验规则 → 验证集把关 → 入库
                                                            ↑          │
                                                            └── 下一题带上经验 ┘

设计边界（作业要求，也是本文件的硬约束）：

1. 不训练、不微调，模型权重不变。所谓"学习"，产物只是后续提示词里的几条经验规则。
2. 不学习模型路由。题目由调用方指定的模型作答，走不走级联、用哪个 Provider
   仍然由 ModelRouter 决定，RSL 不参与选模型。
3. 标准答案由 Python 直接算出，不请模型给自己打分，避免"自己批自己"。
4. 训练 / 验证 / 测试三个划分固定且互不重叠：
   - 训练题：允许带标签反馈，允许产生候选规则；
   - 验证题：每题只作答一次，只用来判断规则能不能入库；
   - 测试题：不提供任何标准答案数值，不更新经验，只用于最终对照。

模块内部分层：
    dataset()      固定数据集与标准答案
    evaluate()     严格批改器（不执行模型写的计算过程）
    refine()       带修订预算的作答循环，保留得分最高的答案
    learn()        从"训练题的一次成功修订"中提炼一条规则
    validate_rule() 用验证集做前后对照，退步或没长进就拒绝
    compare()      三组对照（直接回答 / 反馈修正 / 带经验单次回答）
    Store          经验与运行记录的落盘，按调用身份 + 模型隔离
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# 数据集版本号。数据生成逻辑一改就必须改这里，否则旧经验会被误用到新题上。
DATASET_VERSION = "orders-v1-20261022"

# 每题最多调用模型几次（首答 + 2 次修订）。超过 3 次收益很低，成本却线性上涨。
MAX_ROUNDS = 3
# 经验库上限。规则越多，提示词越长，互相冲突的概率也越高。
MAX_RULES = 5
# 单条规则的长度上限，防止模型把整段题面抄进经验。
MAX_RULE_CHARS = 240
# 单次模型调用超时（秒）。
CALL_TIMEOUT = 120
# 每题的行数。
ROWS_PER_TASK = 12

# 三个划分：(名称, 随机种子, 题量)。种子写死，保证每次运行数据集完全一致。
SPLITS = (
    ("train", 2026102201, 12),
    ("validation", 2026102211, 8),
    ("test", 2026102221, 20),
)

ANSWER_KEYS = ("ids", "count", "total")

TASK_RULES = (
    '只保留 active=true、region="{region}"、qty>={threshold} 的订单。'
    "按 qty 降序排列，qty 相同时按 id 升序。"
    "最终答案是 JSON 对象，只允许 ids、count、total 三个字段："
    "ids 为排好序的 id 数组，count 为订单数，total 为入选订单 qty×price 之和。"
    "count 与 total 必须是整数，布尔值不算整数。"
    "先逐行核对三个筛选条件，列出入选订单及每项 qty×price 的计算，再求和；"
    "计算过程写在前面，最后用一个 ```json 代码块给出最终对象，代码块之后不要再写任何文字。"
)

MEMORY_HEADER = (
    "下面是此前在同类任务上验证有效的经验，可作为检查清单使用。"
    "题目本身的要求优先级更高，冲突时以题目为准："
)


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------


def _make_rows(rng: random.Random) -> list[dict]:
    """生成一道题的订单表。字段固定，便于程序精确比对。"""
    return [
        {
            "id": f"R{k:02d}",
            "active": rng.choice([True, True, False]),
            "region": rng.choice(["east", "west"]),
            "qty": rng.randint(1, 12),
            "price": rng.randint(11, 99),
        }
        for k in range(ROWS_PER_TASK)
    ]


def _select(task: dict) -> list[dict]:
    """按题目条件筛选并排序。筛选逻辑只写一份，出题和判分共用。"""
    rows = [
        row
        for row in task["rows"]
        if row["active"] and row["region"] == task["region"] and row["qty"] >= task["threshold"]
    ]
    rows.sort(key=lambda row: (-row["qty"], row["id"]))
    return rows


def _build_task(split: str, index: int, rng: random.Random) -> dict:
    """出一道题。要求至少选中 2 行，避免出现"空答案"这种退化题目。"""
    while True:
        rows = _make_rows(rng)
        region = rng.choice(["east", "west"])
        threshold = rng.randint(3, 7)
        candidate = {"rows": rows, "region": region, "threshold": threshold}
        if len(_select(candidate)) >= 2:
            break
    prompt = (
        TASK_RULES.format(region=region, threshold=threshold)
        + "\n订单：\n"
        + "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    )
    return {
        "id": f"{split}-{index + 1:02d}",
        "split": split,
        "prompt": prompt,
        "rows": rows,
        "region": region,
        "threshold": threshold,
    }


def tasks() -> list[dict]:
    """返回全部题目。纯函数：同样的代码永远得到同样的数据集。"""
    result = []
    for split, seed, count in SPLITS:
        rng = random.Random(seed)
        for index in range(count):
            result.append(_build_task(split, index, rng))
    return result


def find_task(task_id: str) -> dict | None:
    return next((task for task in tasks() if task["id"] == task_id), None)


def expected(task: dict) -> dict:
    """标准答案由 Python 计算，不经过模型，也不执行模型给出的计算过程。"""
    rows = _select(task)
    return {
        "ids": [row["id"] for row in rows],
        "count": len(rows),
        "total": sum(row["qty"] * row["price"] for row in rows),
    }


# ---------------------------------------------------------------------------
# 批改器
# ---------------------------------------------------------------------------


def parse_json(text: str):
    """严格 JSON：重复字段直接报错，避免"后一个值覆盖前一个值"蒙对答案。"""

    def no_duplicates(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"重复字段: {key}")
            obj[key] = value
        return obj

    return json.loads(text, object_pairs_hook=no_duplicates)


FENCED_JSON = re.compile(r"```json[ \t]*\r?\n(?P<body>.*?)\r?\n```", re.DOTALL)


def normalize_answer(answer):
    """只拆掉"末尾唯一的 json 代码块"这层外壳，不碰字段、数值和顺序。

    允许模型先写计算过程，但要求：
      - 全文恰好一个 json 代码块；
      - 代码块之后只有空白；
      - 代码块之前不能再出现 ```（防止多个代码块里挑一个）。
    返回值第二项标记是否做过这层提取，便于事后审计。
    """
    if not isinstance(answer, str):
        return answer, False
    matches = list(FENCED_JSON.finditer(answer))
    if len(matches) != 1:
        return answer, False
    match = matches[0]
    if answer[match.end():].strip():
        return answer, False
    if "```" in answer[: match.start()]:
        return answer, False
    return match.group("body"), True


def _evaluate_strict(task: dict, answer):
    """五项检查，每项 0.2 分：JSON 合法、字段合法、ids、count、total。

    反馈只指出"错在哪条规则"，不泄露标准答案的具体数值。
    """
    try:
        obj = parse_json(answer)
    except (ValueError, TypeError):
        return {
            "score": 0.0,
            "passed": False,
            "feedback": [
                "最终答案必须是合法 JSON 对象；计算过程放在唯一的末尾 json 代码块之前，且不能出现重复字段。"
            ],
        }

    valid = (
        isinstance(obj, dict)
        and set(obj) == set(ANSWER_KEYS)
        and isinstance(obj["ids"], list)
        and all(type(item) is str for item in obj["ids"])
        and type(obj["count"]) is int
        and type(obj["total"]) is int
    )
    if not valid:
        return {
            "score": 0.2,
            "passed": False,
            "feedback": ["必须且只含 ids（字符串数组）、count（整数）、total（整数）三个字段，布尔值不是整数。"],
        }

    gold = expected(task)
    checks = [obj[key] == gold[key] for key in ANSWER_KEYS]
    tips = {
        "ids": "ids 有误：复核三个筛选条件，按 qty 降序、同 qty 按 id 升序，不能遗漏或重复。",
        "count": "count 有误：重新统计同时满足全部条件的订单数量。",
        "total": "total 有误：只对入选订单逐项计算 qty×price，再求和。",
    }
    return {
        "score": (2 + sum(checks)) / 5,
        "passed": all(checks),
        "feedback": [tips[key] for key, ok in zip(ANSWER_KEYS, checks) if not ok],
    }


def evaluate(task: dict, answer) -> dict:
    """对外统一的批改入口，额外记录规范化结果，便于事后审计。"""
    normalized, postprocessed = normalize_answer(answer)
    result = _evaluate_strict(task, normalized)
    result["normalized_answer"] = normalized
    result["postprocessed"] = postprocessed
    return result


def repair_feedback(task: dict, row: dict, round_index: int) -> str:
    """生成下一轮的修订提示。

    训练题给"带标签"的字段反馈（当前值 / 正确值），因为训练阶段本来就有标签；
    验证题和测试题只给错误类别，绝不出现标准答案数值，否则等于把答案喂回去。
    """
    feedback = list(row["evaluation"]["feedback"])
    if task["split"] == "train":
        try:
            actual = parse_json(normalize_answer(row["response"]["answer"])[0])
        except (ValueError, TypeError):
            actual = {}
        actual = actual if isinstance(actual, dict) else {}
        for key in ANSWER_KEYS:
            target = expected(task)[key]
            if type(actual.get(key)) is not type(target) or actual.get(key) != target:
                feedback.append(
                    f"训练标签 {key}：当前值={json.dumps(actual.get(key), ensure_ascii=False)}；"
                    f"正确值={json.dumps(target, ensure_ascii=False)}。"
                )
    method = (
        "重新逐行检查 active、region、qty 三个条件；先把入选行列齐，再排序、计数、算金额。"
        if round_index == 1
        else "上一轮修正仍未通过。请完全从原始订单重新计算，不要沿用之前的合计；"
        "逐项核对 qty×price，并检查边界条件和 qty 相同时的 id 顺序。"
    )
    return (
        "程序检查发现：\n"
        + "\n".join(feedback)
        + "\n"
        + method
        + "\n先写复核计算过程，最后在唯一的 json 代码块中输出完整答案。"
    )


# ---------------------------------------------------------------------------
# 单题作答
# ---------------------------------------------------------------------------


def messages_for(task: dict, memory) -> list[dict]:
    """经验放在 system 里，题目放在 user 里，经验不修改题目原文。"""
    messages = []
    if memory:
        messages.append({"role": "system", "content": MEMORY_HEADER + "\n" + "\n".join(memory)})
    messages.append({"role": "user", "content": task["prompt"]})
    return messages


async def call_model(generate, messages) -> dict:
    """调用一次模型并留痕。

    上游异常一律记为 status=error 并终止本题，绝不把"接口报错"当成"模型答得差"，
    否则会污染分数和经验。异常只保留类型名，不落盘正文，避免泄露密钥或请求头。
    """
    started = time.monotonic()
    try:
        response = await asyncio.wait_for(generate(messages), timeout=CALL_TIMEOUT)
        if not isinstance(response.get("answer"), str):
            raise ValueError("缺少文本答案")
        return {
            "status": "ok",
            "response": response,
            "messages": messages,
            "seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001 - 任何上游异常都要转成可审计的记录
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "messages": messages,
            "seconds": round(time.monotonic() - started, 3),
        }


async def refine(task: dict, generate, memory=(), max_calls: int = MAX_ROUNDS) -> dict:
    """首答 + 最多 max_calls-1 次修订。

    - 每次修订都接着上一轮的完整对话继续，模型能看到自己错在哪；
    - 保留得分最高的那一轮作为 best_answer，防止"越改越差"；
    - 一旦通过就提前结束，不浪费调用次数。
    """
    if not 1 <= max_calls <= MAX_ROUNDS:
        raise ValueError(f"每题只能调用 1 至 {MAX_ROUNDS} 次")

    history: list[dict] = []
    best: dict | None = None
    messages = messages_for(task, memory)
    stop = "call_limit"

    for index in range(max_calls):
        if history:
            previous = history[-1]
            messages = messages + [
                {"role": "assistant", "content": previous["response"]["answer"]},
                {"role": "user", "content": repair_feedback(task, previous, index)},
            ]
        row = await call_model(generate, list(messages))
        row["round"] = index
        history.append(row)

        if row["status"] == "error":
            stop = "upstream_error"
            break

        row["evaluation"] = evaluate(task, row["response"]["answer"])
        row["accepted"] = best is None or row["evaluation"]["score"] > best["evaluation"]["score"]
        if row["accepted"]:
            best = row
        if row["evaluation"]["passed"]:
            stop = "passed"
            break

    return {
        "task_id": task["id"],
        "split": task["split"],
        "memory": list(memory),
        "feedback_kind": "labeled_training" if task["split"] == "train" else "rule_only",
        "history": history,
        "passed": bool(best and best["evaluation"]["passed"]),
        "best_answer": best["response"]["answer"] if best else None,
        "best_score": best["evaluation"]["score"] if best else 0.0,
        "stop_reason": stop,
    }


# ---------------------------------------------------------------------------
# 学习：提炼规则 + 验证入库
# ---------------------------------------------------------------------------


async def validate_rule(generate, memory, rule: str) -> dict:
    """验证集前后对照：同一批题，分别带"旧经验"和"旧经验 + 新规则"各答一次。

    规则入库需要同时满足三条：
      1. 8 道题全部跑完（没有上游故障，否则数据不完整）；
      2. 没有任何一道题从"对"变成"错"（不退步）；
      3. 通过数严格增加（有长进）。
    "没长进"和"长进但退步"都拒绝——经验库只收严格更优的规则。
    """
    before: list[dict] = []
    after: list[dict] = []
    for task in (item for item in tasks() if item["split"] == "validation"):
        before.append(await refine(task, generate, memory, max_calls=1))
        after.append(await refine(task, generate, [*memory, rule], max_calls=1))

    complete = all(run["stop_reason"] != "upstream_error" for run in before + after)
    old = sum(run["passed"] for run in before)
    new = sum(run["passed"] for run in after)
    no_regression = all(not a["passed"] or b["passed"] for a, b in zip(before, after))
    return {
        "accepted": complete and no_regression and new > old,
        "complete": complete,
        "before_passed": old,
        "after_passed": new,
        "count": len(before),
        "no_regression": no_regression,
        "before": before,
        "after": after,
    }


PROPOSAL_SYSTEM = (
    "你是经验提炼器。阅读一次失败→修正的过程，总结一条简短、通用、可复用的检查规则。"
    '只返回 {"rule":"规则"} 这一个 JSON 对象。'
    "规则里不得出现具体订单 ID、题号、答案数值，也不得改动或新增任务本身的规则。"
)


def _valid_rule(rule, memory) -> tuple[bool, str]:
    """规则的硬性校验。宁可拒绝，也不让"抄答案"或"复述题面"混进经验库。"""
    if not isinstance(rule, str):
        return False, "invalid_proposal"
    rule = rule.strip()
    if not 1 <= len(rule) <= MAX_RULE_CHARS:
        return False, "invalid_or_duplicate_rule"
    if any(char.isdigit() for char in rule):
        # 数字往往意味着把某道题的答案数值写进来了。
        return False, "invalid_or_duplicate_rule"
    if rule in memory:
        return False, "invalid_or_duplicate_rule"
    return True, ""


async def learn(task: dict, generate, memory) -> dict:
    """从一道训练题里学一条规则。

    只有"首答错、修订后对"的样本才有信息量：
      - 首答就对，说明这题不需要新经验；
      - 改也改不对，说明这条修订轨迹没有可复用的东西。
    """
    if task["split"] != "train":
        raise ValueError("只有 train 划分的题目可以触发学习")

    result = {
        "task_id": task["id"],
        "accepted": False,
        "reason": "",
        "memory_before": list(memory),
        "memory_after": list(memory),
    }
    if len(memory) >= MAX_RULES:
        return result | {"reason": "memory_full"}

    run = await refine(task, generate, memory)
    result["training"] = run

    if run["stop_reason"] == "upstream_error":
        return result | {"reason": "upstream_error"}

    first = run["history"][0]
    first_passed = bool((first.get("evaluation") or {}).get("passed"))
    if not run["passed"] or first_passed:
        return result | {"reason": "no_successful_repair"}

    proposal = await call_model(
        generate,
        [
            {"role": "system", "content": PROPOSAL_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": task["prompt"],
                        "wrong": first["response"]["answer"],
                        "corrected": run["best_answer"],
                        "feedback": first["evaluation"]["feedback"],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )
    result["proposal"] = proposal
    if proposal["status"] == "error":
        return result | {"reason": "upstream_error"}

    try:
        rule = parse_json(proposal["response"]["answer"])["rule"]
    except (KeyError, TypeError, ValueError):
        return result | {"reason": "invalid_proposal"}

    ok, reason = _valid_rule(rule, memory)
    if not ok:
        return result | {"reason": reason}
    rule = rule.strip()

    trial = await validate_rule(generate, memory, rule)
    result.update(
        rule=rule,
        validation=trial,
        accepted=trial["accepted"],
        reason="accepted" if trial["accepted"] else "validation_rejected",
    )
    if trial["accepted"]:
        result["memory_after"] = [*memory, rule]
    return result


# ---------------------------------------------------------------------------
# 三组对照
# ---------------------------------------------------------------------------


async def compare(task: dict, generate, memory) -> dict:
    """测试集上的三组对照。

    - baseline：直接回答（复用 feedback 组的首答，避免重复调用引入随机差异）；
    - feedback：反馈修正，最多 3 次调用；
    - memory_single：带上经验只答一次。

    经验为空时 memory_single 直接复用 baseline，并显式标记 shared_with_baseline，
    避免把"同一份答案"当成两组独立结果去比较。
    """
    feedback = await refine(task, generate, max_calls=MAX_ROUNDS)
    first = feedback["history"][0]
    baseline = {
        "passed": bool((first.get("evaluation") or {}).get("passed")),
        "history": [first],
    }
    if memory:
        remembered = await refine(task, generate, memory, max_calls=1)
    else:
        remembered = dict(baseline, shared_with_baseline=True)
    return {
        "task_id": task["id"],
        "memory": list(memory),
        "baseline": baseline,
        "feedback": feedback,
        "memory_single": remembered,
    }


def summarize(records) -> dict:
    """把 evaluate 的记录压成一张对照表，供 CLI 直接打印。"""
    records = list(records)
    summary = {}
    for group in ("baseline", "feedback", "memory_single"):
        rows = [record[group] for record in records if group in record]
        summary[group] = {
            "passed": sum(1 for row in rows if row["passed"]),
            "count": len(rows),
            "mean_calls": round(sum(len(row["history"]) for row in rows) / len(rows), 2) if rows else 0.0,
            "mean_seconds": round(
                sum(sum(step["seconds"] for step in row["history"]) for row in rows) / len(rows), 2
            )
            if rows
            else 0.0,
        }
    return summary


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------


class Store:
    """经验库与运行记录的落盘。

    目录按 (调用身份, 模型, 数据集版本) 的哈希隔离：
      - 换用户 / 换 API Key 不会串味；
      - 换模型不会沿用旧模型的规则；
      - 数据集版本变了，旧规则自动失效。
    写入用 tmp + replace，进程被中断也不会留下半截 memory.json。
    """

    def __init__(self, root):
        self.root = Path(root)
        self.lock = asyncio.Lock()

    def folder(self, scope, model: str) -> Path:
        digest = hashlib.sha256(
            json.dumps([list(scope), model], sort_keys=True).encode("utf-8")
        ).hexdigest()
        return self.root / digest

    def memory(self, scope, model: str) -> list[str]:
        path = self.folder(scope, model) / "memory.json"
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [rule for rule in data if isinstance(rule, str)] if isinstance(data, list) else []

    def runs(self, scope, model: str, limit: int = 20) -> list[dict]:
        """最近的运行记录摘要，供 /v1/rsl/runs 做可视化与排查。"""
        folder = self.folder(scope, model)
        if not folder.exists():
            return []
        files = sorted(
            (path for path in folder.glob("*.json") if path.name != "memory.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:limit]
        result = []
        for path in files:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            result.append(
                {
                    "run_id": record.get("run_id", path.stem),
                    "created_at": record.get("created_at"),
                    "mode": record.get("mode"),
                    "task_id": record.get("task_id"),
                    "passed": record.get("passed"),
                    "accepted": record.get("accepted"),
                    "reason": record.get("reason"),
                    "stop_reason": record.get("stop_reason"),
                    "best_score": record.get("best_score"),
                }
            )
        return result

    def save(self, scope, model: str, record: dict, memory=None) -> dict:
        folder = self.folder(scope, model)
        folder.mkdir(parents=True, exist_ok=True)
        record = dict(record)
        record["run_id"] = uuid.uuid4().hex
        record["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (folder / f"{record['run_id']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if memory is not None:
            temp = folder / "memory.tmp"
            temp.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(folder / "memory.json")
        return record
