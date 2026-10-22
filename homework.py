"""RSL 作业的唯一入口。

    python homework.py demo       离线演示（假模型，不发网络请求）
    python homework.py serve      启动 ModelRouter（含 /v1/rsl 接口）
    python homework.py learn      在 train 划分上跑学习循环，更新经验库
    python homework.py evaluate   在 test 划分上跑三组对照

serve 需要单独开一个终端并保持运行；learn / evaluate 通过 HTTP 调用它，
这样"学习"确实经过了 ModelRouter 的鉴权、模型权限和 Provider 转发。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLACEHOLDER_TOKEN = "replace-with-your-local-access-token"


def server_url() -> str:
    """服务地址。放在函数里读环境变量，确保 .env 已经加载。"""
    return os.environ.get("RSL_SERVER") or f"http://127.0.0.1:{os.environ.get('RSL_PORT', '8000')}"


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 离线演示
# ---------------------------------------------------------------------------


async def demo() -> None:
    """用一个"首答故意算错、收到反馈后改对"的假模型演示机制。

    不连模型、不出网，用来确认批改器、修订循环和落盘都正常。
    这不是模型效果，不能写进实验结论。
    """
    from app.core.rsl import compare, expected, learn, refine, tasks

    by_prompt = {task["prompt"]: task for task in tasks()}

    def is_proposal(messages) -> bool:
        return messages[0]["role"] == "system" and "经验提炼器" in messages[0]["content"]

    async def fake_model(messages):
        """假模型的行为约定：

        - 被要求提炼规则时，返回一条不含数字的通用规则；
        - 提示词里有经验（system 消息），或已经进入修订轮（对话里有 assistant），就答对；
        - 否则答错（total 故意算成 -1）。
        这样"经验是否有效"就完全由提示词内容决定，闭环能被稳定复现。
        """
        if is_proposal(messages):
            rule = "先逐行筛选三个条件，再排序，最后逐项核算金额。"
            return {"answer": json.dumps({"rule": rule}, ensure_ascii=False), "model": "SIMULATED"}

        task = next(t for t in by_prompt.values() if any(m["content"] == t["prompt"] for m in messages))
        has_memory = messages[0]["role"] == "system"
        retried = any(m["role"] == "assistant" for m in messages)
        answer = expected(task)
        if not (has_memory or retried):
            answer["total"] = -1
        return {"answer": json.dumps(answer, ensure_ascii=False), "model": "SIMULATED"}

    train_task = next(t for t in tasks() if t["split"] == "train")
    test_task = next(t for t in tasks() if t["split"] == "test")

    print("【1】单题修订：首答算错金额，程序指出错误，模型在第二轮改对")
    run = await refine(train_task, fake_model)
    for row in run["history"]:
        print(f'  第 {row["round"]} 轮：{row["response"]["answer"]}')
        print(f'    批改 score={row["evaluation"]["score"]} passed={row["evaluation"]["passed"]} '
              f'feedback={row["evaluation"]["feedback"]}')
    print(f'  共 {len(run["history"])} 轮，停止原因 {run["stop_reason"]}，通过={run["passed"]}')

    print()
    print("【2】学习：从这次成功修订里提炼规则，并用 8 道验证题做前后对照")
    learned = await learn(train_task, fake_model, [])
    trial = learned.get("validation") or {}
    print(f'  候选规则：{learned.get("rule")}')
    print(f'  验证通过数 {trial.get("before_passed")}/{trial.get("count")} → '
          f'{trial.get("after_passed")}/{trial.get("count")}，'
          f'无退步={trial.get("no_regression")}，入库={learned["accepted"]}（{learned["reason"]}）')
    memory = learned["memory_after"]
    print(f'  当前经验库（{len(memory)} 条）：{memory}')

    print()
    print("【3】对照：同一道测试题，直接回答 / 反馈修正 / 带经验单次回答")
    result = await compare(test_task, fake_model, memory)
    print(f'  直接回答     通过={result["baseline"]["passed"]}  '
          f'调用 {len(result["baseline"]["history"])} 次')
    print(f'  反馈修正     通过={result["feedback"]["passed"]}  '
          f'调用 {len(result["feedback"]["history"])} 次')
    print(f'  带经验单次   通过={result["memory_single"]["passed"]}  '
          f'调用 {len(result["memory_single"]["history"])} 次')

    payload = {
        "simulated": True,
        "note": "假模型演示，用于验证机制，不代表任何真实模型的效果",
        "refine": run,
        "learn": learned,
        "compare": result,
    }
    save(RESULTS / "offline-demo.json", payload)
    print()
    print("以上为人为设计的假模型演示，不是实际模型效果；完整记录见 results/offline-demo.json")



# ---------------------------------------------------------------------------
# 真实实验
# ---------------------------------------------------------------------------


def request_run(body: dict) -> dict:
    """调用本机的 /v1/rsl/run。trust_env=False：不让系统 HTTP 代理劫持 127.0.0.1。"""
    import httpx

    token = os.environ.get("ROUTER_INTERNAL_API_KEY", "")
    if not token or token == PLACEHOLDER_TOKEN:
        raise SystemExit("请先把 .env.example 复制成 .env，并设置自己的 ROUTER_INTERNAL_API_KEY。")
    with httpx.Client(timeout=1800, trust_env=False) as client:
        response = client.post(
            f"{server_url()}/v1/rsl/run",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
        )
        if response.status_code >= 400:
            raise SystemExit(f"接口返回 {response.status_code}：{response.text[:300]}")
        return response.json()


def has_error(obj) -> bool:
    """递归查找上游故障标记；一旦出现就停止整轮实验，避免把故障算成答错。"""
    if isinstance(obj, dict):
        return obj.get("status") == "error" or any(has_error(value) for value in obj.values())
    if isinstance(obj, list):
        return any(has_error(value) for value in obj)
    return False


def write_summary(path: Path, summary: dict, dataset_version: str) -> None:
    """把对照结果写成 Markdown 表格，方便直接贴进 README。"""
    labels = {
        "baseline": "直接回答",
        "feedback": "反馈修正",
        "memory_single": "带经验单次回答",
    }
    lines = [
        f"# 测试集对照（数据版本 {dataset_version}）",
        "",
        "| 方法 | 完全正确 | 平均调用次数 | 平均调用耗时（秒） |",
        "|---|---:|---:|---:|",
    ]
    for group, label in labels.items():
        row = summary.get(group)
        if not row or not row["count"]:
            continue
        lines.append(
            f'| {label} | {row["passed"]}/{row["count"]} '
            f'| {row["mean_calls"]:.2f} | {row["mean_seconds"]:.2f} |'
        )
    lines.append("")
    lines.append("平均耗时包含本地接口与网络开销，不是纯推理时间；均为单次运行数据。")
    path.write_text("\n".join(lines), encoding="utf-8")


def experiment(command: str, model: str) -> None:
    from app.core.rsl import DATASET_VERSION, summarize, tasks

    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = RESULTS / f"{command}-{stamp}.json"
    if path.exists():
        raise SystemExit("同名结果已存在，请稍后再运行。")

    split = "train" if command == "learn" else "test"
    dataset = [task for task in tasks() if task["split"] == split]
    result = {
        "simulated": False,
        "command": command,
        "model": model,
        "dataset_version": DATASET_VERSION,
        "planned": len(dataset),
        "complete": False,
        "records": [],
    }
    save(path, result)

    frozen_memory = None
    try:
        for task in dataset:
            body = {
                "task_id": task["id"],
                "model": model,
                "mode": "learn" if command == "learn" else "compare",
            }
            row = request_run(body)
            result["records"].append(row)
            save(path, result)

            flag = ""
            if command == "learn":
                flag = f'接受={row.get("accepted")} 原因={row.get("reason")}'
            else:
                flag = (f'直接={row["baseline"]["passed"]} 反馈={row["feedback"]["passed"]} '
                        f'经验={row["memory_single"]["passed"]}')
            print(f'{task["id"]} 已记录 | {flag} | {path.name}', flush=True)

            if has_error(row):
                result["stop_reason"] = "upstream_error"
                print("检测到上游故障，停止本轮；已保存完成的部分。")
                break
            if command == "evaluate":
                if frozen_memory is None:
                    frozen_memory = row["memory"]
                elif frozen_memory != row["memory"]:
                    result["stop_reason"] = "memory_changed_during_test"
                    print("测试期间经验发生变化，停止；本次结果不可作为对照。")
                    break
        else:
            result["complete"] = True
    except KeyboardInterrupt:
        result["stop_reason"] = "interrupted"
        print("已中断，保留完成的部分。")
    except Exception as exc:  # noqa: BLE001
        result["stop_reason"] = type(exc).__name__
        print(f"调用中断（{type(exc).__name__}）；检查服务与模型配置后重试。")
    finally:
        if command == "evaluate" and result["records"]:
            result["summary"] = summarize(result["records"])
            write_summary(RESULTS / f"summary-{stamp}.md", result["summary"], DATASET_VERSION)
        save(path, result)

    brief = {key: value for key, value in result.items() if key != "records"}
    print(json.dumps(brief, ensure_ascii=False, indent=2))
    print(f"完整记录：{path}")


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------


def serve() -> None:
    token = os.environ.get("ROUTER_INTERNAL_API_KEY", "")
    if not token or token == PLACEHOLDER_TOKEN:
        raise SystemExit("先复制 .env.example 为 .env，并把 ROUTER_INTERNAL_API_KEY 改成自己的口令。")
    missing = [
        key for key in ("OPENAI_BASE_URL", "OPENAI_KEY_1", "HOMEWORK_MODEL") if not os.environ.get(key)
    ]
    if missing:
        raise SystemExit(f".env 里还缺：{', '.join(missing)}")

    Path(os.environ.get("ROUTER_DB", "results/router.db")).parent.mkdir(parents=True, exist_ok=True)
    Path(os.environ.get("RSL_OUTPUT", "results/private")).mkdir(parents=True, exist_ok=True)

    import uvicorn

    # 单进程：经验库的串行锁只在单进程内有效。
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=int(os.environ.get("RSL_PORT", "8000")),
        workers=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RSL 作业入口")
    parser.add_argument("command", choices=["demo", "serve", "learn", "evaluate"])
    parser.add_argument(
        "--model",
        default=os.environ.get("HOMEWORK_ROUTING_KEY", "homework"),
        help="routing key，默认 homework（在 config/providers.yaml 中配置）",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    # 本机调用必须绕过系统代理，否则 Windows 上的 HTTP_PROXY 会把 127.0.0.1 也转发出去。
    os.environ["NO_PROXY"] = ",".join(
        filter(None, [os.environ.get("NO_PROXY", ""), "127.0.0.1", "localhost"])
    )

    if args.command == "demo":
        asyncio.run(demo())
    elif args.command == "serve":
        serve()
    else:
        print("即将调用真实模型，付费接口会产生费用；按 Ctrl+C 可中断。", flush=True)
        experiment(args.command, args.model)


if __name__ == "__main__":
    main()
