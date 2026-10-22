"""RSL 离线测试：不连模型、不出网。

覆盖四层：
    1. 批改器（严格 JSON、字段、分数）
    2. 修订循环（预算、保留最优、反馈不泄题）
    3. 学习循环（何时提炼、验证准入、拒绝条件）
    4. 接入层（真实 ModelRouter 的鉴权 / 权限 / Provider 转发 / 重启后经验仍在）
"""
import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.rsl import (
    DATASET_VERSION,
    Store,
    compare,
    evaluate,
    expected,
    learn,
    normalize_answer,
    refine,
    summarize,
    tasks,
    validate_rule,
)


def run(coro):
    return asyncio.run(coro)


def output(obj):
    return {"answer": json.dumps(obj, ensure_ascii=False)}


def task_from_messages(messages):
    return next(t for t in tasks() if any(m["content"] == t["prompt"] for m in messages))


# ---------------------------------------------------------------------------
# 1. 批改器
# ---------------------------------------------------------------------------


def test_verifier_known_answer_and_strict_schema():
    task = {
        "split": "test",
        "region": "east",
        "threshold": 3,
        "rows": [
            {"id": "B", "active": True, "region": "east", "qty": 3, "price": 10},
            {"id": "A", "active": True, "region": "east", "qty": 3, "price": 20},
            {"id": "C", "active": False, "region": "east", "qty": 9, "price": 99},
            {"id": "D", "active": True, "region": "west", "qty": 8, "price": 99},
        ],
    }
    gold = {"ids": ["A", "B"], "count": 2, "total": 90}
    assert expected(task) == gold
    assert evaluate(task, json.dumps(gold))["passed"]

    # 重复字段、布尔冒充整数、多余字段，一律拒绝
    assert not evaluate(task, '{"ids":["A","B"],"count":2,"total":90,"total":90}')["passed"]
    assert not evaluate(task, json.dumps(gold | {"count": True}))["passed"]
    assert not evaluate(task, json.dumps(gold | {"extra": 1}))["passed"]


def test_verifier_extracts_only_one_trailing_json_fence():
    task = next(t for t in tasks() if t["split"] == "test")
    gold = expected(task)
    body = json.dumps(gold, ensure_ascii=False)

    # 允许先写计算过程（含单个反引号标记的字段名），末尾一个 json 代码块
    wrapped = evaluate(task, "核算 `qty` 与 `price`。\n```json\n" + body + "\n```")
    assert wrapped["passed"] and wrapped["postprocessed"]
    assert json.loads(wrapped["normalized_answer"]) == gold

    # 代码块之后还有文字、两个代码块、没有代码块但混了说明文字，都不通过
    assert not evaluate(task, "```json\n" + body + "\n```\n补充说明")["passed"]
    assert not evaluate(task, "```json\n{}\n```\n```json\n" + body + "\n```")["passed"]
    assert not evaluate(task, "答案是 " + body)["passed"]


def test_normalize_answer_is_pure_extraction():
    assert normalize_answer('{"a":1}') == ('{"a":1}', False)
    assert normalize_answer("前文\n```json\n{\"a\":1}\n```\n  ") == ('{"a":1}', True)
    assert normalize_answer("```json\n{\"a\":1}\n```\n尾巴")[1] is False
    assert normalize_answer(None) == (None, False)


def test_dataset_is_fixed_disjoint_and_non_degenerate():
    data = tasks()
    assert data == tasks()
    assert len({t["id"] for t in data}) == 40
    assert len({t["prompt"] for t in data}) == 40
    counts = [sum(t["split"] == s for t in data) for s in ("train", "validation", "test")]
    assert counts == [12, 8, 20]
    # 每题至少选中 2 行，避免"空答案"退化题
    assert all(len(expected(t)["ids"]) >= 2 for t in data)
    assert all(t["id"].startswith(t["split"]) for t in data)


# ---------------------------------------------------------------------------
# 2. 修订循环
# ---------------------------------------------------------------------------


def test_repair_shares_initial_answer_and_hides_gold_from_test():
    task, calls = tasks()[-1], []

    async def fake(messages):
        calls.append(messages)
        return output(expected(task) if len(calls) > 1 else expected(task) | {"total": -1})

    result = run(compare(task, fake, []))
    assert not result["baseline"]["passed"] and result["feedback"]["passed"]
    # 基线复用反馈组的首答，避免重复调用带来的随机差异
    assert result["baseline"]["history"][0] == result["feedback"]["history"][0]
    assert result["memory_single"]["shared_with_baseline"]
    assert len(calls) == 2
    # 测试题的修订反馈里不能出现标准答案数值
    assert str(expected(task)["total"]) not in calls[1][-1]["content"]


def test_regression_keeps_best_answer_and_budget_stops():
    task = tasks()[0]
    answers = iter(
        [
            output(expected(task) | {"total": -1}),
            {"answer": "不是 JSON"},
            {"answer": "{}"},
        ]
    )

    async def fake(messages):
        return next(answers)

    result = run(refine(task, fake))
    assert len(result["history"]) == 3
    assert result["best_score"] == 0.8  # 首答只错 total，保留为最优
    assert result["stop_reason"] == "call_limit"
    assert result["passed"] is False


def test_labeled_feedback_is_training_only_and_rounds_grow():
    for split in ("train", "validation", "test"):
        task = next(t for t in tasks() if t["split"] == split)
        calls = []

        async def fake(messages):
            calls.append(messages)
            return output(expected(task) | {"total": -1})

        result = run(refine(task, fake))
        assert len(calls) == 3
        assert [len(m) for m in calls] == [1, 3, 5]  # 每轮接着上一轮，不是重开
        assert calls[1][-1]["content"] != calls[2][-1]["content"]
        feedback = calls[1][-1]["content"]
        is_train = split == "train"
        assert ("正确值=" in feedback) is is_train
        assert (str(expected(task)["total"]) in feedback) is is_train
        assert result["feedback_kind"] == ("labeled_training" if is_train else "rule_only")


def test_upstream_failure_is_not_scored_as_bad_answer():
    async def fail(messages):
        raise RuntimeError("provider secret detail")

    result = run(refine(tasks()[0], fail))
    assert result["stop_reason"] == "upstream_error"
    assert len(result["history"]) == 1 and not result["passed"]
    assert "provider secret detail" not in json.dumps(result)


def test_memory_is_injected_as_system_and_rules_do_not_touch_prompt():
    task = tasks()[0]
    seen = []

    async def fake(messages):
        seen.append(messages)
        return output(expected(task))

    run(refine(task, fake, ["先筛条件再排序。"], max_calls=1))
    assert seen[0][0]["role"] == "system" and "先筛条件再排序。" in seen[0][0]["content"]
    assert seen[0][-1]["content"] == task["prompt"]


# ---------------------------------------------------------------------------
# 3. 学习循环
# ---------------------------------------------------------------------------


def test_rule_is_saved_only_after_validation_improves():
    async def fake(messages):
        head = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if "经验提炼器" in head:
            return output({"rule": "先逐行筛条件，再排序，最后逐项核算金额。"})
        task = task_from_messages(messages)
        fixed = len(messages) > 1
        return output(expected(task) if fixed else expected(task) | {"total": -1})

    result = run(learn(tasks()[0], fake, []))
    assert result["accepted"] and result["reason"] == "accepted"
    assert result["validation"]["before_passed"] == 0
    assert result["validation"]["after_passed"] == 8
    assert len(result["memory_after"]) == 1
    assert all(row["task_id"].startswith("validation-") for row in result["validation"]["after"])


@pytest.mark.parametrize("case", ["regression", "equal", "failure"])
def test_rule_rejected_on_regression_equal_score_or_upstream_failure(case):
    async def fake(messages):
        task = task_from_messages(messages)
        new = messages[0]["role"] == "system"
        index = int(task["id"].split("-")[1])
        if case == "failure" and new:
            raise TimeoutError()
        if case == "equal":
            good = True
        else:  # 通过数从 1 涨到 7，但 validation-01 从对变错
            good = (index != 1) if new else (index == 1)
        return output(expected(task) if good else expected(task) | {"total": -1})

    result = run(validate_rule(fake, [], "先核对过滤条件。"))
    assert not result["accepted"]


def test_no_learning_from_test_split_and_none_from_already_correct():
    async def fake(messages):
        return output(expected(task_from_messages(messages)))

    with pytest.raises(ValueError):
        run(learn(tasks()[-1], fake, []))  # test 划分禁止学习
    result = run(learn(tasks()[0], fake, []))
    assert not result["accepted"] and result["reason"] == "no_successful_repair"


def test_memory_cap_and_rule_sanity_checks():
    async def fake(messages):
        head = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if "经验提炼器" in head:
            return output({"rule": "规则 20261022"})  # 含数字，疑似抄答案
        task = task_from_messages(messages)
        fixed = len(messages) > 1
        return output(expected(task) if fixed else expected(task) | {"total": -1})

    rejected = run(learn(tasks()[0], fake, []))
    assert not rejected["accepted"] and rejected["reason"] == "invalid_or_duplicate_rule"

    full = run(learn(tasks()[0], fake, ["一", "二", "三", "四", "五"]))
    assert full["reason"] == "memory_full" and full["accepted"] is False


# ---------------------------------------------------------------------------
# 4. 落盘与接入层
# ---------------------------------------------------------------------------


def test_store_isolates_scope_and_model_and_lists_runs(tmp_path):
    store = Store(tmp_path)
    store.save(["user1"], "m1", {"mode": "learn", "accepted": True}, memory=["核对条件。"])
    restored = Store(tmp_path)
    assert restored.memory(["user1"], "m1") == ["核对条件。"]
    assert restored.memory(["user2"], "m1") == []
    assert restored.memory(["user1"], "m2") == []
    runs = restored.runs(["user1"], "m1")
    assert len(runs) == 1 and runs[0]["run_id"] and runs[0]["accepted"] is True
    # 新版本数据集不会沿用旧经验
    assert restored.memory(["user1"], "m1") != restored.memory(["user1"], f"m1:{DATASET_VERSION}")


def test_summarize_reports_each_group():
    records = [
        {
            "baseline": {"passed": True, "history": [{"seconds": 1.0}]},
            "feedback": {"passed": True, "history": [{"seconds": 1.0}, {"seconds": 2.0}]},
            "memory_single": {"passed": False, "history": [{"seconds": 1.5}]},
        },
        {
            "baseline": {"passed": False, "history": [{"seconds": 1.0}]},
            "feedback": {"passed": False, "history": [{"seconds": 1.0}]},
            "memory_single": {"passed": True, "history": [{"seconds": 0.5}]},
        },
    ]
    summary = summarize(records)
    assert summary["baseline"] == {"passed": 1, "count": 2, "mean_calls": 1.0, "mean_seconds": 1.0}
    assert summary["feedback"]["mean_calls"] == 1.5
    assert summary["memory_single"]["passed"] == 1


def test_real_modelrouter_auth_policy_provider_and_restart(tmp_path, monkeypatch):
    """走真实 ModelRouter 启动链路：鉴权 → 模型权限 → Provider → 上游（MockTransport，不出网）。"""
    import httpx
    from app.main import app

    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    for key, value in {
        "ROUTER_RUNTIME_MODE": "internal",
        "ROUTER_ENABLE_ADMIN_API": "false",
        "ROUTER_ENABLE_STANDALONE_KEYS": "false",
        "ROUTER_INTERNAL_API_KEY": "test-local-token",
        "ROUTER_DB": str(tmp_path / "router.db"),
        "RSL_OUTPUT": str(tmp_path / "rsl"),
        "OPENAI_BASE_URL": "http://mock.invalid/v1",
        "OPENAI_KEY_1": "mock",
        "HOMEWORK_MODEL": "mock-model",
        "AUTO_SYNC_PROVIDER_MODELS": "false",
    }.items():
        monkeypatch.setenv(key, value)

    calls = []

    def transport(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["model"] == "mock-model"
        task = task_from_messages(body["messages"])
        return httpx.Response(
            200,
            json={
                "id": "mock",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": json.dumps(expected(task))}}
                ],
            },
        )

    original_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        return original_client(*args, **kwargs, transport=httpx.MockTransport(transport))

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)

    headers = {"Authorization": "Bearer test-local-token"}
    with TestClient(app) as client:
        assert client.get("/v1/rsl/tasks").status_code == 401
        assert len(client.get("/v1/rsl/tasks", headers=headers).json()) == 40
        # 测试题不能触发学习
        assert client.post(
            "/v1/rsl/run", headers=headers, json={"task_id": "test-01", "mode": "learn"}
        ).status_code == 422
        # 模型权限由 routing_profile / X-Allowed-Models 决定
        assert client.post(
            "/v1/rsl/run",
            headers=headers | {"X-Allowed-Models": "other"},
            json={"task_id": "train-01"},
        ).status_code == 403
        # 未知题目
        assert client.post(
            "/v1/rsl/run", headers=headers, json={"task_id": "nope"}
        ).status_code == 404
        response = client.post("/v1/rsl/run", headers=headers, json={"task_id": "train-01"})
        assert response.status_code == 200
        body = response.json()
        assert body["passed"] and body["simulated"] is False and body["run_id"]
        assert len(calls) == 1
        assert client.get("/health").json()["features"]["rsl"] is True

        app.state.rsl.save(
            ["internal", "internal", None, None],
            f"homework:mock-model:{DATASET_VERSION}",
            {"mode": "learn", "accepted": True},
            memory=["核对字段。"],
        )

    # 重启后经验仍在，且按用户隔离
    with TestClient(app) as client:
        assert client.get("/v1/rsl/memory", headers=headers).json()["rules"] == ["核对字段。"]
        assert client.get(
            "/v1/rsl/memory", headers=headers | {"X-User-Id": "other"}
        ).json()["rules"] == []
        assert client.get("/v1/rsl/runs", headers=headers).json()["runs"]
