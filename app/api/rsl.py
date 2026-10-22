"""RSL 接入层：复用 ModelRouter 的身份验证、模型权限和 Provider。

这里刻意不新建一套鉴权，也不自己拼 HTTP 请求：
    resolve_request_context  → 谁在调用、能看哪些模型（与 /v1/chat/completions 一致）
    load_routing_policy      → routing_profile 里的 allowed / deny 规则
    engine.route             → 拿到 (provider, model_config)，密钥轮转、健康度、重试全走原逻辑

RSL 只负责"问什么、问几次、记不记住经验"，不负责"选哪个模型"。
"""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.auth import resolve_request_context
from app.core.request_context import RequestContext
from app.core.router import DISABLED_ROUTING_KEYS
from app.core.routing_policy import load_routing_policy
from app.core.rsl import DATASET_VERSION, compare, learn, refine, tasks
from app.schemas.chat import ChatMessage, ChatRequest

router = APIRouter(prefix="/v1/rsl", tags=["rsl"])

DEFAULT_MODEL = "homework"
MAX_TOKENS = 2048


class RunRequest(BaseModel):
    task_id: str
    model: str = Field(default=DEFAULT_MODEL, min_length=1, max_length=120)
    mode: Literal["single", "feedback", "learn", "compare"] = "feedback"
    use_memory: bool = True


def scope_for(context: RequestContext) -> list:
    """经验隔离维度：调用主体 + 用户 + 路由档案。"""
    return [context.subject_type, context.subject_id, context.user_id, context.routing_profile_id]


def identity_for(model_key: str, actual_model: str) -> str:
    """把"路由键"和"真实模型名"一起写进隔离标识。

    路由键可能被运维改指向别的模型，只按路由键隔离会导致换模型后误用旧经验。
    """
    return f"{model_key}:{actual_model}:{DATASET_VERSION}"


async def resolve_model(request: Request, context: RequestContext, key: str):
    """按 ModelRouter 的策略校验模型权限，返回 (provider, model_config)。"""
    engine = request.app.state.router
    policy = await load_routing_policy(request.app.state.db, context.routing_profile_id)
    allowed = policy.effective_allowed_models(context.allowed_models, set(engine.model_map))
    if key not in allowed or key in DISABLED_ROUTING_KEYS or key not in engine.model_map:
        raise HTTPException(403, "模型不存在或没有使用权限")
    return await engine.route(key)


@router.get("/tasks")
async def catalog(context: RequestContext = Depends(resolve_request_context)):
    """题目清单。不含标准答案，也不含 rows 以外的内部字段。"""
    return [
        {"id": task["id"], "split": task["split"], "prompt": task["prompt"]}
        for task in tasks()
    ]


@router.get("/memory")
async def read_memory(
    request: Request,
    model: str = DEFAULT_MODEL,
    context: RequestContext = Depends(resolve_request_context),
):
    """查看当前身份 + 当前模型下已入库的经验规则。"""
    _, config = await resolve_model(request, context, model)
    identity = identity_for(model, config.name)
    return {
        "model": model,
        "actual_model": config.name,
        "dataset_version": DATASET_VERSION,
        "rules": request.app.state.rsl.memory(scope_for(context), identity),
    }


@router.get("/runs")
async def read_runs(
    request: Request,
    model: str = DEFAULT_MODEL,
    limit: int = 20,
    context: RequestContext = Depends(resolve_request_context),
):
    """最近的运行记录摘要，方便复现和排查（完整记录仍在磁盘上）。"""
    _, config = await resolve_model(request, context, model)
    identity = identity_for(model, config.name)
    return {
        "model": model,
        "actual_model": config.name,
        "runs": request.app.state.rsl.runs(scope_for(context), identity, limit=limit),
    }


@router.post("/run")
async def run(
    body: RunRequest,
    request: Request,
    context: RequestContext = Depends(resolve_request_context),
):
    """跑一道题。

    mode:
      single   只答一次，带经验（如果有）
      feedback 首答 + 最多 2 次反馈修订
      learn    训练题专用：作答 → 提炼规则 → 验证 → 可能入库
      compare  测试题专用：直接回答 / 反馈修正 / 带经验单次回答 三组对照
    """
    task = next((item for item in tasks() if item["id"] == body.task_id), None)
    if task is None:
        raise HTTPException(404, "未知题目")
    if body.mode == "learn" and task["split"] != "train":
        raise HTTPException(422, "只能从 train 题目学习；验证题和测试题禁止更新经验")

    provider, config = await resolve_model(request, context, body.model)
    store = request.app.state.rsl
    scope = scope_for(context)
    identity = identity_for(body.model, config.name)

    async def generate(messages):
        """一次模型调用。仍然走 ModelRouter 的 Provider，密钥与重试由它处理。"""
        response = await provider.chat_completions(
            ChatRequest(
                model=body.model,
                messages=[ChatMessage(**message) for message in messages],
                max_tokens=MAX_TOKENS,
                temperature=0,
            ),
            config,
        )
        return {
            "answer": response.choices[0].message.content or "",
            "model": response.model,
            "usage": response.usage.model_dump() if response.usage else None,
        }

    # 单进程串行化：避免两个请求同时读到旧经验、各自写入、互相覆盖。
    async with store.lock:
        saved = store.memory(scope, identity)
        rules = saved if body.use_memory or body.mode == "learn" else []
        if body.mode == "learn":
            result = await learn(task, generate, saved)
        elif body.mode == "compare":
            result = await compare(task, generate, rules)
        else:
            result = await refine(
                task, generate, rules, max_calls=1 if body.mode == "single" else 3
            )

        result.update(
            mode=body.mode,
            model=body.model,
            actual_model=config.name,
            dataset_version=DATASET_VERSION,
            simulated=False,
        )
        return store.save(
            scope,
            identity,
            result,
            memory=result["memory_after"] if body.mode == "learn" else None,
        )
