import asyncio
import json

from app.core.exceptions import InvalidRequestError, ModelAccessError, ProviderError
from app.schemas.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EnsembleMetadata,
    EnsembleParticipantResult,
    Usage,
)


def ensemble_enabled(request: ChatRequest) -> bool:
    return bool(request.ensemble and request.ensemble.enabled is not False)


def _normalize_participants(participants: list[str] | None) -> list[str]:
    result = []
    seen = set()
    for model in participants or []:
        value = (model or "").strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def validate_ensemble_request(request: ChatRequest):
    if request.stream:
        raise InvalidRequestError("Streaming is not supported for ensemble requests")
    if not request.ensemble:
        raise InvalidRequestError("Ensemble config is required")

    participants = _normalize_participants(request.ensemble.participants)
    if len(participants) < 2:
        raise InvalidRequestError("Ensemble requests require at least two participant models")

    judge_model = (request.ensemble.judge_model or "").strip()
    if not judge_model:
        raise InvalidRequestError("Ensemble requests require a judge_model")

    strategy = request.ensemble.strategy or "synthesize"
    return {
        "participants": participants,
        "judge_model": judge_model,
        "strategy": strategy,
        "judge_system_prompt": request.ensemble.judge_system_prompt,
    }


def _calc_cost(pricing, usage: Usage | None) -> float:
    if not pricing or not usage:
        return 0.0
    return (
        (usage.prompt_tokens / 1_000_000) * pricing.prompt
        + (usage.completion_tokens / 1_000_000) * pricing.completion
    )


def _merge_usage(usages: list[Usage | None]) -> Usage | None:
    present = [usage for usage in usages if usage]
    if not present:
        return None
    return Usage(
        prompt_tokens=sum(usage.prompt_tokens for usage in present),
        completion_tokens=sum(usage.completion_tokens for usage in present),
        total_tokens=sum(usage.total_tokens for usage in present),
    )


def _choice_content(response: ChatResponse) -> str:
    if not response.choices:
        return ""
    return response.choices[0].message.content or ""


def _default_judge_system_prompt(strategy: str) -> str:
    if strategy == "select":
        return (
            "You are an algorithmic judge for a multi-model ensemble. "
            "Choose the single best candidate answer for the original request. "
            "You may make only minimal edits for clarity or factual correction. "
            "Return only the final answer, with no commentary about the candidates."
        )
    return (
        "You are an algorithmic judge for a multi-model ensemble. "
        "Synthesize the strongest final answer from the candidate responses. "
        "Prefer correctness, completeness, grounded reasoning, and directness. "
        "Return only the final answer, with no commentary about the candidates."
    )


def _build_judge_messages(
    request: ChatRequest,
    participant_results: list[EnsembleParticipantResult],
    strategy: str,
    judge_system_prompt: str | None = None,
) -> list[ChatMessage]:
    candidates = [
        {
            "model": item.routing_key,
            "provider": item.provider,
            "content": item.content or "",
        }
        for item in participant_results
        if item.status == "success"
    ]
    failures = [
        {
            "model": item.routing_key,
            "provider": item.provider,
            "error": item.error or "",
        }
        for item in participant_results
        if item.status == "error"
    ]
    original_messages = [message.model_dump() for message in request.messages]
    task = (
        "Select the single best candidate answer."
        if strategy == "select"
        else "Synthesize a single best final answer from the candidates."
    )
    user_content = (
        f"Ensemble strategy: {strategy}\n"
        f"Task: {task}\n\n"
        f"Original conversation:\n{json.dumps(original_messages, ensure_ascii=False, indent=2)}\n\n"
        f"Successful candidate responses:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
        f"Failed candidate responses:\n{json.dumps(failures, ensure_ascii=False, indent=2)}\n\n"
        "Return only the final answer for the user."
    )
    return [
        ChatMessage(
            role="system",
            content=judge_system_prompt or _default_judge_system_prompt(strategy),
        ),
        ChatMessage(role="user", content=user_content),
    ]


async def _execute_single_member(
    router_engine,
    request: ChatRequest,
    routing_key: str,
) -> dict:
    provider_name, model_config = router_engine.model_map[routing_key]
    provider = router_engine.providers[provider_name]
    member_request = request.model_copy(
        update={
            "model": routing_key,
            "stream": False,
            "cascade": None,
            "ensemble": None,
        }
    )
    try:
        response = await provider.chat_completions(member_request, model_config)
        usage = response.usage
        participant = EnsembleParticipantResult(
            routing_key=routing_key,
            model_name=model_config.name,
            provider=provider_name,
            status="success",
            content=_choice_content(response),
            finish_reason=response.choices[0].finish_reason if response.choices else None,
        )
        return {
            "participant": participant,
            "response": response,
            "usage": usage,
            "cost": _calc_cost(model_config.pricing, usage),
        }
    except Exception as exc:
        participant = EnsembleParticipantResult(
            routing_key=routing_key,
            model_name=model_config.name,
            provider=provider_name,
            status="error",
            error=str(exc),
        )
        return {
            "participant": participant,
            "error": exc,
            "usage": None,
            "cost": 0.0,
        }


async def execute_ensemble(request: ChatRequest, router_engine, allowed_models: set[str] | None = None):
    config = validate_ensemble_request(request)
    executed_models = config["participants"] + [config["judge_model"]]

    if allowed_models is not None:
        for model in executed_models:
            if model not in allowed_models:
                raise ModelAccessError(model)

    for model in executed_models:
        await router_engine.route(model)

    participant_runs = await asyncio.gather(
        *[_execute_single_member(router_engine, request, routing_key) for routing_key in config["participants"]]
    )

    participant_results = [item["participant"] for item in participant_runs]
    successful_runs = [item for item in participant_runs if item["participant"].status == "success"]

    if not successful_runs:
        sample_errors = [item["participant"].error for item in participant_runs if item["participant"].error]
        raise ProviderError("ensemble", 502, f"all participant models failed: {sample_errors[:3]}")

    judge_messages = _build_judge_messages(
        request,
        participant_results,
        config["strategy"],
        config["judge_system_prompt"],
    )
    judge_request = request.model_copy(
        update={
            "model": config["judge_model"],
            "messages": judge_messages,
            "stream": False,
            "cascade": None,
            "ensemble": None,
            "temperature": 0.0 if config["strategy"] == "select" else 0.2,
        }
    )
    judge_provider_name, judge_model_config = router_engine.model_map[config["judge_model"]]
    judge_provider = router_engine.providers[judge_provider_name]
    judge_response = await judge_provider.chat_completions(judge_request, judge_model_config)

    aggregate_usage = _merge_usage(
        [item.get("usage") for item in successful_runs] + [judge_response.usage]
    )
    total_cost = sum(item.get("cost", 0.0) for item in successful_runs) + _calc_cost(
        judge_model_config.pricing,
        judge_response.usage,
    )
    ensemble_meta = EnsembleMetadata(
        enabled=True,
        strategy=config["strategy"],
        requested_model=request.model,
        judge_model=config["judge_model"],
        final_model=config["judge_model"],
        participant_models=config["participants"],
        successful_models=[item["participant"].routing_key for item in successful_runs],
        failed_models=[
            item["participant"].routing_key
            for item in participant_runs
            if item["participant"].status == "error"
        ],
        participant_results=participant_results,
    )

    judge_response.model = request.model
    judge_response.actual_model = (
        config["judge_model"] if config["judge_model"] != request.model else None
    )
    judge_response.usage = aggregate_usage
    judge_response.ensemble = ensemble_meta

    return judge_response, total_cost, judge_model_config
