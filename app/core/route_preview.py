from app.core.exceptions import ModelAccessError, ModelNotFoundError
from app.core.routing_policy import DEFAULT_MODEL_ALIASES, RoutingPolicy


def parse_allowed_models(raw: str | None) -> set[str] | None:
    if not raw or raw.strip() == "*":
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


async def build_route_preview(
    body,
    router_engine,
    cascade_engine,
    *,
    routing_policy: RoutingPolicy | None = None,
    routing_profile_id: str | None = None,
    keyword_routing_enabled: bool = False,
    semantic_routing_enabled: bool = False,
    embedder=None,
    recommendation_enabled: bool = False,
    recommender=None,
    cascade_enabled: bool = False,
    allowed_models: set[str] | None = None,
) -> dict:
    routing_policy = routing_policy or RoutingPolicy()
    requested_model = body.model
    effective_model = requested_model
    intent_category = None
    route_reason = "direct"
    decision_trace = [
        {
            "step": "requested",
            "model": requested_model,
            "available": requested_model in router_engine.model_map,
        }
    ]

    try:
        resolved_requested_model = router_engine.resolve_model_target(
            requested_model,
            default_model=routing_policy.default_model,
            preferred_providers=routing_policy.preferred_providers,
            cost_bias=routing_policy.cost_bias_value,
            latency_bias=routing_policy.latency_bias_value,
            quality_bias=routing_policy.quality_bias_value,
        )
        if resolved_requested_model != requested_model:
            effective_model = resolved_requested_model
            route_reason = "profile_policy"
            decision_trace.append(
                {
                    "step": "profile_policy",
                    "requested_model": requested_model,
                    "resolved_model": resolved_requested_model,
                    "default_model": routing_policy.default_model,
                    "preferred_providers": routing_policy.preferred_providers,
                }
            )
    except ModelNotFoundError as exc:
        resolved_requested_model = requested_model
        decision_trace.append(
            {
                "step": "profile_policy",
                "error": str(exc),
                "default_model": routing_policy.default_model,
                "preferred_providers": routing_policy.preferred_providers,
            }
        )
    else:
        requested_model = resolved_requested_model

    recommendation_detail = {
        "enabled": recommendation_enabled,
        "applied": False,
        "selected_model": None,
        "route_reason": None,
        "confidence": None,
        "task_profile": None,
        "candidates": [],
        "excluded": [],
        "objective": None,
    }
    recommendation_requested = (
        requested_model in DEFAULT_MODEL_ALIASES
        or routing_policy.recommendation_apply_to_explicit
    )
    if recommendation_enabled and recommendation_requested and recommender:
        recommendation = recommender.recommend(
            messages=body.messages,
            router_engine=router_engine,
            policy=routing_policy,
            allowed_models=allowed_models,
        )
        recommendation_detail.update(recommendation.to_dict())
        decision_trace.append(
            {
                "step": "recommendation",
                "applied": recommendation.applied,
                "selected_model": recommendation.selected_model,
                "confidence": round(recommendation.confidence, 4),
                "task": recommendation.task_profile.get("name"),
            }
        )
        if recommendation.applied and recommendation.selected_model:
            effective_model = recommendation.selected_model
            route_reason = "recommendation"
    elif recommendation_enabled:
        reason = "explicit_model_not_recommendation_target"
        if recommendation_requested:
            reason = "recommender_unavailable"
        recommendation_detail["route_reason"] = reason
        decision_trace.append(
            {
                "step": "recommendation",
                "applied": False,
                "reason": reason,
            }
        )

    keyword_detail = {
        "enabled": keyword_routing_enabled,
        "matched": False,
        "applied": False,
        "target_model": None,
        "rule": None,
    }
    if keyword_routing_enabled and not recommendation_detail["applied"]:
        match = await router_engine.inspect_keyword(
            body.messages,
            routing_profile_id=routing_profile_id,
            only_available=True,
        )
        if match:
            keyword_detail.update(
                {
                    "matched": True,
                    "applied": match["target_routing_key"] in router_engine.model_map,
                    "target_model": match["target_routing_key"],
                    "rule": match,
                }
            )
            decision_trace.append(
                {
                    "step": "keyword",
                    "matched": True,
                    "target_model": match["target_routing_key"],
                    "match_mode": match["match_mode"],
                    "keyword": match["keyword"],
                }
            )
            if keyword_detail["applied"]:
                effective_model = match["target_routing_key"]
                route_reason = "keyword_rule"
        else:
            decision_trace.append({"step": "keyword", "matched": False})
    elif keyword_routing_enabled:
        decision_trace.append(
            {
                "step": "keyword",
                "skipped": True,
                "reason": "recommendation_already_applied",
            }
        )

    semantic_detail = {
        "enabled": semantic_routing_enabled,
        "considered": effective_model == requested_model,
        "matched": False,
        "target_model": None,
        "category": None,
        "score": None,
        "applied": False,
    }
    if effective_model != requested_model:
        decision_trace.append(
            {
                "step": "semantic",
                "skipped": True,
                "reason": f"{route_reason}_already_matched",
            }
        )
    elif semantic_routing_enabled:
        match = await router_engine.inspect_semantic(body.messages, embedder)
        if match:
            semantic_detail.update(
                {
                    "matched": True,
                    "target_model": match["target_routing_key"],
                    "category": match["category"],
                    "score": round(match["score"], 4),
                    "applied": match["target_routing_key"] in router_engine.model_map,
                }
            )
            decision_trace.append(
                {
                    "step": "semantic",
                    "matched": True,
                    "category": match["category"],
                    "score": round(match["score"], 4),
                    "target_model": match["target_routing_key"],
                }
            )
            if semantic_detail["applied"]:
                effective_model = match["target_routing_key"]
                intent_category = match["category"]
                route_reason = "semantic_routing"
        else:
            decision_trace.append({"step": "semantic", "matched": False})

    resolved = None
    route_error = None
    try:
        provider, model_config = await router_engine.route(effective_model)
        provider_name = getattr(provider, "provider_name", None)
        if not provider_name:
            provider_name = router_engine.model_map.get(effective_model, ("",))[0]
        resolved = {
            "provider": provider_name,
            "model_name": model_config.name,
            "routing_key": effective_model,
        }
        decision_trace.append(
            {
                "step": "resolved",
                "provider": resolved["provider"],
                "model_name": resolved["model_name"],
                "routing_key": effective_model,
            }
        )
    except ModelNotFoundError as exc:
        route_error = str(exc)
        decision_trace.append(
            {
                "step": "resolved",
                "error": str(exc),
                "available_models": exc.available,
            }
        )

    access = {
        "allowed_models": sorted(allowed_models) if allowed_models is not None else None,
        "requested_allowed": True,
        "effective_allowed": True,
        "blocked": False,
        "error": None,
    }
    if allowed_models is not None:
        requested_is_alias = requested_model in DEFAULT_MODEL_ALIASES
        access["requested_allowed"] = True if requested_is_alias else requested_model in allowed_models
        access["effective_allowed"] = effective_model in allowed_models
        if not access["requested_allowed"]:
            exc = ModelAccessError(requested_model)
            access["blocked"] = True
            access["error"] = str(exc)
            route_error = str(exc)
            decision_trace.append(
                {
                    "step": "access",
                    "blocked": True,
                    "model": requested_model,
                    "reason": "requested_model_not_allowed",
                }
            )
        elif not access["effective_allowed"]:
            exc = ModelAccessError(effective_model)
            access["blocked"] = True
            access["error"] = str(exc)
            route_error = str(exc)
            decision_trace.append(
                {
                    "step": "access",
                    "blocked": True,
                    "model": effective_model,
                    "reason": "effective_model_not_allowed",
                }
            )
        else:
            decision_trace.append({"step": "access", "blocked": False})

    cascade_config = None
    if body.cascade:
        cascade_config = body.cascade.model_dump() if hasattr(body.cascade, "model_dump") else body.cascade
    elif cascade_enabled:
        cascade_config = {"enabled": True}

    cascade_detail = {
        "enabled": bool(cascade_config and cascade_config.get("enabled") is not False),
        "chain": [],
        "max_escalations": 0,
        "validator": None,
        "local_judge": {"enabled": False},
        "webhook_validator": None,
    }
    if cascade_detail["enabled"]:
        chain = await cascade_engine.resolve_chain(
            effective_model,
            cascade_config,
            routing_profile_id=routing_profile_id,
        )
        if chain and allowed_models is not None:
            chain = [entry for entry in chain if entry[0] in allowed_models]
        if chain:
            validator = cascade_engine.get_validator(
                effective_model,
                cascade_config,
                routing_profile_id=routing_profile_id,
            )
            local_judge = cascade_engine.get_local_judge(
                effective_model,
                cascade_config,
                routing_profile_id=routing_profile_id,
            )
            webhook_validator = cascade_engine.get_webhook_validator(
                effective_model,
                routing_profile_id=routing_profile_id,
            )
            cascade_detail.update(
                {
                    "chain": [routing_key for routing_key, _, _ in chain],
                    "max_escalations": cascade_engine.get_max_escalations(
                        effective_model,
                        cascade_config,
                        routing_profile_id=routing_profile_id,
                    ),
                    "validator": {
                        "min_content_length": validator.min_content_length,
                        "check_refusal": validator.check_refusal,
                        "refusal_patterns": validator.refusal_patterns,
                        "check_finish_reason": validator.check_finish_reason,
                    },
                    "local_judge": (
                        {
                            "enabled": True,
                            "model": local_judge.judge_model,
                            "threshold": local_judge.threshold,
                            "system_prompt_configured": bool(local_judge.system_prompt),
                        }
                        if local_judge
                        else {"enabled": False}
                    ),
                    "webhook_validator": (
                        {
                            "enabled": True,
                            "url": webhook_validator.url,
                            "timeout_s": webhook_validator.timeout,
                        }
                        if webhook_validator
                        else {"enabled": False}
                    ),
                }
            )
        decision_trace.append(
            {
                "step": "cascade",
                "enabled": cascade_detail["enabled"],
                "chain": cascade_detail["chain"],
                "max_escalations": cascade_detail["max_escalations"],
            }
        )
    else:
        decision_trace.append({"step": "cascade", "enabled": False})

    return {
        "ok": route_error is None,
        "requested_model": requested_model,
        "effective_model": effective_model,
        "actual_model": effective_model if effective_model != requested_model else None,
        "intent_category": intent_category,
        "route_reason": route_reason,
        "resolved": resolved,
        "recommendation": recommendation_detail,
        "keyword_routing": keyword_detail,
        "semantic_routing": semantic_detail,
        "cascade": cascade_detail,
        "access": access,
        "error": route_error,
        "decision_trace": decision_trace,
    }
