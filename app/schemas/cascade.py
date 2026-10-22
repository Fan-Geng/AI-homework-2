from pydantic import BaseModel
from typing import Optional


class ValidatorConfig(BaseModel):
    min_content_length: int = 10
    check_refusal: bool = True
    refusal_patterns: list[str] = [
        r"(?i)i cannot|i am unable|as an ai",
        r"(?i)i('m| am) (sorry|afraid|not)",
    ]
    check_finish_reason: bool = True
    local_judge_model: str | None = None
    local_judge_threshold: float = 0.7
    local_judge_system_prompt: str | None = None
    custom_validator_url: str | None = None
    custom_validator_timeout: int = 5000


class CascadeRuleCreate(BaseModel):
    name: str
    source_routing_key: str
    escalation_chain: list[str]
    min_content_length: int = 10
    check_refusal: bool = True
    refusal_patterns: list[str] = []
    check_finish_reason: bool = True
    local_judge_model: str | None = None
    local_judge_threshold: float = 0.7
    local_judge_system_prompt: str | None = None
    custom_validator_url: str | None = None
    custom_validator_timeout: int = 5000
    max_escalations: int = 3
    cost_threshold: float = 0.0
    priority: int = 0
    note: str = ""


class CascadeRuleUpdate(BaseModel):
    name: Optional[str] = None
    source_routing_key: Optional[str] = None
    escalation_chain: Optional[list[str]] = None
    min_content_length: Optional[int] = None
    check_refusal: Optional[bool] = None
    refusal_patterns: Optional[list[str]] = None
    check_finish_reason: Optional[bool] = None
    local_judge_model: Optional[str] = None
    local_judge_threshold: Optional[float] = None
    local_judge_system_prompt: Optional[str] = None
    custom_validator_url: Optional[str] = None
    custom_validator_timeout: Optional[int] = None
    max_escalations: Optional[int] = None
    cost_threshold: Optional[float] = None
    priority: Optional[int] = None
    enabled: Optional[bool] = None
    note: Optional[str] = None


class CascadeRuleResponse(BaseModel):
    id: int
    name: str
    source_routing_key: str
    escalation_chain: str
    min_content_length: int
    check_refusal: bool
    refusal_patterns: str
    check_finish_reason: bool
    local_judge_model: str | None
    local_judge_threshold: float
    local_judge_system_prompt: str | None
    custom_validator_url: str | None
    custom_validator_timeout: int
    max_escalations: int
    cost_threshold: float
    priority: int
    enabled: bool
    note: str
    created_at: str


class CascadeStatsQuery(BaseModel):
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    model: Optional[str] = None
    limit: int = 100
    offset: int = 0


class CascadeStatsResponse(BaseModel):
    id: int
    timestamp: str
    api_key_id: int
    original_model: str
    models_attempted: str
    final_model: str
    total_escalations: int
    validation_reasons: str | None
    total_cost: float
    total_latency_ms: int
    status: str


class CascadeStatsSummary(BaseModel):
    total_requests: int
    total_escalations: int
    escalation_rate: float
    total_cost: float
    avg_latency_ms: float
    escalations_by_model: dict[str, int]


class CascadeConfig(BaseModel):
    enabled: bool | None = None
    chain: list[str] | None = None
    max_escalations: int | None = None
    min_content_length: int | None = None
    judge_model: str | None = None
    judge_threshold: float | None = None
    judge_system_prompt: str | None = None
