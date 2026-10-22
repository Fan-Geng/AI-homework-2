from pydantic import BaseModel
from typing import Optional
from app.schemas.chat import ChatMessage, CascadeConfig


class ProviderCreate(BaseModel):
    name: str
    base_url: str
    health_url: str = ""
    timeout: int = 0
    max_retries: int = 3
    key_strategy: str = "round_robin"


class ProviderUpdate(BaseModel):
    base_url: Optional[str] = None
    health_url: Optional[str] = None
    timeout: Optional[int] = None
    max_retries: Optional[int] = None
    key_strategy: Optional[str] = None
    enabled: Optional[bool] = None


class ProviderKeyCreate(BaseModel):
    key_value: str
    rpm: int = 60
    tpm: Optional[int] = None
    weight: int = 1
    note: str = ""


class ProviderModelSyncRequest(BaseModel):
    timeout: int = 20


class ProviderHealthCheckRequest(BaseModel):
    timeout: int = 10
    limit: int = 10


class RoutePreviewRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    cascade: CascadeConfig | None = None
    allowed_models: Optional[str] = None
    api_key: Optional[str] = None
    user_id: Optional[str] = None
    routing_profile_id: Optional[str] = None


class ModelCreate(BaseModel):
    provider_name: str
    model_name: str
    routing_key: str
    prompt_price: float = 0
    completion_price: float = 0


class ModelUpdate(BaseModel):
    model_name: Optional[str] = None
    routing_key: Optional[str] = None
    prompt_price: Optional[float] = None
    completion_price: Optional[float] = None
    enabled: Optional[bool] = None


class ApiKeyCreate(BaseModel):
    note: str = ""
    rate_limit_rpm: int = 60
    allowed_models: str = "*"


class ApiKeyResponse(BaseModel):
    id: int
    key_value: str
    note: str
    rate_limit_rpm: int
    allowed_models: str
    enabled: bool
    created_at: str


class ApiKeyListResponse(BaseModel):
    id: int
    note: str
    rate_limit_rpm: int
    allowed_models: str
    enabled: bool
    created_at: str
    key_suffix: str


class RoutingProfileCreate(BaseModel):
    name: str
    description: str = ""
    config_json: str = "{}"


class RoutingProfileUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config_json: Optional[str] = None
    enabled: Optional[bool] = None


class RoutingProfileBindingUpdate(BaseModel):
    routing_profile_id: Optional[int] = None


class UserRoutingProfileBindingUpdate(BaseModel):
    routing_profile_id: Optional[int] = None


class UsageQueryParams(BaseModel):
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    model: Optional[str] = None
    status: Optional[str] = None
    limit: int = 100
    offset: int = 0


class KeywordRuleCreate(BaseModel):
    routing_profile_id: Optional[int] = None
    keyword: str
    match_mode: str = "contains"
    target_routing_key: str
    priority: int = 0
    note: str = ""


class KeywordRuleUpdate(BaseModel):
    routing_profile_id: Optional[int] = None
    keyword: Optional[str] = None
    match_mode: Optional[str] = None
    target_routing_key: Optional[str] = None
    priority: Optional[int] = None
    enabled: Optional[bool] = None
    note: Optional[str] = None


class SettingsUpdate(BaseModel):
    keyword_routing_enabled: Optional[bool] = None
    semantic_routing_enabled: Optional[bool] = None
    cascade_enabled: Optional[bool] = None


class CascadeRuleCreate(BaseModel):
    routing_profile_id: Optional[int] = None
    name: str
    source_routing_key: str
    escalation_chain: list[str]
    min_content_length: int = 10
    check_refusal: bool = True
    refusal_patterns: list[str] = []
    check_finish_reason: bool = True
    local_judge_model: Optional[str] = None
    local_judge_threshold: float = 0.7
    local_judge_system_prompt: Optional[str] = None
    custom_validator_url: Optional[str] = None
    custom_validator_timeout: int = 5000
    max_escalations: int = 3
    cost_threshold: float = 0.0
    priority: int = 0
    note: str = ""


class CascadeRuleUpdate(BaseModel):
    routing_profile_id: Optional[int] = None
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


class CascadeStatsQuery(BaseModel):
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    model: Optional[str] = None
    limit: int = 100
    offset: int = 0
