from pydantic import BaseModel, Field
from typing import Literal, Optional


class CascadeConfig(BaseModel):
    enabled: bool | None = None
    chain: list[str] | None = None
    max_escalations: int | None = None
    min_content_length: int | None = None
    judge_model: str | None = None
    judge_threshold: float | None = None
    judge_system_prompt: str | None = None


class EnsembleConfig(BaseModel):
    enabled: bool | None = None
    participants: list[str] | None = None
    judge_model: str | None = None
    strategy: Literal["synthesize", "select"] = "synthesize"
    judge_system_prompt: str | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None
    cascade: CascadeConfig | None = None
    ensemble: EnsembleConfig | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class CascadeMetadata(BaseModel):
    enabled: bool = False
    level: int = 0
    escalated_from: str | None = None
    final_model: str | None = None
    validation_passed: bool | None = None
    escalation_reason: str | None = None
    models_attempted: list[str] | None = None
    judge_model: str | None = None
    judge_score: float | None = None
    judge_reason: str | None = None


class RoutingRetryMetadata(BaseModel):
    retry_count: int = 0
    attempted_models: list[str] = Field(default_factory=list)
    reason: str | None = None


class EnsembleParticipantResult(BaseModel):
    routing_key: str
    model_name: str
    provider: str
    status: Literal["success", "error"]
    content: str | None = None
    finish_reason: str | None = None
    error: str | None = None


class EnsembleMetadata(BaseModel):
    enabled: bool = False
    strategy: Literal["synthesize", "select"]
    requested_model: str
    judge_model: str
    final_model: str
    participant_models: list[str]
    successful_models: list[str]
    failed_models: list[str]
    participant_results: list[EnsembleParticipantResult]


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    actual_model: str | None = None
    intent_category: str | None = None
    choices: list[ChatChoice]
    usage: Usage | None = None
    cascade: CascadeMetadata | None = None
    routing_retry: RoutingRetryMetadata | None = None
    ensemble: EnsembleMetadata | None = None


class Delta(BaseModel):
    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: Delta
    finish_reason: str | None = None


class StreamChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    actual_model: str | None = None
    intent_category: str | None = None
    choices: list[StreamChoice]
    usage: Usage | None = None
    cascade: CascadeMetadata | None = None
    ensemble: EnsembleMetadata | None = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]
