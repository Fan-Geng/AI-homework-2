import time
from abc import ABC, abstractmethod
from collections import defaultdict
from statistics import mean

from app.core.config import ProviderConfig, ApiKeyConfig, ModelConfig
from app.core.exceptions import ProviderError
from app.core.model_availability import is_model_unavailable_error
from app.schemas.chat import ChatRequest, ChatResponse


class KeyPool:
    def __init__(self, keys: list[ApiKeyConfig], strategy: str = "round_robin"):
        self.keys = list(keys)
        self.strategy = strategy
        self._index = 0
        self._usage: dict[int, dict] = defaultdict(lambda: {
            "rpm_count": 0,
            "rpm_reset": time.time(),
            "tpm_count": 0,
            "tpm_reset": time.time(),
            "cooldown_until": 0,
        })

    def _stats(self, idx: int) -> dict:
        return self._usage[idx]

    def _is_available(self, idx: int, key: ApiKeyConfig) -> bool:
        stats = self._stats(idx)
        now = time.time()

        if now < stats["cooldown_until"]:
            return False

        if now - stats["rpm_reset"] >= 60:
            stats["rpm_count"] = 0
            stats["rpm_reset"] = now

        if key.rpm and stats["rpm_count"] >= key.rpm:
            return False

        if now - stats["tpm_reset"] >= 60:
            stats["tpm_count"] = 0
            stats["tpm_reset"] = now

        if key.tpm and stats["tpm_count"] >= key.tpm:
            return False

        return True

    def acquire(self) -> tuple[ApiKeyConfig, int] | None:
        n = len(self.keys)
        for _ in range(n):
            idx = self._index % n
            self._index = (self._index + 1) % n
            key = self.keys[idx]
            if self._is_available(idx, key):
                self._stats(idx)["rpm_count"] += 1
                return key, idx
        return None

    def mark_failed(self, idx: int):
        self._stats(idx)["cooldown_until"] = time.time() + 15

    def release(self, idx: int, tokens: int = 0):
        if tokens:
            self._stats(idx)["tpm_count"] += tokens

    def key_tag(self, key: ApiKeyConfig) -> str:
        return key.key[-4:]


class ProviderHealth:
    def __init__(self, circuit_failures: int = 5, circuit_cooldown_s: int = 30):
        self.circuit_failures = circuit_failures
        self.circuit_cooldown_s = circuit_cooldown_s
        self.total_requests = 0
        self.successes = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.last_error = None
        self.last_status_code = None
        self.last_success_at = None
        self.last_failure_at = None
        self.circuit_open_until = 0.0
        self._recent_latency_ms: list[float] = []

    def is_circuit_open(self) -> bool:
        return time.time() < self.circuit_open_until

    def record_success(self, latency_ms: float | None = None):
        self.total_requests += 1
        self.successes += 1
        self.consecutive_failures = 0
        self.last_error = None
        self.last_status_code = 200
        self.last_success_at = int(time.time())
        self.circuit_open_until = 0.0
        if latency_ms is not None:
            self._recent_latency_ms.append(round(latency_ms, 1))
            self._recent_latency_ms = self._recent_latency_ms[-20:]

    def record_failure(self, error: Exception | str, status_code: int | None = None):
        self.total_requests += 1
        self.failures += 1
        self.consecutive_failures += 1
        self.last_error = str(error)
        self.last_status_code = status_code
        self.last_failure_at = int(time.time())
        if self.consecutive_failures >= self.circuit_failures:
            self.circuit_open_until = time.time() + self.circuit_cooldown_s

    def snapshot(self) -> dict:
        avg_latency_ms = mean(self._recent_latency_ms) if self._recent_latency_ms else 0.0
        success_rate = (self.successes / self.total_requests * 100.0) if self.total_requests else 0.0
        return {
            "total_requests": self.total_requests,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": round(success_rate, 1),
            "consecutive_failures": self.consecutive_failures,
            "avg_latency_ms": round(avg_latency_ms, 1),
            "last_status_code": self.last_status_code,
            "last_error": self.last_error,
            "last_success_at": self.last_success_at,
            "last_failure_at": self.last_failure_at,
            "circuit_open": self.is_circuit_open(),
            "circuit_open_until": int(self.circuit_open_until) if self.circuit_open_until else 0,
            "circuit_failures": self.circuit_failures,
            "circuit_cooldown_s": self.circuit_cooldown_s,
        }


class BaseProvider(ABC):
    def __init__(self, config: ProviderConfig, key_pool: KeyPool):
        self.config = config
        self.key_pool = key_pool
        self.health = ProviderHealth()

    def raise_if_unhealthy(self):
        if self.health.is_circuit_open():
            raise ProviderError(
                self.__class__.__name__,
                503,
                f"provider circuit open until {int(self.health.circuit_open_until)}",
            )

    def record_success(self, started_at: float):
        latency_ms = (time.monotonic() - started_at) * 1000
        self.health.record_success(latency_ms)

    def record_failure(self, error: Exception, status_code: int | None = None):
        code = status_code
        if code is None and isinstance(error, ProviderError):
            code = error.status_code
        self.health.record_failure(error, code)

    def mark_key_failed(self, key_index: int, error: Exception):
        # A missing model channel must not cool down a shared provider key.
        if not is_model_unavailable_error(error):
            self.key_pool.mark_failed(key_index)

    def record_provider_failure(self, error: Exception):
        # Model-level outages should not lower the health score of every model
        # served by the provider.
        if not is_model_unavailable_error(error):
            self.record_failure(error)

    @abstractmethod
    async def chat_completions(
        self, request: ChatRequest, model_config: ModelConfig
    ) -> ChatResponse:
        ...

    @abstractmethod
    async def chat_completions_stream(
        self, request: ChatRequest, model_config: ModelConfig
    ):
        ...
