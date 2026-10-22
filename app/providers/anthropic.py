import json
import time
import httpx

from app.core.config import ProviderConfig, ModelConfig
from app.core.exceptions import ProviderError, AllKeysFailedError
from app.providers.base import BaseProvider, KeyPool
from app.schemas.chat import (
    ChatRequest, ChatResponse, ChatChoice, ChatMessage,
    StreamChunk, Delta, StreamChoice, Usage,
)


class AnthropicProvider(BaseProvider):
    def __init__(self, config: ProviderConfig, key_pool: KeyPool):
        super().__init__(config, key_pool)
        self.client = httpx.AsyncClient(timeout=None if config.timeout <= 0 else config.timeout)

    def _headers(self, api_key: str) -> dict:
        return {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _build_request(self, request: ChatRequest, model_config: ModelConfig) -> dict:
        system = None
        messages = []
        for msg in request.messages:
            if msg.role == "system":
                system = msg.content
            else:
                messages.append({"role": msg.role, "content": msg.content})

        data = {
            "model": model_config.name,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
        }
        if system:
            data["system"] = system
        # The gateway currently rejects temperature for Claude variants even
        # when the value is 0. Omit the field and use the model default.
        if request.top_p is not None:
            data["top_p"] = request.top_p
        if request.stop is not None:
            data["stop_sequences"] = [request.stop] if isinstance(request.stop, str) else request.stop
        return data

    def _convert_stop(self, reason: str | None) -> str | None:
        mapping = {"end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop"}
        return mapping.get(reason, reason)

    def _parse_response(self, body: dict, model: str) -> ChatResponse:
        content = body.get("content", [])
        text = "".join(block["text"] for block in content if block.get("type") == "text")
        usage_data = body.get("usage", {})
        return ChatResponse(
            id=body["id"],
            created=int(time.time()),
            model=model,
            choices=[
                ChatChoice(
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason=self._convert_stop(body.get("stop_reason")),
                )
            ],
            usage=Usage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
                total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
            ) if usage_data else None,
        )

    def _decode_json_response(self, resp: httpx.Response) -> dict:
        try:
            return resp.json()
        except json.JSONDecodeError:
            content_type = resp.headers.get("content-type", "")
            text = resp.text[:300]
            raise ProviderError("anthropic", resp.status_code, f"non-json upstream response ({content_type}): {text}")

    async def chat_completions(
        self, request: ChatRequest, model_config: ModelConfig
    ) -> ChatResponse:
        self.raise_if_unhealthy()
        started_at = time.monotonic()
        payload = self._build_request(request, model_config)
        payload["stream"] = False

        last_error = None
        for _ in range(len(self.key_pool.keys)):
            acquired = self.key_pool.acquire()
            if not acquired:
                continue
            key, kidx = acquired

            try:
                resp = await self._try_send(payload, key.key)
                body = self._decode_json_response(resp)
                result = self._parse_response(body, model_config.name)
                tokens = result.usage.total_tokens if result.usage else 0
                self.key_pool.release(kidx, tokens)
                self.record_success(started_at)
                return result
            except ProviderError as e:
                last_error = e
                self.mark_key_failed(kidx, e)
            except httpx.RequestError as e:
                last_error = ProviderError("anthropic", 0, f"request error: {e}")
                self.mark_key_failed(kidx, last_error)

        if last_error:
            self.record_provider_failure(last_error)
        raise last_error or AllKeysFailedError("anthropic")

    async def chat_completions_stream(
        self, request: ChatRequest, model_config: ModelConfig
    ):
        self.raise_if_unhealthy()
        started_at = time.monotonic()
        payload = self._build_request(request, model_config)
        payload["stream"] = True

        last_error = None
        for _ in range(len(self.key_pool.keys)):
            acquired = self.key_pool.acquire()
            if not acquired:
                continue
            key, kidx = acquired

            try:
                async with self.client.stream(
                    "POST", f"{self.config.base_url}/messages",
                    json=payload,
                    headers=self._headers(key.key),
                ) as resp:
                    if resp.status_code != 200:
                        text = await resp.aread()
                        raise ProviderError("anthropic", resp.status_code, text.decode())

                    chunk_id = ""
                    has_started = False
                    input_tokens = 0
                    output_tokens = 0

                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("event: "):
                            event = line[7:]
                        elif line.startswith("data: "):
                            data = json.loads(line[6:])

                            if event == "message_start":
                                msg = data.get("message", {})
                                chunk_id = msg.get("id", "")
                                input_tokens = msg.get("usage", {}).get("input_tokens", 0)
                                has_started = True
                                yield StreamChunk(
                                    id=chunk_id,
                                    created=int(time.time()),
                                    model=model_config.name,
                                    choices=[StreamChoice(delta=Delta(role="assistant"))],
                                )

                            elif event == "content_block_delta":
                                delta = data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    yield StreamChunk(
                                        id=chunk_id,
                                        created=int(time.time()),
                                        model=model_config.name,
                                        choices=[StreamChoice(delta=Delta(content=delta.get("text", "")))],
                                    )

                            elif event == "message_delta":
                                delta = data.get("delta", {})
                                output_tokens = data.get("usage", {}).get("output_tokens", 0)
                                yield StreamChunk(
                                    id=chunk_id,
                                    created=int(time.time()),
                                    model=model_config.name,
                                    choices=[StreamChoice(
                                        delta=Delta(),
                                        finish_reason=self._convert_stop(delta.get("stop_reason")),
                                    )],
                                    usage=Usage(
                                        prompt_tokens=input_tokens,
                                        completion_tokens=output_tokens,
                                        total_tokens=input_tokens + output_tokens,
                                    ),
                                )

                    self.key_pool.release(kidx, input_tokens + output_tokens)
                    self.record_success(started_at)
                    return
            except ProviderError as e:
                last_error = e
                self.mark_key_failed(kidx, e)

        if last_error:
            self.record_provider_failure(last_error)
        raise last_error or AllKeysFailedError("anthropic")

    async def _try_send(self, payload: dict, api_key: str) -> httpx.Response:
        headers = self._headers(api_key)
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = await self.client.post(
                    f"{self.config.base_url}/messages",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code in (429,) or resp.status_code >= 500:
                    if attempt < self.config.max_retries:
                        continue
                    raise ProviderError("anthropic", resp.status_code, resp.text[:200])
                resp.raise_for_status()
                return resp
            except httpx.TimeoutException:
                if attempt < self.config.max_retries:
                    continue
                raise ProviderError("anthropic", 0, "timeout")
            except httpx.HTTPStatusError as e:
                raise ProviderError("anthropic", e.response.status_code, e.response.text[:200])
            except httpx.RequestError as e:
                if attempt < self.config.max_retries:
                    continue
                raise ProviderError("anthropic", 0, f"request error: {e}")
        raise ProviderError("anthropic", 0, "max retries exceeded")
