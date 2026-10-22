import logging
import httpx
from app.schemas.chat import ChatResponse, ChatRequest

logger = logging.getLogger("model_router.cascade.webhook")


class WebhookValidator:
    def __init__(self, url: str, timeout: int = 5000):
        self.url = url
        self.timeout = timeout / 1000

    async def validate(self, response: ChatResponse, request: ChatRequest) -> tuple[bool, str]:
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "request": {
                        "model": request.model,
                        "messages": [m.model_dump() for m in request.messages],
                    },
                    "response": {
                        "id": response.id,
                        "model": response.model,
                        "choices": [
                            {
                                "message": {
                                    "role": c.message.role,
                                    "content": c.message.content,
                                },
                                "finish_reason": c.finish_reason,
                            }
                            for c in response.choices
                        ],
                        "usage": response.usage.model_dump() if response.usage else None,
                    },
                }
                resp = await client.post(
                    self.url,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                result = resp.json()
                return result.get("valid", True), result.get("reason", "webhook_pass")
        except httpx.TimeoutException:
            logger.warning("Webhook validation timed out for %s", self.url)
            return True, "webhook_timeout"
        except Exception as e:
            logger.warning("Webhook validation failed: %s", e)
            return True, "webhook_error"
