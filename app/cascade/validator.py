import re
import logging
from app.schemas.chat import ChatResponse

logger = logging.getLogger("model_router.cascade.validator")


class ResponseValidator:
    def __init__(
        self,
        min_content_length: int = 10,
        check_refusal: bool = True,
        refusal_patterns: list[str] | None = None,
        check_finish_reason: bool = True,
    ):
        self.min_content_length = min_content_length
        self.check_refusal = check_refusal
        self.refusal_patterns = refusal_patterns or [
            r"(?i)i cannot|i am unable|as an ai",
            r"(?i)i('m| am) (sorry|afraid|not)",
            r"(?i)i (will not|won't|can't|do not|don't) (help|assist|provide|share)",
            r"(?i)(against|violat(e|ing)) (my|the) (policy|guideline|rule)",
            r"(?i)i (must|have to) (decline|refuse|deny)",
        ]
        self.check_finish_reason = check_finish_reason
        self._compiled_patterns = [re.compile(p) for p in self.refusal_patterns]

    def validate(self, response: ChatResponse) -> tuple[bool, str]:
        if not response.choices:
            return False, "no_choices"

        choice = response.choices[0]
        content = choice.message.content or ""

        if not content.strip():
            return False, "empty_content"

        if len(content.strip()) < self.min_content_length:
            return False, "content_too_short"

        if self.check_finish_reason and choice.finish_reason:
            if choice.finish_reason in ("content_filter", "length", "error"):
                return False, f"bad_finish_reason:{choice.finish_reason}"

        if self.check_refusal:
            for pattern in self._compiled_patterns:
                if pattern.search(content):
                    return False, f"refusal_detected"

        return True, "valid"

    def validate_stream_content(self, content: str, finish_reason: str | None = None) -> tuple[bool, str]:
        if not content.strip():
            return False, "empty_content"

        if len(content.strip()) < self.min_content_length:
            return False, "content_too_short"

        if self.check_finish_reason and finish_reason:
            if finish_reason in ("content_filter", "length", "error"):
                return False, f"bad_finish_reason:{finish_reason}"

        if self.check_refusal:
            for pattern in self._compiled_patterns:
                if pattern.search(content):
                    return False, "refusal_detected"

        return True, "valid"
