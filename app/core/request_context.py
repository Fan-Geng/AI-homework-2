from dataclasses import dataclass


def parse_allowed_models(raw: str | None) -> set[str] | None:
    if not raw or raw.strip() == "*":
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


@dataclass(slots=True)
class RequestContext:
    subject_type: str
    subject_id: str
    api_key_id: int = 0
    user_id: str | None = None
    routing_profile_id: str | None = None
    request_id: str | None = None
    allowed_models: set[str] | None = None
    subject_note: str | None = None

    def log_fields(self) -> dict[str, str | int | None]:
        return {
            "api_key_id": self.api_key_id,
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "user_id": self.user_id,
            "routing_profile_id": self.routing_profile_id,
            "request_id": self.request_id,
        }
