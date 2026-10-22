class ModelRouterError(Exception):
    pass


class InvalidRequestError(ModelRouterError):
    def __init__(self, message: str):
        super().__init__(message)


class ModelNotFoundError(ModelRouterError):
    def __init__(self, model: str, available: list[str]):
        self.model = model
        self.available = available
        super().__init__(f"Model '{model}' not found. Available: {available}")


class ProviderError(ModelRouterError):
    def __init__(self, provider: str, status_code: int, message: str):
        self.provider = provider
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{provider}] {status_code}: {message}")


class AllKeysFailedError(ModelRouterError):
    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(f"[{provider}] All API keys failed")


class AuthError(ModelRouterError):
    def __init__(self, message: str = "Invalid API key"):
        super().__init__(message)


class ModelAccessError(ModelRouterError):
    def __init__(self, model: str):
        self.model = model
        super().__init__(f"API key is not allowed to access model '{model}'")


class RateLimitError(ModelRouterError):
    def __init__(self, limit: int):
        self.limit = limit
        super().__init__(f"API key rate limit exceeded ({limit} requests/minute)")
