import time


class FixedWindowRateLimiter:
    def __init__(self):
        self._windows: dict[int, dict[str, float | int]] = {}

    def check_and_increment(self, key_id: int, limit: int) -> bool:
        if limit <= 0:
            return True

        now = time.time()
        window = self._windows.get(key_id)
        if not window or now - float(window["started_at"]) >= 60:
            self._windows[key_id] = {"count": 1, "started_at": now}
            return True

        if int(window["count"]) >= limit:
            return False

        window["count"] = int(window["count"]) + 1
        return True
