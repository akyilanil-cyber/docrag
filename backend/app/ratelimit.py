import time

from fastapi import HTTPException, Request

from .config import get_settings


class InMemoryLimiter:
    """Fixed-window counter. Used when Redis is not configured (tests)."""

    def __init__(self) -> None:
        self.hits: dict[str, tuple[int, int]] = {}

    def hit(self, key: str, window: int) -> int:
        bucket = int(time.time() // window)
        old_bucket, count = self.hits.get(key, (bucket, 0))
        count = count + 1 if old_bucket == bucket else 1
        self.hits[key] = (bucket, count)
        return count


class RedisLimiter:
    def __init__(self, url: str) -> None:
        import redis

        self.client = redis.Redis.from_url(url, socket_timeout=1)

    def hit(self, key: str, window: int) -> int:
        bucket = int(time.time() // window)
        redis_key = f"rl:{key}:{bucket}"
        pipe = self.client.pipeline()
        pipe.incr(redis_key)
        pipe.expire(redis_key, window * 2)
        count, _ = pipe.execute()
        return int(count)


_limiter = None


def get_limiter():
    global _limiter
    if _limiter is None:
        _limiter = RedisLimiter(get_settings().redis_url)
    return _limiter


def set_limiter(limiter) -> None:
    global _limiter
    _limiter = limiter


def rate_limit(request: Request) -> None:
    """Dependency: caps expensive endpoints (LLM calls, uploads) per client IP."""
    settings = get_settings()
    if not settings.rate_limit_enabled:
        return
    client = request.client.host if request.client else "unknown"
    try:
        count = get_limiter().hit(client, 60)
    except Exception:
        # Redis down: fail open for availability, the LLM token cap still bounds spend.
        return
    if count > settings.rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="Too many requests, slow down")
