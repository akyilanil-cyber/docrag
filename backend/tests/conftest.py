import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="docrag-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/test.db"
os.environ["UPLOAD_DIR"] = f"{_TMP}/uploads"
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import ratelimit  # noqa: E402
from app.celery_app import celery_app  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import Base, get_engine  # noqa: E402
from app.main import app  # noqa: E402

celery_app.conf.task_always_eager = True  # run tasks inline, no Redis needed


@pytest.fixture()
def client():
    get_settings.cache_clear()
    ratelimit.set_limiter(ratelimit.InMemoryLimiter())
    engine = get_engine()
    Base.metadata.drop_all(engine)
    with TestClient(app) as c:  # lifespan creates the tables
        yield c
