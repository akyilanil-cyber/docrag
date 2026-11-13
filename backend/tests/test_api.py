import json
from unittest.mock import patch

from sqlalchemy import func, select

from app import qa, ratelimit
from app.config import get_settings
from app.db import get_session_factory
from app.models import Chunk, Document
from app.tasks import run_pipeline

DOC = (
    "Refund policy. Customers may return any item within 30 days for a full refund. "
    "Shipping costs are not refunded. "
    + "Filler sentence about unrelated warehouse logistics and pallets. " * 60
    + "Warranty. All electronic devices carry a two year warranty against manufacturing defects."
)


def upload(client, content: bytes, name="policy.txt"):
    return client.post("/documents", files={"file": (name, content, "text/plain")})


def test_upload_returns_immediately_and_processes(client):
    r = upload(client, DOC.encode())
    assert r.status_code == 202
    doc_id = r.json()["id"]
    got = client.get(f"/documents/{doc_id}").json()
    assert got["status"] == "done" and got["progress"] == 100


def test_ask_finds_the_relevant_chunk(client):
    doc_id = upload(client, DOC.encode()).json()["id"]
    r = client.post(f"/documents/{doc_id}/ask", json={"question": "How long is the device warranty?"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "extractive"
    assert "warranty" in body["sources"][0]["text"].lower()


def test_ask_before_ready_is_409(client):
    with patch("app.main.process_document.delay"):  # simulate worker not run yet
        doc_id = upload(client, DOC.encode()).json()["id"]
    r = client.post(f"/documents/{doc_id}/ask", json={"question": "anything here?"})
    assert r.status_code == 409


def test_rejects_unsupported_and_fake_pdf(client):
    assert upload(client, b"x", "run.exe").status_code == 415
    assert upload(client, b"not really a pdf", "fake.pdf").status_code == 415


def test_oversize_upload_is_rejected_and_not_kept(client, tmp_path):
    from pathlib import Path

    from app.config import get_settings

    before = len(list(Path(get_settings().upload_dir).glob("*")))
    big = b"a " * (11 * 1024 * 1024 // 2 + 10)
    assert upload(client, big).status_code == 413
    assert len(list(Path(get_settings().upload_dir).glob("*"))) == before


def test_path_traversal_filename_is_neutralised(client):
    doc_id = upload(client, DOC.encode(), "../../etc/passwd.txt").json()["id"]
    doc = client.get(f"/documents/{doc_id}").json()
    assert "/" not in doc["filename"] and ".." not in doc["filename"]


def test_empty_document_fails_cleanly(client):
    doc_id = upload(client, b"   \n  ").json()["id"]
    got = client.get(f"/documents/{doc_id}").json()
    assert got["status"] == "failed" and "no extractable text" in got["error"]


def test_pipeline_is_idempotent(client):
    doc_id = upload(client, DOC.encode()).json()["id"]
    with get_session_factory()() as db:
        first = db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == doc_id))
    run_pipeline(doc_id)  # simulates a redelivered task
    with get_session_factory()() as db:
        second = db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == doc_id))
    assert first == second > 1


def test_delete_removes_file_and_chunks(client):
    doc_id = upload(client, DOC.encode()).json()["id"]
    with get_session_factory()() as db:
        path = db.get(Document, doc_id).stored_path
    assert client.delete(f"/documents/{doc_id}").status_code == 204
    from pathlib import Path

    assert not Path(path).exists()
    assert client.get(f"/documents/{doc_id}").status_code == 404


def test_question_length_limits(client):
    doc_id = upload(client, DOC.encode()).json()["id"]
    assert client.post(f"/documents/{doc_id}/ask", json={"question": "x" * 501}).status_code == 422
    assert client.post(f"/documents/{doc_id}/ask", json={"question": "hi"}).status_code == 422


def test_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    get_settings.cache_clear()
    ratelimit.set_limiter(ratelimit.InMemoryLimiter())
    codes = [client.get("/health").status_code]  # not limited
    doc_id = upload(client, DOC.encode()).json()["id"]  # hit 1
    codes += [
        client.post(f"/documents/{doc_id}/ask", json={"question": "warranty period?"}).status_code
        for _ in range(4)
    ]  # hits 2..5
    assert codes[0] == 200
    assert codes[1:3] == [200, 200] and 429 in codes[3:]


def test_sse_stream_reports_done(client):
    doc_id = upload(client, DOC.encode()).json()["id"]
    with client.stream("GET", f"/documents/{doc_id}/events?poll_seconds=0.05") as r:
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert events[-1] == {"status": "done", "progress": 100, "error": None}


def test_llm_prompt_isolates_document_and_failure_falls_back(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    get_settings.cache_clear()
    doc_id = upload(client, DOC.encode()).json()["id"]

    captured = {}

    class FakeClient:
        def __init__(self, **kw):
            self.messages = self

        def create(self, **kw):
            captured.update(kw)
            raise RuntimeError("boom")

    with patch("anthropic.Anthropic", FakeClient):
        r = client.post(f"/documents/{doc_id}/ask", json={"question": "warranty period?"})
    assert r.status_code == 200 and r.json()["mode"] == "extractive-fallback"
    assert captured["max_tokens"] == get_settings().llm_max_tokens
    assert "untrusted" in captured["system"]
    assert "<document" in captured["messages"][0]["content"]
    assert qa.SYSTEM_PROMPT == captured["system"]
