import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import qa
from .config import get_settings
from .db import get_db, get_session_factory, init_db
from .extract import UnsupportedFile, validate_upload
from .models import Document, Status
from .ratelimit import rate_limit
from .tasks import process_document

CHUNK = 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    Path(get_settings().upload_dir).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="DocRAG", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in get_settings().cors_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


class DocumentOut(BaseModel):
    id: str
    filename: str
    status: str
    progress: int
    error: str | None = None

    @classmethod
    def of(cls, d: Document) -> "DocumentOut":
        return cls(id=d.id, filename=d.filename, status=d.status, progress=d.progress, error=d.error)


class AskIn(BaseModel):
    question: str = Field(min_length=3)


class SourceOut(BaseModel):
    chunk: int
    score: float
    text: str


class AnswerOut(BaseModel):
    answer: str
    mode: str
    sources: list[SourceOut]


def _get_or_404(db: Session, document_id: str) -> Document:
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents", status_code=202, response_model=DocumentOut, dependencies=[Depends(rate_limit)])
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    settings = get_settings()
    original = Path(file.filename or "upload").name  # strip any path components
    max_bytes = settings.max_upload_mb * 1024 * 1024

    head = await file.read(2048)
    try:
        ext = validate_upload(original, head)
    except UnsupportedFile as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    # Stored under a random name: the client never controls the on-disk path.
    stored = Path(settings.upload_dir) / f"{uuid.uuid4().hex}{ext}"
    stored.parent.mkdir(parents=True, exist_ok=True)
    size = len(head)
    try:
        with stored.open("wb") as out:
            out.write(head)
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status_code=413, detail=f"File larger than {settings.max_upload_mb} MB")
                out.write(chunk)
    except HTTPException:
        stored.unlink(missing_ok=True)
        raise

    doc = Document(filename=original[:255], stored_path=str(stored))
    db.add(doc)
    db.commit()
    process_document.delay(doc.id)  # returns immediately; worker does the heavy work
    return DocumentOut.of(doc)


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    docs = db.scalars(select(Document).order_by(Document.created_at.desc()).limit(100)).all()
    return [DocumentOut.of(d) for d in docs]


@app.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, db: Session = Depends(get_db)):
    return DocumentOut.of(_get_or_404(db, document_id))


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, db: Session = Depends(get_db)):
    doc = _get_or_404(db, document_id)
    Path(doc.stored_path).unlink(missing_ok=True)
    db.delete(doc)
    db.commit()


def _snapshot(document_id: str) -> dict | None:
    with get_session_factory()() as db:
        doc = db.get(Document, document_id)
        if doc is None:
            return None
        return {"status": doc.status, "progress": doc.progress, "error": doc.error}


@app.get("/documents/{document_id}/events")
async def document_events(document_id: str, poll_seconds: float = 0.5, max_seconds: int = 300):
    """Server-Sent Events: one-way progress stream, lighter than a WebSocket."""
    if await run_in_threadpool(_snapshot, document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    poll_seconds = min(max(poll_seconds, 0.05), 5.0)

    async def stream():
        last = None
        waited = 0.0
        while waited <= max_seconds:
            snap = await run_in_threadpool(_snapshot, document_id)
            if snap is None:
                yield "event: error\ndata: {\"detail\": \"deleted\"}\n\n"
                return
            if snap != last:
                yield f"data: {json.dumps(snap)}\n\n"
                last = snap
            if snap["status"] in (Status.DONE, Status.FAILED):
                return
            await asyncio.sleep(poll_seconds)
            waited += poll_seconds

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/documents/{document_id}/ask", response_model=AnswerOut, dependencies=[Depends(rate_limit)])
def ask(document_id: str, body: AskIn, db: Session = Depends(get_db)):
    settings = get_settings()
    if len(body.question) > settings.max_question_chars:
        raise HTTPException(status_code=422, detail=f"Question longer than {settings.max_question_chars} characters")
    doc = _get_or_404(db, document_id)
    if doc.status != Status.DONE:
        raise HTTPException(status_code=409, detail=f"Document is not ready (status: {doc.status})")
    text, sources, mode = qa.answer(db, document_id, body.question, settings)
    return AnswerOut(
        answer=text,
        mode=mode,
        sources=[SourceOut(chunk=s.idx, score=s.score, text=s.text) for s in sources],
    )
