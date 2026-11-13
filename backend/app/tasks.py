import logging

from sqlalchemy import delete
from sqlalchemy.exc import OperationalError

from .celery_app import celery_app
from .chunking import chunk_text
from .config import get_settings
from .db import get_session_factory
from .embeddings import HashingEmbedder
from .extract import extract_text
from .models import Chunk, Document, Status

log = logging.getLogger(__name__)

BATCH = 32


class PermanentError(Exception):
    """Input problem that retrying will not fix."""


def _set(db, doc: Document, **fields) -> None:
    for key, value in fields.items():
        setattr(doc, key, value)
    db.commit()


def run_pipeline(document_id: str) -> None:
    """Extract -> chunk -> embed -> store. Idempotent: safe to run twice."""
    settings = get_settings()
    embedder = HashingEmbedder(settings.embed_dim)

    with get_session_factory()() as db:
        doc = db.get(Document, document_id)
        if doc is None:
            raise PermanentError(f"document {document_id} not found")

        _set(db, doc, status=Status.PROCESSING, progress=5, error=None)
        # Idempotency: a redelivered task must not duplicate chunks.
        db.execute(delete(Chunk).where(Chunk.document_id == document_id))
        db.commit()

        try:
            text = extract_text(doc.stored_path)
        except Exception as exc:  # unreadable / corrupt file
            raise PermanentError(f"could not read file: {exc}") from exc

        pieces = chunk_text(text, settings.chunk_words, settings.chunk_overlap)
        if not pieces:
            raise PermanentError("document contains no extractable text")
        _set(db, doc, progress=20)

        total = len(pieces)
        for start in range(0, total, BATCH):
            batch = pieces[start : start + BATCH]
            vectors = embedder.embed(batch)
            db.add_all(
                Chunk(document_id=document_id, idx=start + i, text=t, embedding=v)
                for i, (t, v) in enumerate(zip(batch, vectors))
            )
            progress = 20 + int(75 * min(start + BATCH, total) / total)
            _set(db, doc, progress=progress)

        _set(db, doc, status=Status.DONE, progress=100)


def _mark_failed(document_id: str, message: str) -> None:
    with get_session_factory()() as db:
        doc = db.get(Document, document_id)
        if doc is not None:
            _set(db, doc, status=Status.FAILED, error=message[:500])


@celery_app.task(
    bind=True,
    name="docrag.process_document",
    autoretry_for=(OperationalError,),
    retry_backoff=True,
    retry_backoff_max=60,
    max_retries=3,
)
def process_document(self, document_id: str) -> None:
    try:
        run_pipeline(document_id)
    except PermanentError as exc:
        log.warning("document %s failed: %s", document_id, exc)
        _mark_failed(document_id, str(exc))
    except OperationalError:
        if self.request.retries >= self.max_retries:
            _mark_failed(document_id, "database unavailable, gave up after retries")
            return
        raise
