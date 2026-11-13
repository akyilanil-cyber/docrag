from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        url = get_settings().database_url
        kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
        _engine = create_engine(url, pool_pre_ping=True, **kwargs)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def get_db() -> Iterator[Session]:
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from sqlalchemy import text

    from . import models  # noqa: F401  (register tables)

    engine = get_engine()
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)


def reset_engine() -> None:
    """Used by tests to point the app at a fresh database."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
