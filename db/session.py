"""
db/session.py — SQLAlchemy engine + session factory.
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from config import settings, Paths

_DEFAULT_SQLITE_URL = f"sqlite:///{Paths.DATA_PROCESSED / 'local.db'}"


def _build_engine():
    url = settings.database_url or _DEFAULT_SQLITE_URL

    if url.startswith("sqlite"):
        # SQLite needs this flag to be usable across FastAPI's threaded requests
        return create_engine(url, connect_args={"check_same_thread": False})

    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=2,
        pool_recycle=300,
    )


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Create all tables that don't exist yet. Safe to call repeatedly."""
    from db.models import Base
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Use as:
        with get_session() as session:
            session.query(...)
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session_dependency() -> Generator[Session, None, None]:
    """FastAPI dependency form: `session: Session = Depends(get_session_dependency)`."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()