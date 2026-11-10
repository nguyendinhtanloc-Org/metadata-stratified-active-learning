"""
Database connection setup.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import engine
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from db.models import Base

# Database từ environments hoặc default
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/bdd100k"
)

def get_engine():
    """Tạo SQLAlchemy engine."""

    return create_engine(
        DATABASE_URL,
        poolclass=StaticPool, # Single connection cho dev
        echo=False
    )

def get_session() -> Session:
    """Tạo database session."""

    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)

    return SessionLocal

def init_db():
    """Khởi tạo database (tạo tables)."""

    engine = get_engine()
    Base.metadata.create_all(engine)
