"""
Database module cho DE minimal.
Load metadata vào PostgreSQL.
"""

from src.db.models import Base, ImageMetadata
from src.db.connection import get_engine, get_session

__all__ = ["Base", "ImageMetadata", "get_engine", "get_session"]
