"""
SQLAlchemy models cho BDD100K metadata.
"""

from sqlalchemy import Column, Integer, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class ImageMetadata(Base):
    """Table lưu metadata của ảnh BDD100K."""

    __tablename__ = "image_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), unique=True, nullable=False, index=True)
    weather = Column(String(50), nullable=False, index=True)
    scene = Column(String(50), nullable=False, index=True)
    timeofday = Column(String(50), nullable=False, index=True)
    labels_count = Column(Integer, default=0)
    labels_summary = Column(JSON)
    created_at = Column(Datetime, default=datetime.utcnow)

    def __repr__ (self):
        return f"<ImageMetadata(filename={self.filename}, weather={self.weather})>"