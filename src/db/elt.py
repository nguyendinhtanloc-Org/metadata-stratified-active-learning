"""
ETL pipeline: Load BDD100K metadata vào PostgreSQL.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.connection import get_session, init_db
from db.models import ImageMetadata
from data.loader import get_all_metadata

def elt_load_metadata(json_path: str) -> int:
    """
    Extract từ JSON và Load vào PostgreSQL.
    
    Args:
        json_path: Đường dẫn đến BDD100K annotations JSON
    
    Returns:
        Số records đã insert
    """

    # init database
    init_db()

    # extract metadata
    metadata_list = get_all_metadata(json_path)

    # load vào database
    session = get_session()
    count = 0

    for meta in metadata_list:
        # kiểm tra có tồn tại hay chưa
        existing = session.query(ImageMetadata).filter_by(
            filename=meta["filename"]
        ).first()

        if existing:
            continue

        # tạo record mới
        record = ImageMetadata(
            filename=meta["filename"],
            weather=meta["weather"],
            scene=meta["scene"],
            timeofday=meta["timeofday"],
            labels_count=len(meta.get("labels", [])),
        )

        session.add(record)
        count += 1

    session.commit()
    session.close()

    return count
