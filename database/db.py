"""
Quick inspection helper for fx_features table.
"""

from sqlalchemy import text

import project_paths  # noqa: F401
from database.save_db import get_engine

engine = get_engine()

with engine.connect() as conn:
    rows = conn.execute(text("SELECT * FROM fx_features ORDER BY date DESC LIMIT 10;"))
    for row in rows:
        print(row)
