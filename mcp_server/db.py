"""
Shared SQLite connection helper.

All tool handlers import `get_connection()` from here instead of opening
their own connections, so the whole server talks to ONE database file
consistently (db/meridian.db, built from db/schema.sql + db/seed_data.sql).
"""

import sqlite3
from pathlib import Path

# Path to the sqlite file produced from db/schema.sql + db/seed_data.sql
DB_PATH = Path(__file__).resolve().parent.parent / "db" / "meridian.db"


def get_connection() -> sqlite3.Connection:
    """Return a connection with row access by column name (row['col'])."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")  # enforce FK constraints
    return conn