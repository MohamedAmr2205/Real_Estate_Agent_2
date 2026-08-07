"""
Builds db/database.sqlite from schema.sql + seed.sql.
Run once after pulling new schema/seed changes, or any time you want
to reset the database to a clean state.
"""

import sqlite3
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent
DB_PATH = DB_DIR / "database.sqlite"
SCHEMA_PATH = DB_DIR / "schema.sql"
SEED_PATH = DB_DIR / "seed.sql"

# start clean every time this is run
if DB_PATH.exists():
    DB_PATH.unlink()
    print(f"Removed existing {DB_PATH.name}")

conn = sqlite3.connect(DB_PATH)

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    conn.executescript(f.read())
print("Schema applied.")

with open(SEED_PATH, "r", encoding="utf-8") as f:
    conn.executescript(f.read())
print("Seed data inserted.")

conn.commit()
conn.close()
print(f"Done. Database built at {DB_PATH}")