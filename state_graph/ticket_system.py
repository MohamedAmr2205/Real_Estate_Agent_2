import sqlite3
import os
from typing import Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "checkpoints.sqlite")

def create_failure_ticket(thread_id: str, graph_name: str, error_msg: str, current_state: Dict[str, Any]) -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT,
            graph_name TEXT,
            error_message TEXT,
            status TEXT DEFAULT 'OPEN',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute(
        "INSERT INTO tickets (thread_id, graph_name, error_message) VALUES (?, ?, ?)",
        (thread_id, graph_name, str(error_msg))
    )
    conn.commit()
    ticket_id = cursor.lastrowid
    conn.close()
    
    return str(ticket_id)