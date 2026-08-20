import sqlite3
import os
from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = os.path.join(os.path.dirname(__file__), "checkpoints.sqlite")

db_connection = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(db_connection)

def get_thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}