import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "checkpoints.sqlite")


def get_checkpointer() -> SqliteSaver:
    """SQLite tabanlı long-term memory (checkpointer) döndürür.
    
    Konuşma geçmişi, durum anlık görüntüleri (snapshots) ve iterasyonlar
    SQLite veritabanında thread_id bazlı saklanır.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return SqliteSaver(conn)
