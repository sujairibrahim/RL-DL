"""
remarl/rl/memory.py
-------------------
SQLite-backed episode memory.
Stores (episode_id, domain, total_reward, experiences_json, timestamp).
Used for logging, debugging, and future experience replay.
"""

import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class EpisodeMemory:
    """
    Lightweight episode store backed by SQLite.

    Usage:
        mem = EpisodeMemory("data/episodes.db")
        mem.store(ep_id=1, domain="healthcare", reward=0.72, experiences=[...])
        rows = mem.recent(n=10)
        stats = mem.stats()
    """

    CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS episodes (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        episode_id  INTEGER NOT NULL,
        domain      TEXT,
        difficulty  TEXT,
        reward      REAL,
        n_steps     INTEGER,
        covered_pct REAL,
        experiences TEXT,
        created_at  TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_episode ON episodes(episode_id);
    CREATE INDEX IF NOT EXISTS idx_domain  ON episodes(domain);
    """

    def __init__(self, db_path: str = "data/episodes.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def store(
        self,
        episode_id: int,
        domain: str,
        reward: float,
        experiences: list,
        difficulty: str = "",
        covered_pct: float = 0.0,
    ):
        n_steps = len(experiences)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO episodes
                   (episode_id, domain, difficulty, reward, n_steps,
                    covered_pct, experiences, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    episode_id,
                    domain,
                    difficulty,
                    round(reward, 6),
                    n_steps,
                    round(covered_pct, 4),
                    json.dumps(experiences),
                    datetime.utcnow().isoformat(),
                ),
            )

    def recent(self, n: int = 20) -> List[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM episodes ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*), AVG(reward), MAX(reward), MIN(reward) FROM episodes"
            ).fetchone()
        return {
            "total_episodes": row[0] or 0,
            "mean_reward":    round(row[1] or 0, 4),
            "max_reward":     round(row[2] or 0, 4),
            "min_reward":     round(row[3] or 0, 4),
        }

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            for stmt in self.CREATE_SQL.strip().split(";"):
                if stmt.strip():
                    conn.execute(stmt)
