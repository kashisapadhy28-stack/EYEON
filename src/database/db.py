"""
EYEON - Database Module for Event and Alert Persistence
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from src.utils.config_loader import load_config, get_path


class DatabaseManager:
    """
    SQLite Database Manager for saving, updating, and querying EYEON security alerts & events.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        if db_path is None:
            try:
                self.db_path = get_path("database.path")
            except Exception:
                self.db_path = Path("data/eyeon.db")
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self):
        """Context manager for SQLite connection that automatically closes on exit."""
        class _ConnCtx:
            def __init__(self, db_path):
                self.db_path = db_path
                self.conn = None

            def __enter__(self):
                self.conn = sqlite3.connect(str(self.db_path))
                self.conn.row_factory = sqlite3.Row
                return self.conn

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.conn:
                    if exc_type is None:
                        self.conn.commit()
                    self.conn.close()

        return _ConnCtx(self.db_path)

    def init_db(self) -> None:
        """Creates the events table and indices if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    timestamp REAL NOT NULL,
                    datetime_str TEXT NOT NULL,
                    person_track_id INTEGER NOT NULL,
                    risk_score REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    reasons TEXT NOT NULL,
                    associated_objects TEXT,
                    zones TEXT,
                    snapshot_path TEXT,
                    status TEXT DEFAULT 'ACTIVE'
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_level ON events(risk_level);
            """)
            conn.commit()

    def log_event(
        self,
        alert_id: str,
        timestamp: float,
        person_track_id: int,
        risk_score: float,
        risk_level: str,
        reasons: List[str],
        associated_objects: Optional[List[int]] = None,
        zones: Optional[List[str]] = None,
        snapshot_path: Optional[str] = None,
        status: str = "ACTIVE",
    ) -> bool:
        """
        Inserts a new event record into the database. Returns True if successful.
        """
        datetime_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        reasons_json = json.dumps(reasons)
        assoc_json = json.dumps(associated_objects or [])
        zones_json = json.dumps(zones or [])

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO events (
                        alert_id, timestamp, datetime_str, person_track_id,
                        risk_score, risk_level, reasons, associated_objects,
                        zones, snapshot_path, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        alert_id,
                        timestamp,
                        datetime_str,
                        person_track_id,
                        risk_score,
                        risk_level,
                        reasons_json,
                        assoc_json,
                        zones_json,
                        snapshot_path,
                        status,
                    ),
                )
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            # Alert ID already exists
            return False
        except Exception as e:
            print(f"Error logging event to DB: {e}")
            return False

    def update_event_status(self, alert_id: str, status: str) -> bool:
        """
        Updates the status of an alert (e.g. ACKNOWLEDGED, RESOLVED).
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE events SET status = ? WHERE alert_id = ?",
                (status, alert_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_recent_events(
        self, limit: int = 50, level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetches the most recent events, optionally filtered by risk level.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if level:
                cursor.execute(
                    "SELECT * FROM events WHERE risk_level = ? ORDER BY timestamp DESC LIMIT ?",
                    (level, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                item = dict(row)
                item["reasons"] = json.loads(item["reasons"])
                item["associated_objects"] = json.loads(item["associated_objects"])
                item["zones"] = json.loads(item["zones"])
                results.append(item)
            return results

    def get_event_counts(self) -> Dict[str, int]:
        """
        Returns count of events per risk level.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT risk_level, COUNT(*) as count FROM events GROUP BY risk_level"
            )
            rows = cursor.fetchall()
            counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
            for row in rows:
                counts[row["risk_level"]] = row["count"]
            return counts
