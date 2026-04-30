"""Local event store for DDIL/air-gapped Cairn-Edge operation."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "to_json"):
        return obj.to_json()  # type: ignore[no-any-return]
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}


class LocalEventStore:
    def __init__(self, sqlite_path: str | Path, jsonl_path: Optional[str | Path] = None) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.sqlite_path))
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                node_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                track_id TEXT,
                threat_level TEXT,
                payload_json TEXT NOT NULL,
                forwarded INTEGER DEFAULT 0
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_forwarded ON events(forwarded)")
        self._conn.commit()

    def write_event(
        self,
        *,
        node_id: str,
        event_type: str,
        payload: Any,
        track_id: str | None = None,
        threat_level: str | None = None,
    ) -> None:
        record = {
            "ts": utc_now(),
            "node_id": node_id,
            "event_type": event_type,
            "track_id": track_id,
            "threat_level": threat_level,
            "payload": _safe_json(payload),
        }
        payload_json = json.dumps(record, separators=(",", ":"), sort_keys=True)
        self._conn.execute(
            "INSERT INTO events(ts, node_id, event_type, track_id, threat_level, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
            (record["ts"], node_id, event_type, track_id, threat_level, payload_json),
        )
        self._conn.commit()

        if self.jsonl_path:
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(payload_json + "\n")

    def fetch_unforwarded(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, payload_json FROM events WHERE forwarded = 0 ORDER BY id ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"id": row[0], "payload": json.loads(row[1])} for row in rows]

    def mark_forwarded(self, event_id: int) -> None:
        self._conn.execute("UPDATE events SET forwarded = 1 WHERE id = ?", (event_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
