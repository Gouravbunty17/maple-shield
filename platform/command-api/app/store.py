"""SQLite-backed store. Append-only audit log with a hash chain.

The store is intentionally simple. There are NO delete endpoints; archive
operations write status changes, never erase.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .models import (
    Alert,
    AuditEntry,
    Incident,
    IncidentNote,
    IncidentStatus,
    Severity,
)


_GENESIS_HASH = "sha256:genesis"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _hash_audit(seq: int, ts: str, operator_id: str, action: str,
                target: Optional[str], payload: dict, prev_hash: str) -> str:
    """Deterministic hash so we can verify the chain on read."""
    blob = json.dumps({
        "seq": seq, "ts": ts, "operator_id": operator_id,
        "action": action, "target": target, "payload": payload,
        "prev_hash": prev_hash,
    }, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()


class Store:
    """Thread-safe SQLite store. One Store per process is fine for MVP."""

    def __init__(self, db_path: str | Path = ":memory:"):
        self._lock = threading.RLock()
        self._db_path = str(db_path)
        # check_same_thread=False is fine because we serialise via _lock.
        self._con = sqlite3.connect(
            self._db_path, check_same_thread=False, isolation_level=None,
        )
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        c = self._con
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                track_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                severity TEXT NOT NULL,
                rule TEXT NOT NULL,
                score REAL NOT NULL,
                ts TEXT NOT NULL,
                snapshot_b64 TEXT
            );
            CREATE INDEX IF NOT EXISTS alerts_ts_idx ON alerts(ts);

            CREATE TABLE IF NOT EXISTS incidents (
                incident_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_ts TEXT NOT NULL,
                updated_ts TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS incident_alerts (
                incident_id TEXT NOT NULL,
                alert_id TEXT NOT NULL,
                PRIMARY KEY (incident_id, alert_id)
            );
            CREATE TABLE IF NOT EXISTS incident_notes (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id TEXT NOT NULL,
                operator_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                text TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                operator_id TEXT NOT NULL,
                action TEXT NOT NULL,
                target TEXT,
                payload TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                hash TEXT NOT NULL
            );
            """
        )

    # ---------- alerts ----------

    def add_alert(self, alert: Alert) -> Alert:
        with self._lock:
            self._con.execute(
                "INSERT INTO alerts VALUES (?,?,?,?,?,?,?,?)",
                (alert.alert_id, alert.track_id, alert.camera_id,
                 alert.severity.value, alert.rule, alert.score,
                 _iso(alert.ts), alert.snapshot_b64),
            )
        return alert

    def list_alerts(self, limit: int = 100, severity: Optional[Severity] = None) -> List[Alert]:
        with self._lock:
            if severity is None:
                rows = self._con.execute(
                    "SELECT alert_id, track_id, camera_id, severity, rule, score, ts, snapshot_b64 "
                    "FROM alerts ORDER BY ts DESC LIMIT ?", (limit,),
                ).fetchall()
            else:
                rows = self._con.execute(
                    "SELECT alert_id, track_id, camera_id, severity, rule, score, ts, snapshot_b64 "
                    "FROM alerts WHERE severity = ? ORDER BY ts DESC LIMIT ?",
                    (severity.value, limit),
                ).fetchall()
        return [
            Alert(alert_id=r[0], track_id=r[1], camera_id=r[2],
                  severity=Severity(r[3]), rule=r[4], score=r[5],
                  ts=datetime.fromisoformat(r[6]), snapshot_b64=r[7])
            for r in rows
        ]

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        with self._lock:
            r = self._con.execute(
                "SELECT alert_id, track_id, camera_id, severity, rule, score, ts, snapshot_b64 "
                "FROM alerts WHERE alert_id = ?", (alert_id,),
            ).fetchone()
        if not r:
            return None
        return Alert(alert_id=r[0], track_id=r[1], camera_id=r[2],
                     severity=Severity(r[3]), rule=r[4], score=r[5],
                     ts=datetime.fromisoformat(r[6]), snapshot_b64=r[7])

    # ---------- incidents ----------

    def create_incident(self, alert_ids: List[str]) -> Incident:
        inc = Incident(alert_ids=alert_ids)
        with self._lock:
            self._con.execute(
                "INSERT INTO incidents VALUES (?,?,?,?)",
                (inc.incident_id, inc.status.value,
                 _iso(inc.created_ts), _iso(inc.updated_ts)),
            )
            for aid in alert_ids:
                self._con.execute(
                    "INSERT OR IGNORE INTO incident_alerts VALUES (?,?)",
                    (inc.incident_id, aid),
                )
        return inc

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        with self._lock:
            row = self._con.execute(
                "SELECT incident_id, status, created_ts, updated_ts "
                "FROM incidents WHERE incident_id = ?", (incident_id,),
            ).fetchone()
            if not row:
                return None
            alert_rows = self._con.execute(
                "SELECT alert_id FROM incident_alerts WHERE incident_id = ?",
                (incident_id,),
            ).fetchall()
            note_rows = self._con.execute(
                "SELECT operator_id, ts, text FROM incident_notes "
                "WHERE incident_id = ? ORDER BY rowid ASC",
                (incident_id,),
            ).fetchall()
        return Incident(
            incident_id=row[0],
            status=IncidentStatus(row[1]),
            created_ts=datetime.fromisoformat(row[2]),
            updated_ts=datetime.fromisoformat(row[3]),
            alert_ids=[r[0] for r in alert_rows],
            notes=[IncidentNote(operator_id=r[0], ts=datetime.fromisoformat(r[1]), text=r[2])
                   for r in note_rows],
        )

    def list_incidents(self, limit: int = 100) -> List[Incident]:
        with self._lock:
            rows = self._con.execute(
                "SELECT incident_id FROM incidents ORDER BY updated_ts DESC LIMIT ?", (limit,),
            ).fetchall()
        return [self.get_incident(r[0]) for r in rows if self.get_incident(r[0])]

    def set_incident_status(self, incident_id: str, status: IncidentStatus) -> Optional[Incident]:
        with self._lock:
            cur = self._con.execute(
                "UPDATE incidents SET status = ?, updated_ts = ? WHERE incident_id = ?",
                (status.value, _iso(_now()), incident_id),
            )
            if cur.rowcount == 0:
                return None
        return self.get_incident(incident_id)

    def add_incident_note(self, incident_id: str, operator_id: str, text: str) -> Optional[Incident]:
        ts = _iso(_now())
        with self._lock:
            row = self._con.execute(
                "SELECT 1 FROM incidents WHERE incident_id = ?", (incident_id,),
            ).fetchone()
            if not row:
                return None
            self._con.execute(
                "INSERT INTO incident_notes (incident_id, operator_id, ts, text) VALUES (?,?,?,?)",
                (incident_id, operator_id, ts, text),
            )
            self._con.execute(
                "UPDATE incidents SET updated_ts = ? WHERE incident_id = ?",
                (ts, incident_id),
            )
        return self.get_incident(incident_id)


    def find_open_incident_for_track(self, track_id: str) -> Optional[Incident]:
        """Return the most recently updated non-closed incident whose alerts
        include any alert for this track, or None."""
        with self._lock:
            row = self._con.execute(
                """
                SELECT i.incident_id
                FROM incidents i
                JOIN incident_alerts ia ON ia.incident_id = i.incident_id
                JOIN alerts a ON a.alert_id = ia.alert_id
                WHERE a.track_id = ? AND i.status != 'closed'
                ORDER BY i.updated_ts DESC LIMIT 1
                """,
                (track_id,),
            ).fetchone()
            if not row:
                return None
        return self.get_incident(row[0])

    def append_alert_to_incident(self, incident_id: str, alert_id: str) -> Optional[Incident]:
        with self._lock:
            row = self._con.execute(
                "SELECT 1 FROM incidents WHERE incident_id = ?", (incident_id,),
            ).fetchone()
            if not row:
                return None
            self._con.execute(
                "INSERT OR IGNORE INTO incident_alerts VALUES (?,?)",
                (incident_id, alert_id),
            )
            self._con.execute(
                "UPDATE incidents SET updated_ts = ? WHERE incident_id = ?",
                (_iso(_now()), incident_id),
            )
        return self.get_incident(incident_id)

    # ---------- audit log ----------

    def append_audit(self, operator_id: str, action: str,
                     target: Optional[str] = None,
                     payload: Optional[dict] = None) -> AuditEntry:
        payload = payload or {}
        with self._lock:
            row = self._con.execute(
                "SELECT seq, hash FROM audit_log ORDER BY seq DESC LIMIT 1",
            ).fetchone()
            prev_seq, prev_hash = (row[0], row[1]) if row else (0, _GENESIS_HASH)
            seq = prev_seq + 1
            ts = _iso(_now())
            h = _hash_audit(seq, ts, operator_id, action, target, payload, prev_hash)
            self._con.execute(
                "INSERT INTO audit_log (seq, ts, operator_id, action, target, payload, prev_hash, hash) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (seq, ts, operator_id, action, target,
                 json.dumps(payload, sort_keys=True), prev_hash, h),
            )
        return AuditEntry(
            seq=seq, ts=datetime.fromisoformat(ts), operator_id=operator_id,
            action=action, target=target, payload=payload,
            prev_hash=prev_hash, hash=h,
        )

    def list_audit(self, limit: int = 200) -> List[AuditEntry]:
        with self._lock:
            rows = self._con.execute(
                "SELECT seq, ts, operator_id, action, target, payload, prev_hash, hash "
                "FROM audit_log ORDER BY seq ASC LIMIT ?", (limit,),
            ).fetchall()
        return [AuditEntry(
            seq=r[0], ts=datetime.fromisoformat(r[1]),
            operator_id=r[2], action=r[3], target=r[4],
            payload=json.loads(r[5]), prev_hash=r[6], hash=r[7],
        ) for r in rows]

    def verify_audit(self) -> tuple[bool, Optional[int]]:
        """Verify the hash chain. Return (ok, first_bad_seq_or_None)."""
        prev_hash = _GENESIS_HASH
        with self._lock:
            rows = self._con.execute(
                "SELECT seq, ts, operator_id, action, target, payload, prev_hash, hash "
                "FROM audit_log ORDER BY seq ASC",
            ).fetchall()
        for r in rows:
            (seq, ts, op, action, target, payload_s, stored_prev, stored_hash) = r
            if stored_prev != prev_hash:
                return False, seq
            recomputed = _hash_audit(seq, ts, op, action, target,
                                     json.loads(payload_s), stored_prev)
            if recomputed != stored_hash:
                return False, seq
            prev_hash = stored_hash
        return True, None
