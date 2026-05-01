"""Maple Shield command-api.

FastAPI service that owns:
  * incidents
  * alerts
  * tracks (read-through)
  * replay (read-through)
  * audit log
  * health endpoints

The service NEVER originates an action against any drone or other target.
It only persists events, exposes them to the operator UI, and records
operator-initiated state changes to the audit log.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .models import Alert, Incident, IncidentStatus, Severity
from .store import Store


# ---------- bootstrap ----------

_DB_PATH = os.environ.get("MAPLE_SHIELD_DB", ":memory:")
store = Store(_DB_PATH)

# WebSocket subscribers - simple list of asyncio Queues for live UI streaming
_subscribers: list[asyncio.Queue] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    # boot-time announcement, helps detect misconfig
    if os.environ.get("MAPLE_SHIELD_LAWFUL_USE_ACK") != "true":
        print("[command-api] WARNING: MAPLE_SHIELD_LAWFUL_USE_ACK is not set. "
              "Confirm lawful deployment in your jurisdiction.")
    yield


app = FastAPI(
    title="Maple Shield - command-api",
    version="0.1.0",
    description="Passive airspace monitoring API. No interception. No jamming. No autonomous engagement.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


# ---------- request bodies ----------


class IncidentStatusUpdate(BaseModel):
    status: IncidentStatus
    operator_id: str


class IncidentNoteCreate(BaseModel):
    operator_id: str
    text: str


class IncidentCreateFromAlert(BaseModel):
    alert_id: str
    operator_id: str = "system"


# ---------- health ----------


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "service": "command-api",
        "lawful_use_ack": os.environ.get("MAPLE_SHIELD_LAWFUL_USE_ACK") == "true",
    }


@app.get("/readyz")
def readyz():
    # cheap query to confirm SQLite is responsive
    store.list_alerts(limit=1)
    return {"status": "ready"}


# ---------- alerts ----------


@app.post("/alerts", response_model=Alert, status_code=201)
def post_alert(alert: Alert):
    """Ingest an alert from fusion-engine.

    A new alert with severity >= med opens an incident, OR is appended to
    the existing open incident for the same track. This is a bookkeeping
    convenience, not an action against any target.
    """
    saved = store.add_alert(alert)
    if alert.severity in (Severity.med, Severity.high):
        existing = store.find_open_incident_for_track(alert.track_id)
        if existing is None:
            store.create_incident([alert.alert_id])
            store.append_audit(
                operator_id="system",
                action="incident.auto_create",
                target=alert.alert_id,
                payload={"reason": f"severity={alert.severity.value}"},
            )
        else:
            store.append_alert_to_incident(existing.incident_id, alert.alert_id)
            store.append_audit(
                operator_id="system",
                action="incident.auto_append",
                target=existing.incident_id,
                payload={"alert_id": alert.alert_id},
            )
    # broadcast to UI subscribers
    for q in list(_subscribers):
        try:
            q.put_nowait({"kind": "alert", "alert": saved.model_dump(mode="json")})
        except asyncio.QueueFull:
            pass
    return saved


@app.get("/alerts", response_model=list[Alert])
def list_alerts(limit: int = Query(100, ge=1, le=500),
                severity: Optional[Severity] = None):
    return store.list_alerts(limit=limit, severity=severity)


@app.get("/alerts/{alert_id}", response_model=Alert)
def get_alert(alert_id: str):
    a = store.get_alert(alert_id)
    if not a:
        raise HTTPException(404, "alert not found")
    return a


# ---------- incidents ----------


@app.get("/incidents", response_model=list[Incident])
def list_incidents(limit: int = Query(100, ge=1, le=500)):
    return store.list_incidents(limit=limit)


@app.get("/incidents/{incident_id}", response_model=Incident)
def get_incident(incident_id: str):
    inc = store.get_incident(incident_id)
    if not inc:
        raise HTTPException(404, "incident not found")
    return inc


@app.post("/incidents", response_model=Incident, status_code=201)
def create_incident(body: IncidentCreateFromAlert):
    if not store.get_alert(body.alert_id):
        raise HTTPException(404, "alert not found")
    inc = store.create_incident([body.alert_id])
    store.append_audit(
        operator_id=body.operator_id,
        action="incident.create",
        target=inc.incident_id,
        payload={"alert_id": body.alert_id},
    )
    return inc


@app.patch("/incidents/{incident_id}/status", response_model=Incident)
def patch_incident_status(incident_id: str, body: IncidentStatusUpdate):
    inc = store.set_incident_status(incident_id, body.status)
    if not inc:
        raise HTTPException(404, "incident not found")
    store.append_audit(
        operator_id=body.operator_id,
        action="incident.status_change",
        target=incident_id,
        payload={"new_status": body.status.value},
    )
    return inc


@app.post("/incidents/{incident_id}/notes", response_model=Incident)
def post_incident_note(incident_id: str, body: IncidentNoteCreate):
    inc = store.add_incident_note(incident_id, body.operator_id, body.text)
    if not inc:
        raise HTTPException(404, "incident not found")
    store.append_audit(
        operator_id=body.operator_id,
        action="incident.note_add",
        target=incident_id,
        payload={"len": len(body.text)},
    )
    return inc


@app.get("/incidents/{incident_id}/export")
def export_incident(incident_id: str, operator_id: str = "system"):
    inc = store.get_incident(incident_id)
    if not inc:
        raise HTTPException(404, "incident not found")
    alerts = [store.get_alert(aid) for aid in inc.alert_ids]
    alerts = [a.model_dump(mode="json") for a in alerts if a]
    md_lines = [
        f"# Incident {inc.incident_id}",
        f"- status: **{inc.status.value}**",
        f"- created: {inc.created_ts.isoformat()}",
        f"- updated: {inc.updated_ts.isoformat()}",
        f"- alert count: {len(alerts)}",
        "",
        "## Alerts",
    ]
    for a in alerts:
        md_lines.append(
            f"- `{a['alert_id']}` severity=**{a['severity']}** rule={a['rule']} "
            f"score={a['score']:.2f} ts={a['ts']}"
        )
    if inc.notes:
        md_lines.append("\n## Operator notes")
        for n in inc.notes:
            md_lines.append(f"- {n.ts.isoformat()} _{n.operator_id}_: {n.text}")

    store.append_audit(
        operator_id=operator_id,
        action="incident.export",
        target=incident_id,
    )
    return {
        "incident": inc.model_dump(mode="json"),
        "alerts": alerts,
        "summary_md": "\n".join(md_lines),
    }


# ---------- replay (read-through; track state lives in fusion) ----------


@app.get("/replay/{incident_id}")
def replay_incident(incident_id: str):
    """Return data sufficient to render an incident replay timeline.

    For MVP this returns the alerts ordered in time, plus the snapshot
    payloads. A future version will include track waypoints from
    fusion-engine.
    """
    inc = store.get_incident(incident_id)
    if not inc:
        raise HTTPException(404, "incident not found")
    alerts = [store.get_alert(aid) for aid in inc.alert_ids]
    alerts = [a for a in alerts if a]
    alerts.sort(key=lambda a: a.ts)
    return {
        "incident": inc.model_dump(mode="json"),
        "frames": [
            {
                "ts": a.ts.isoformat(),
                "alert_id": a.alert_id,
                "severity": a.severity.value,
                "snapshot_b64": a.snapshot_b64,
            }
            for a in alerts
        ],
    }


# ---------- audit ----------


@app.get("/audit")
def list_audit(limit: int = Query(200, ge=1, le=1000)):
    entries = store.list_audit(limit=limit)
    ok, bad = store.verify_audit()
    return {
        "verified": ok,
        "first_bad_seq": bad,
        "entries": [e.model_dump(mode="json") for e in entries],
    }


# ---------- websocket (live alerts) ----------


@app.websocket("/ws/alerts")
async def ws_alerts(websocket):
    """Stream new alerts to the operator-ui."""
    await websocket.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=128)
    _subscribers.append(q)
    try:
        while True:
            msg = await q.get()
            await websocket.send_json(msg)
    except Exception:
        pass
    finally:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass
