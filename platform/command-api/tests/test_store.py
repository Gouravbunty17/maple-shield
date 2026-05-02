"""Unit tests for the SQLite-backed store and audit chain."""

from datetime import datetime, timezone

from app.models import Alert, IncidentStatus, Severity
from app.store import Store


def _alert(severity=Severity.med, score=0.7, track="trk-1") -> Alert:
    return Alert(
        track_id=track, camera_id="cam-01", severity=severity,
        rule="dwell_over_threshold", score=score,
        ts=datetime.now(timezone.utc),
    )


def test_add_and_list_alert():
    s = Store(":memory:")
    a = s.add_alert(_alert())
    out = s.list_alerts()
    assert len(out) == 1
    assert out[0].alert_id == a.alert_id


def test_filter_by_severity():
    s = Store(":memory:")
    s.add_alert(_alert(severity=Severity.low))
    s.add_alert(_alert(severity=Severity.high))
    assert len(s.list_alerts(severity=Severity.low)) == 1
    assert len(s.list_alerts(severity=Severity.high)) == 1
    assert len(s.list_alerts()) == 2


def test_create_incident_and_status_lifecycle():
    s = Store(":memory:")
    a = s.add_alert(_alert())
    inc = s.create_incident([a.alert_id])
    assert inc.status == IncidentStatus.new
    inc2 = s.set_incident_status(inc.incident_id, IncidentStatus.acknowledged)
    assert inc2.status == IncidentStatus.acknowledged
    inc3 = s.set_incident_status(inc.incident_id, IncidentStatus.reviewed)
    assert inc3.status == IncidentStatus.reviewed
    inc4 = s.set_incident_status(inc.incident_id, IncidentStatus.closed)
    assert inc4.status == IncidentStatus.closed


def test_incident_notes_persist():
    s = Store(":memory:")
    a = s.add_alert(_alert())
    inc = s.create_incident([a.alert_id])
    s.add_incident_note(inc.incident_id, "op-1", "saw it on the south fence")
    s.add_incident_note(inc.incident_id, "op-1", "operator escalated")
    inc2 = s.get_incident(inc.incident_id)
    assert len(inc2.notes) == 2
    assert "south fence" in inc2.notes[0].text


def test_audit_chain_verifies():
    s = Store(":memory:")
    s.append_audit("op-1", "incident.create", "inc-x")
    s.append_audit("op-1", "incident.note_add", "inc-x", payload={"len": 12})
    s.append_audit("op-2", "incident.status_change", "inc-x",
                   payload={"new_status": "reviewed"})
    ok, bad = s.verify_audit()
    assert ok is True
    assert bad is None
    assert len(s.list_audit()) == 3


def test_audit_chain_detects_tampering():
    s = Store(":memory:")
    s.append_audit("op-1", "incident.create", "inc-x")
    s.append_audit("op-1", "incident.note_add", "inc-x", payload={"len": 12})
    # Tamper with one row directly in SQLite.
    s._con.execute(
        "UPDATE audit_log SET payload = ? WHERE seq = 2",
        ('{"len": 9999}',),
    )
    ok, bad = s.verify_audit()
    assert ok is False
    assert bad == 2


def test_no_delete_methods():
    """Store must not expose a way to erase records (audit + incidents)."""
    s = Store(":memory:")
    forbidden = {"delete", "remove", "drop", "wipe", "purge", "erase"}
    public = {n for n in dir(s) if not n.startswith("_")}
    overlap = {n for n in public if any(f in n.lower() for f in forbidden)}
    assert overlap == set(), f"Store exposes delete-like methods: {overlap}"
