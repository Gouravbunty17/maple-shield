"""HTTP-level smoke tests for command-api."""

from fastapi.testclient import TestClient

import app.main as main_mod


def _client(monkeypatch):
    # fresh in-memory store per test
    from app.store import Store
    main_mod.store = Store(":memory:")
    return TestClient(main_mod.app)


def test_healthz(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "command-api"
    assert "lawful_use_ack" in body


def test_post_alert_creates_incident_when_severity_high(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/alerts", json={
        "track_id": "trk-1", "camera_id": "cam-01",
        "severity": "high", "rule": "dwell_over_threshold", "score": 0.92,
    })
    assert r.status_code == 201
    incs = c.get("/incidents").json()
    assert len(incs) == 1
    assert incs[0]["status"] == "new"


def test_low_severity_does_not_auto_create_incident(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/alerts", json={
        "track_id": "trk-1", "camera_id": "cam-01",
        "severity": "low", "rule": "single_obs", "score": 0.30,
    })
    assert r.status_code == 201
    assert c.get("/incidents").json() == []


def test_status_lifecycle_via_api(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/alerts", json={
        "track_id": "trk-1", "camera_id": "cam-01",
        "severity": "high", "rule": "dwell", "score": 0.9,
    })
    aid = r.json()["alert_id"]
    incs = c.get("/incidents").json()
    inc_id = incs[0]["incident_id"]

    r2 = c.patch(f"/incidents/{inc_id}/status", json={
        "status": "acknowledged", "operator_id": "op-7",
    })
    assert r2.status_code == 200
    assert r2.json()["status"] == "acknowledged"

    r3 = c.post(f"/incidents/{inc_id}/notes", json={
        "operator_id": "op-7", "text": "checked perimeter",
    })
    assert r3.status_code == 200
    assert len(r3.json()["notes"]) == 1


def test_export_returns_summary_md(monkeypatch):
    c = _client(monkeypatch)
    c.post("/alerts", json={
        "track_id": "trk-1", "camera_id": "cam-01",
        "severity": "high", "rule": "dwell", "score": 0.9,
    })
    incs = c.get("/incidents").json()
    inc_id = incs[0]["incident_id"]
    r = c.get(f"/incidents/{inc_id}/export")
    assert r.status_code == 200
    body = r.json()
    assert "summary_md" in body
    assert body["summary_md"].startswith(f"# Incident {inc_id}")


def test_audit_log_records_operator_actions(monkeypatch):
    c = _client(monkeypatch)
    c.post("/alerts", json={
        "track_id": "trk-1", "camera_id": "cam-01",
        "severity": "high", "rule": "dwell", "score": 0.9,
    })
    incs = c.get("/incidents").json()
    inc_id = incs[0]["incident_id"]
    c.patch(f"/incidents/{inc_id}/status", json={
        "status": "acknowledged", "operator_id": "op-7",
    })
    audit = c.get("/audit").json()
    assert audit["verified"] is True
    actions = [e["action"] for e in audit["entries"]]
    assert "incident.auto_create" in actions
    assert "incident.status_change" in actions


def test_no_delete_endpoint_exists(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/alerts", json={
        "track_id": "trk-1", "camera_id": "cam-01",
        "severity": "high", "rule": "dwell", "score": 0.9,
    })
    aid = r.json()["alert_id"]
    # No DELETE on alerts/incidents/audit. 405 expected (method not allowed).
    assert c.delete(f"/alerts/{aid}").status_code in (404, 405)
    incs = c.get("/incidents").json()
    inc_id = incs[0]["incident_id"]
    assert c.delete(f"/incidents/{inc_id}").status_code in (404, 405)
    assert c.delete("/audit/1").status_code in (404, 405)


def test_repeat_high_alerts_for_same_track_dont_create_duplicate_incidents(monkeypatch):
    c = _client(monkeypatch)
    for _ in range(5):
        c.post("/alerts", json={
            "track_id": "trk-X", "camera_id": "cam-01",
            "severity": "high", "rule": "dwell", "score": 0.9,
        })
    incs = c.get("/incidents").json()
    assert len(incs) == 1
    assert len(incs[0]["alert_ids"]) == 5

    audit = c.get("/audit").json()
    actions = [e["action"] for e in audit["entries"]]
    assert actions.count("incident.auto_create") == 1
    assert actions.count("incident.auto_append") == 4


def test_repeat_alerts_for_different_tracks_create_separate_incidents(monkeypatch):
    c = _client(monkeypatch)
    for trk in ("trk-A", "trk-B", "trk-C"):
        c.post("/alerts", json={
            "track_id": trk, "camera_id": "cam-01",
            "severity": "high", "rule": "dwell", "score": 0.9,
        })
    incs = c.get("/incidents").json()
    assert len(incs) == 3
