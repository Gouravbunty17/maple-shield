from fastapi.testclient import TestClient

import app.main as main_mod


def test_cairn_health_endpoint():
    client = TestClient(main_mod.app)
    r = client.get("/cairn/health")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine"] == "CAIRN"
    assert body["engine_version"] == "2.0.0-dev"
    assert body["compatible"] is True
    assert isinstance(body["frames_processed"], int)
    assert "risk_config" in body
    assert "operator" not in body
