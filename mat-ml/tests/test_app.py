from fastapi.testclient import TestClient

from app.main import app

c = TestClient(app)


def test_health_and_static():
    assert c.get("/api/health").json()["status"] == "ok"
    assert "2D-materials" in c.get("/").text


def test_results_endpoints():
    h = c.get("/api/headline").json()
    assert any(r["analysis"] == "TaskA|Y2D" and r["scheme"] == "family" for r in h)
    assert len(c.get("/api/panel/c2db").json()) == 12 and len(c.get("/api/panel/jarvis").json()) == 10
    assert c.get("/api/panel/nope").status_code == 404
    assert "gnn_seeds" in c.get("/api/robustness").json()
    assert c.get("/api/figures").json()


def test_predict_validation_and_value():
    bad = {"formula": "not a formula!", "a": 3, "b": 3, "gamma": 120, "area_per_atom": 3, "nat": 3, "thickness": 3, "spg_number": 187}
    assert c.post("/api/predict/y2d", json=bad).status_code == 422
    ok = {**bad, "formula": "MoS2", "a": 3.19, "b": 3.19, "area_per_atom": 2.93, "thickness": 3.13}
    r = c.post("/api/predict/y2d", json=ok)
    assert r.status_code == 200 and 20 < r.json()["Y2D_N_per_m"] < 400
