import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pymatgen.core import Lattice, Structure

from app import config, main, ood

MOS2 = {"formula": "MoS2", "nat": 3, "a": 3.19, "b": 3.19, "gamma": 120, "thickness": 3.13, "spg_number": 187}


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


def _fake_reference(tmp_path, cols, scale_shift=0.0):
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(500, len(cols))).astype("float32")
    p = tmp_path / "ref.npz"
    np.savez_compressed(p, Z=Z, median=np.zeros(len(cols)), mean=np.zeros(len(cols)), scale=np.ones(len(cols)), columns=np.array(cols))
    return p


def test_health_and_models(client):
    assert client.get("/api/health").json()["status"] == "ok"
    m = client.get("/api/models").json()
    assert "taskA_Y2D" in m["models"] and m["models"]["taskA_Y2D"]["n_features"] == 141


def test_predict_returns_positive_stiffness_with_error_context(client):
    r = client.post("/api/predict", json=MOS2)
    assert r.status_code == 200
    j = r.json()
    assert j["stiffness"]["value"] > 0 and j["stiffness"]["unit"] == "N/m"
    assert abs(j["stiffness"]["error_known_family"] - 13.8) < 0.1 and abs(j["stiffness"]["error_new_family"] - 19.9) < 0.1
    assert -1 <= j["poisson"]["value"] <= 1 and j["poisson"]["confidence"] == "low"
    assert 0 <= j["stiffness"]["position"]["position"] <= 1


def test_prediction_matches_the_saved_model_directly(client):
    import joblib
    from app.features import featurize
    m = joblib.load(config.MODELS_DIR / "taskA_Y2D.joblib")
    X = featurize([MOS2], m["features"])
    direct = float(np.exp(m["pipeline"].predict(X))[0])
    assert abs(client.post("/api/predict", json=MOS2).json()["stiffness"]["value"] - direct) < 1e-9


def test_invalid_inputs_are_rejected_with_422(client):
    assert client.post("/api/predict", json={**MOS2, "formula": "Xx9"}).status_code == 422
    assert client.post("/api/predict", json={**MOS2, "a": -1}).status_code == 422
    assert client.post("/api/predict", json={**MOS2, "gamma": 170}).status_code == 422
    assert client.post("/api/predict", json={"formula": "MoS2"}).status_code == 422


def test_structure_upload_derives_geometry(client):
    lat = Lattice.hexagonal(3.19, 20.0)
    s = Structure(lat, ["Mo", "S", "S"], [[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5 + 0.0783], [2 / 3, 1 / 3, 0.5 - 0.0783]])
    r = client.post("/api/predict/structure", files={"file": ("mos2.cif", s.to(fmt="cif").encode(), "text/plain")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["input"]["nat"] == 3 and abs(j["input"]["a"] - 3.19) < 1e-6 and abs(j["input"]["gamma"] - 120) < 1e-6
    assert abs(j["input"]["thickness"] - 3.13) < 0.02 and j["input"]["spg_number"] == 187
    bad = client.post("/api/predict/structure", files={"file": ("junk.cif", b"not a structure", "text/plain")})
    assert bad.status_code == 422


def test_structure_without_vacuum_gets_a_warning(client):
    s = Structure(Lattice.hexagonal(3.19, 5.0), ["Mo", "S", "S"], [[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.62], [2 / 3, 1 / 3, 0.38]])
    j = client.post("/api/predict/structure", files={"file": ("x.cif", s.to(fmt="cif").encode(), "text/plain")}).json()
    assert any("vacuum" in w for w in j["warnings"])


def test_batch_reports_good_and_bad_rows(client):
    csv = "formula,nat,a,b,gamma,thickness,spg_number\nMoS2,3,3.19,3.19,120,3.13,187\nBogus,3,3.19,3.19,120,3.13,187\nWSe2,3,3.28,3.28,120,3.36,187\n"
    r = client.post("/api/predict/batch", files={"file": ("b.csv", csv.encode(), "text/csv")})
    j = r.json()
    assert r.status_code == 200 and j["n_ok"] == 2 and j["n_errors"] == 1 and j["errors"][0]["line"] == 3
    assert client.post("/api/predict/batch", files={"file": ("b.csv", b"formula,a\nMoS2,3\n", "text/csv")}).status_code == 422


def test_distance_check_levels_with_a_reference(client, tmp_path, monkeypatch):
    cols = main.S.columns
    ref = ood.Reference(_fake_reference(tmp_path, cols))
    monkeypatch.setattr(main.S, "reference", ref)
    j = client.post("/api/predict", json=MOS2).json()
    assert j["domain"]["available"] and j["domain"]["level"] in ("typical", "somewhat_unusual", "unusual")
    known, new = j["stiffness"]["error_known_family"], j["stiffness"]["error_new_family"]
    expected = {"typical": known, "unusual": new, "somewhat_unusual": (known + new) / 2}[j["domain"]["level"]]
    assert abs(j["stiffness"]["expected_abs_error"] - expected) < 1e-9


def test_assess_thresholds():
    assert ood.assess(1.0)["level"] == "typical"
    assert ood.assess((config.DIST_TYPICAL + config.DIST_UNUSUAL) / 2)["level"] == "somewhat_unusual"
    assert ood.assess(config.DIST_UNUSUAL + 0.1)["level"] == "unusual"
    assert ood.assess(None)["available"] is False
