import importlib
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def client(checkpoint, manifest, tmp_path, monkeypatch):
    manifest_path, *_ = manifest
    monkeypatch.setenv("MODEL_PATH", str(checkpoint))
    monkeypatch.setenv("REFERENCE_PATH", str(manifest_path.parent / "reference_stats.json"))
    monkeypatch.setenv("MONITORING_DB", str(tmp_path / "pred.db"))
    monkeypatch.setenv("MANIFEST_PATH", str(manifest_path))
    import app.api as api
    importlib.reload(api)
    with TestClient(api.app) as c:
        yield c


def _png(color=(60, 120, 60)):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="PNG")
    return buf.getvalue()


def test_health_and_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    assert client.get("/model-info").json()["classes"] == ["Forest", "Industrial", "SeaLake"]


def test_predict_with_explanation(client):
    r = client.post("/predict?explain=true&top_k=2", files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ["Forest", "Industrial", "SeaLake"]
    assert len(body["top_k"]) == 2 and body["heatmap_png_base64"]
    fb = client.post("/feedback", json={"request_id": body["request_id"], "true_label": "Forest"})
    assert fb.status_code == 200


def test_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_batch_and_monitoring(client):
    files = [("files", (f"{i}.png", _png((i * 20, 100, 50)), "image/png")) for i in range(5)]
    assert len(client.post("/predict/batch", files=files).json()["results"]) == 5
    assert client.get("/monitoring/summary").json()["total_predictions"] == 5
    assert client.get("/monitoring/drift").json()["status"] == "insufficient_data"
    assert "geovision_predictions_total 5" in client.get("/metrics").text


def test_web_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "GeoVision" in r.text


def test_samples_report_and_artifacts(client):
    samples = client.get("/samples?per_class=true").json()
    assert sorted(s["class_name"] for s in samples) == ["Forest", "Industrial", "SeaLake"]
    assert client.get(samples[0]["url"]).status_code == 200
    assert client.get("/samples/999999").status_code == 404
    assert client.get("/model/report").status_code == 200
    assert client.get("/artifacts/best_model.pt").status_code == 404


def test_reset_monitoring(client):
    client.post("/predict", files={"file": ("a.png", _png(), "image/png")})
    assert client.delete("/monitoring").json()["deleted"] == 1
    assert client.get("/monitoring/summary").json()["total_predictions"] == 0
