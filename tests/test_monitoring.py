import json

import numpy as np

from geovision.monitoring import FEATURES, DriftMonitor, PredictionStore, psi


def test_psi_same_vs_shifted():
    rng = np.random.default_rng(0)
    ref = rng.normal(0.5, 0.1, 5000)
    assert psi(ref, rng.normal(0.5, 0.1, 1000)) < 0.1
    assert psi(ref, rng.normal(0.25, 0.1, 1000)) > 0.25


def _stats(b):
    return {"brightness": b, "contrast": 0.1, "r_mean": b, "g_mean": b, "b_mean": b}


def test_store_and_drift(tmp_path):
    rng = np.random.default_rng(1)
    ref = {"features": {f: rng.normal(0.4, 0.05, 2000).tolist() for f in FEATURES},
           "class_distribution": {"A": 0.5, "B": 0.5}}
    ref["features"]["contrast"] = rng.normal(0.1, 0.01, 2000).tolist()
    ref_path = tmp_path / "ref.json"
    ref_path.write_text(json.dumps(ref))

    store = PredictionStore(tmp_path / "p.db")
    monitor = DriftMonitor(ref_path)
    assert monitor.report(store.recent())["status"] == "insufficient_data"

    for i in range(100):  # in-distribution traffic
        store.log(f"r{i}", "x.jpg", "v1", "A" if i % 2 else "B", 0.95, 5.0, _stats(float(rng.normal(0.4, 0.05))))
    assert monitor.report(store.recent())["data_drift"]["brightness"]["status"] == "ok"

    for i in range(100, 200):  # much darker images, all predicted "A"
        store.log(f"r{i}", "x.jpg", "v1", "A", 0.4, 5.0, _stats(float(rng.normal(0.15, 0.05))))
    report = monitor.report(store.recent(100))
    assert report["status"] == "alert"
    assert report["data_drift"]["brightness"]["status"] == "alert"

    assert store.add_feedback("r1", "A")
    assert store.summary()["live_accuracy"] == 1.0
