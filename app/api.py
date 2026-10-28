"""REST inference service (FastAPI).

Endpoints
  GET  /health               liveness + whether the model is loaded
  GET  /model-info           model name, version, classes, validation metrics
  POST /predict              one image -> label, confidence, top-k, optional Grad-CAM (base64 PNG)
  POST /predict/batch        up to 32 images in one call
  POST /feedback             attach the true label to a past request (live accuracy)
  GET  /monitoring/summary   traffic, confidence, latency, live accuracy
  GET  /monitoring/drift     data / prediction drift report vs. training reference
  GET  /metrics              Prometheus-format counters for Grafana
  DELETE /monitoring         clear the prediction log (demo reset)
  GET  /                     web UI (app/static/index.html)
  GET  /samples              random held-out test images for the UI (+ /samples/{id} serves one)
  GET  /model/report         test metrics, per-class report and training history
  GET  /artifacts/{name}     evaluation figures (confusion matrix, Grad-CAM gallery, ...)

Run:  uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

import base64
import csv
import io
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geovision.config import load_config, resolve  # noqa: E402
from geovision.data import image_stats  # noqa: E402
from geovision.inference import Predictor  # noqa: E402
from geovision.monitoring import DriftMonitor, PredictionStore  # noqa: E402

CFG = load_config(os.getenv("GEOVISION_CONFIG"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", resolve(CFG["paths"]["artifacts"]) / "best_model.pt"))
REFERENCE_PATH = Path(os.getenv("REFERENCE_PATH", resolve(CFG["data"]["manifest"]).parent / "reference_stats.json"))
DB_PATH = Path(os.getenv("MONITORING_DB", resolve(CFG["paths"]["monitoring_db"])))
MANIFEST_PATH = Path(os.getenv("MANIFEST_PATH", resolve(CFG["data"]["manifest"])))
ARTIFACTS_DIR = resolve(CFG["paths"]["artifacts"])
UI_PATH = ROOT / "app" / "static" / "index.html"
ARTIFACT_IMAGES = ("confusion_matrix.png", "gradcam_examples.png", "training_curves.png",
                   "reliability_diagram.png", "misclassified.png", "eda_samples.png", "eda_class_distribution.png")
MAX_BYTES = 10 * 1024 * 1024
MAX_BATCH = 32

state = {"predictor": None, "store": None, "drift": None, "samples": [],
         "counters": {"requests_total": 0, "errors_total": 0, "predictions_total": 0}, "latencies": []}


def _load_samples():
    """Held-out test images, so the UI can offer one-click examples the model never trained on."""
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    return [{"id": i, "class_name": r["class_name"], "path": r["path"]} for i, r in enumerate(rows)]


@asynccontextmanager
async def lifespan(_app):
    state["store"] = PredictionStore(DB_PATH)
    if MODEL_PATH.exists():
        state["predictor"] = Predictor(MODEL_PATH)
    if REFERENCE_PATH.exists():
        mcfg = CFG["monitoring"]
        state["drift"] = DriftMonitor(REFERENCE_PATH, mcfg["psi_warning"], mcfg["psi_alert"],
                                      mcfg["low_conf_threshold"])
    state["samples"] = _load_samples()
    yield


app = FastAPI(title="GeoVision Land-Use API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Feedback(BaseModel):
    request_id: str
    true_label: str


def _predictor():
    if state["predictor"] is None:
        raise HTTPException(503, f"Model not loaded - expected checkpoint at {MODEL_PATH}")
    return state["predictor"]


async def _read_image(file: UploadFile):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 10 MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, f"'{file.filename}' is not a valid image")


def _to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _log(predictor, img, filename, result):
    request_id = uuid.uuid4().hex
    state["store"].log(request_id, filename, predictor.meta.get("run_id"), result["label"],
                       result["confidence"], result["latency_ms"], image_stats(img))
    state["counters"]["predictions_total"] += 1
    state["latencies"] = (state["latencies"] + [result["latency_ms"]])[-1000:]
    return request_id


@app.middleware("http")
async def count_requests(request, call_next):
    state["counters"]["requests_total"] += 1
    response = await call_next(request)
    if response.status_code >= 400:
        state["counters"]["errors_total"] += 1
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["predictor"] is not None}


@app.get("/model-info")
def model_info():
    return _predictor().info


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = Query(3, ge=1, le=10),
                  explain: bool = Query(False)):
    predictor = _predictor()
    img = await _read_image(file)
    result = predictor.predict(img, top_k=top_k, explain=explain)
    result.pop("probabilities")
    if explain:
        result["heatmap_png_base64"] = _to_b64(result.pop("heatmap"))
    result["request_id"] = _log(predictor, img, file.filename, result)
    return result


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...), top_k: int = Query(3, ge=1, le=10)):
    predictor = _predictor()
    if len(files) > MAX_BATCH:
        raise HTTPException(413, f"At most {MAX_BATCH} images per batch")
    images = [await _read_image(f) for f in files]
    t0 = time.perf_counter()
    results = predictor.predict_batch(images, top_k=top_k)
    per_image_ms = (time.perf_counter() - t0) * 1000 / len(images)
    out = []
    for f, img, r in zip(files, images, results):
        r.pop("probabilities")
        r["latency_ms"] = per_image_ms
        r["request_id"] = _log(predictor, img, f.filename, r)
        out.append({"filename": f.filename, **r})
    return {"results": out}


@app.post("/feedback")
def feedback(fb: Feedback):
    if state["predictor"] and fb.true_label not in state["predictor"].classes:
        raise HTTPException(422, f"Unknown label '{fb.true_label}'")
    if not state["store"].add_feedback(fb.request_id, fb.true_label):
        raise HTTPException(404, "request_id not found")
    return {"status": "recorded"}


@app.get("/monitoring/summary")
def monitoring_summary():
    return state["store"].summary()


@app.get("/monitoring/recent")
def monitoring_recent(n: int = Query(100, ge=1, le=5000)):
    return state["store"].recent(n)


@app.get("/monitoring/drift")
def monitoring_drift():
    if state["drift"] is None:
        raise HTTPException(503, "Reference statistics not found - run scripts/prepare_data.py")
    return state["drift"].report(state["store"].recent(CFG["monitoring"]["window"]))


@app.delete("/monitoring")
def monitoring_reset():
    return {"deleted": state["store"].clear()}


@app.get("/", include_in_schema=False)
def ui():
    if not UI_PATH.exists():
        raise HTTPException(404, "UI not found")
    return FileResponse(UI_PATH, media_type="text/html", headers={"Cache-Control": "no-cache"})


@app.get("/samples")
def samples(n: int = Query(10, ge=1, le=200), per_class: bool = Query(False), cls: str | None = None):
    import random

    pool = [s for s in state["samples"] if cls is None or s["class_name"] == cls]
    if per_class:
        by_class = {}
        for s in pool:
            by_class.setdefault(s["class_name"], []).append(s)
        picked = [random.choice(v) for _, v in sorted(by_class.items())]
    else:
        picked = random.sample(pool, min(n, len(pool)))
    return [{"id": s["id"], "class_name": s["class_name"], "url": f"/samples/{s['id']}"} for s in picked]


@app.get("/samples/{sample_id}")
def sample_image(sample_id: int):
    if not 0 <= sample_id < len(state["samples"]):
        raise HTTPException(404, "Sample not found")
    return FileResponse(state["samples"][sample_id]["path"])


@app.get("/model/report")
def model_report():
    metrics_path, history_path = ARTIFACTS_DIR / "test_metrics.json", ARTIFACTS_DIR / "history.csv"
    test_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    history = []
    if history_path.exists():
        with open(history_path, newline="") as f:
            history = [{k: (v if k == "backbone_frozen" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
    images = [name for name in ARTIFACT_IMAGES if (ARTIFACTS_DIR / name).exists()]
    return {"test_metrics": test_metrics, "history": history, "images": images}


@app.get("/artifacts/{name}")
def artifact_image(name: str):
    if name not in ARTIFACT_IMAGES or not (ARTIFACTS_DIR / name).exists():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(ARTIFACTS_DIR / name, media_type="image/png")


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lines = []
    for name, value in state["counters"].items():
        lines += [f"# TYPE geovision_{name} counter", f"geovision_{name} {value}"]
    lat = sorted(state["latencies"])
    if lat:
        lines += ["# TYPE geovision_latency_ms summary",
                  f'geovision_latency_ms{{quantile="0.5"}} {lat[len(lat) // 2]:.3f}',
                  f'geovision_latency_ms{{quantile="0.95"}} {lat[int(len(lat) * 0.95) - 1]:.3f}']
    lines.append(f"geovision_model_loaded {int(state['predictor'] is not None)}")
    return "\n".join(lines) + "\n"
