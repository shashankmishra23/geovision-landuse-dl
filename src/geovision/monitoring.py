"""Post-deployment monitoring.

1. PredictionStore - every request is logged to SQLite (input stats, prediction, confidence, latency).
2. DriftMonitor    - compares the most recent window of traffic against the training reference:
     * data drift       : PSI on image brightness / contrast / channel means
     * prediction drift : PSI on predicted-class distribution vs. training label distribution
     * model health     : mean confidence and share of low-confidence predictions
   Status is "ok" / "warning" / "alert" using standard PSI thresholds (0.1 / 0.25).
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FEATURES = ("brightness", "contrast", "r_mean", "g_mean", "b_mean")


class PredictionStore:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, request_id TEXT, filename TEXT, model_version TEXT,
                label TEXT, confidence REAL, latency_ms REAL,
                brightness REAL, contrast REAL, r_mean REAL, g_mean REAL, b_mean REAL,
                feedback_label TEXT)""")

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def log(self, request_id, filename, model_version, label, confidence, latency_ms, stats):
        row = (datetime.now(timezone.utc).isoformat(timespec="seconds"), request_id, filename, model_version,
               label, confidence, latency_ms, *(stats[f] for f in FEATURES))
        with self._lock, self._conn() as c:
            c.execute("""INSERT INTO predictions (ts, request_id, filename, model_version, label, confidence,
                         latency_ms, brightness, contrast, r_mean, g_mean, b_mean)
                         VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", row)

    def add_feedback(self, request_id, true_label):
        """Ground-truth feedback (e.g. from an analyst) enables live accuracy tracking."""
        with self._lock, self._conn() as c:
            cur = c.execute("UPDATE predictions SET feedback_label=? WHERE request_id=?", (true_label, request_id))
            return cur.rowcount > 0

    def clear(self):
        """Delete all logged predictions (used to reset a demo). Returns the number removed."""
        with self._lock, self._conn() as c:
            return c.execute("DELETE FROM predictions").rowcount

    def recent(self, n=500):
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            rows = c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [dict(r) for r in rows]

    def summary(self):
        with self._conn() as c:
            total, mean_conf, mean_lat = c.execute(
                "SELECT COUNT(*), AVG(confidence), AVG(latency_ms) FROM predictions").fetchone()
            lat = [r[0] for r in c.execute("SELECT latency_ms FROM predictions ORDER BY id DESC LIMIT 1000")]
            by_class = dict(c.execute("SELECT label, COUNT(*) FROM predictions GROUP BY label").fetchall())
            fb = c.execute("""SELECT COUNT(*), SUM(label = feedback_label) FROM predictions
                              WHERE feedback_label IS NOT NULL""").fetchone()
        return {
            "total_predictions": total,
            "mean_confidence": mean_conf,
            "mean_latency_ms": mean_lat,
            "p95_latency_ms": float(np.percentile(lat, 95)) if lat else None,
            "predictions_by_class": by_class,
            "feedback_count": fb[0],
            "live_accuracy": (fb[1] / fb[0]) if fb[0] else None,
        }


def auto_bins(n):
    """About 25 samples per bin, between 4 and 10 bins."""
    return int(np.clip(n // 25, 4, 10))


def psi(reference, current, bins=None, eps=1e-4):
    """Population Stability Index with quantile bins taken from the reference distribution.

    PSI has a sampling-noise floor of roughly (bins - 1) / len(current), so small windows get fewer
    bins and DriftMonitor subtracts that floor before judging drift.
    """
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    if bins is None:
        bins = auto_bins(len(current))
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference) + eps
    cur_pct = np.histogram(current, edges)[0] / len(current) + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def categorical_psi(ref_dist, cur_counts, eps=1e-4):
    total = max(sum(cur_counts.values()), 1)
    score = 0.0
    for cls, p_ref in ref_dist.items():
        p_cur = cur_counts.get(cls, 0) / total
        score += (p_cur - p_ref) * np.log((p_cur + eps) / (p_ref + eps))
    return float(score)


class DriftMonitor:
    def __init__(self, reference_path, psi_warning=0.1, psi_alert=0.25, low_conf_threshold=0.6, min_samples=30):
        with open(reference_path) as f:
            self.reference = json.load(f)
        self.psi_warning, self.psi_alert = psi_warning, psi_alert
        self.low_conf_threshold = low_conf_threshold
        self.min_samples = min_samples

    def _status(self, value):
        return "alert" if value >= self.psi_alert else "warning" if value >= self.psi_warning else "ok"

    def report(self, records):
        n = len(records)
        if n < self.min_samples:
            return {"status": "insufficient_data", "window_size": n, "min_samples": self.min_samples}

        # Status is judged on noise-corrected PSI: raw PSI minus its expected value under no drift,
        # (bins - 1) / n, so small windows of in-distribution traffic do not raise false warnings.
        feature_floor = (auto_bins(n) - 1) / n
        features = {}
        for f in FEATURES:
            raw = psi(self.reference["features"][f], [r[f] for r in records])
            score = max(0.0, raw - feature_floor)
            features[f] = {"psi": round(score, 4), "psi_raw": round(raw, 4), "status": self._status(score),
                           "reference_mean": round(float(np.mean(self.reference["features"][f])), 4),
                           "current_mean": round(float(np.mean([r[f] for r in records])), 4)}

        counts = {}
        for r in records:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        pred_raw = categorical_psi(self.reference["class_distribution"], counts)
        pred_psi = max(0.0, pred_raw - (len(self.reference["class_distribution"]) - 1) / n)

        # Reported for context only: this model is deliberately under-confident (MixUp + label
        # smoothing), so a fixed low-confidence threshold would flag normal traffic.
        conf = np.array([r["confidence"] for r in records])
        low_conf_rate = float((conf < self.low_conf_threshold).mean())

        statuses = [v["status"] for v in features.values()] + [self._status(pred_psi)]
        overall = "alert" if "alert" in statuses else "warning" if "warning" in statuses else "ok"

        return {
            "status": overall,
            "window_size": n,
            "data_drift": features,
            "prediction_drift": {"psi": round(pred_psi, 4), "psi_raw": round(pred_raw, 4),
                                 "status": self._status(pred_psi)},
            "model_health": {"mean_confidence": round(float(conf.mean()), 4),
                             "low_confidence_rate": round(low_conf_rate, 4),
                             "low_confidence_threshold": self.low_conf_threshold},
            "recommendation": {
                "ok": "No action needed.",
                "warning": "Investigate input sources; collect and label recent samples.",
                "alert": "Significant drift - label recent data and retrain (python -m geovision.train).",
            }[overall],
        }
