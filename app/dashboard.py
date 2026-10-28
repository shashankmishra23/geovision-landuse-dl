"""Streamlit front-end: classify satellite tiles and watch model health. Talks to the FastAPI service.

Run:  streamlit run app/dashboard.py   (API must be running; set GEOVISION_API to change its URL)
"""

import base64
import io
import os

import pandas as pd
import requests
import streamlit as st
from PIL import Image

API = os.getenv("GEOVISION_API", "http://127.0.0.1:8000")
STATUS_COLOR = {"ok": "🟢", "warning": "🟠", "alert": "🔴", "insufficient_data": "⚪"}

st.set_page_config(page_title="GeoVision", page_icon="🛰️", layout="wide")
st.title("🛰️ GeoVision — Satellite Land-Use Classification")


def api_get(path):
    try:
        r = requests.get(f"{API}{path}", timeout=10)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


health = api_get("/health")
if not health:
    st.error(f"Cannot reach the API at {API}. Start it with `uvicorn app.api:app --port 8000`.")
    st.stop()
if not health["model_loaded"]:
    st.warning("API is up but no trained model is loaded. Train one with `python -m geovision.train`.")

info = api_get("/model-info") or {}
with st.sidebar:
    st.header("Model")
    st.write(f"**Architecture:** {info.get('model_name', '-')}")
    st.write(f"**Run:** {info.get('run_id', '-')}")
    if info.get("val_metrics"):
        st.metric("Validation accuracy", f"{info['val_metrics']['accuracy']:.2%}")
        st.metric("Validation macro-F1", f"{info['val_metrics']['macro_f1']:.3f}")
    st.caption(f"Classes: {', '.join(info.get('classes', []))}")

tab_predict, tab_batch, tab_monitor = st.tabs(["Classify", "Batch", "Monitoring"])

with tab_predict:
    up = st.file_uploader("Upload a satellite image tile", type=["jpg", "jpeg", "png", "tif", "tiff"])
    explain = st.toggle("Show Grad-CAM explanation", value=True)
    if up:
        r = requests.post(f"{API}/predict", params={"top_k": 5, "explain": explain},
                          files={"file": (up.name, up.getvalue(), up.type)}, timeout=60)
        if not r.ok:
            st.error(r.json().get("detail", r.text))
        else:
            res = r.json()
            c1, c2, c3 = st.columns([1, 1, 1.3])
            c1.image(Image.open(io.BytesIO(up.getvalue())), caption="Input", use_container_width=True)
            if explain:
                c2.image(base64.b64decode(res["heatmap_png_base64"]), caption="Grad-CAM", use_container_width=True)
            c3.metric("Prediction", res["label"], f"{res['confidence']:.1%} confidence")
            if res["confidence"] < 0.6:
                c3.warning("Low confidence — consider manual review.")
            c3.bar_chart(pd.DataFrame(res["top_k"]).set_index("label"))
            c3.caption(f"Latency {res['latency_ms']:.1f} ms · request {res['request_id'][:8]}")

            with st.expander("Was this wrong? Send the correct label"):
                true = st.selectbox("True label", info.get("classes", []))
                if st.button("Submit feedback"):
                    fb = requests.post(f"{API}/feedback", json={"request_id": res["request_id"], "true_label": true})
                    st.success("Thanks — recorded.") if fb.ok else st.error(fb.text)

with tab_batch:
    ups = st.file_uploader("Upload up to 32 tiles", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if ups and st.button("Classify all"):
        r = requests.post(f"{API}/predict/batch", files=[("files", (u.name, u.getvalue(), u.type)) for u in ups],
                          timeout=120)
        if r.ok:
            df = pd.DataFrame(r.json()["results"])[["filename", "label", "confidence"]]
            st.dataframe(df, use_container_width=True)
            st.bar_chart(df["label"].value_counts())
            st.download_button("Download CSV", df.to_csv(index=False), "predictions.csv")
        else:
            st.error(r.text)

with tab_monitor:
    if st.button("Refresh"):
        st.rerun()
    summary = api_get("/monitoring/summary") or {}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predictions served", summary.get("total_predictions", 0))
    m2.metric("Mean confidence", f"{summary['mean_confidence']:.1%}" if summary.get("mean_confidence") else "-")
    m3.metric("p95 latency", f"{summary['p95_latency_ms']:.1f} ms" if summary.get("p95_latency_ms") else "-")
    m4.metric("Live accuracy (feedback)",
              f"{summary['live_accuracy']:.1%}" if summary.get("live_accuracy") is not None else "-",
              f"{summary.get('feedback_count', 0)} labelled")

    drift = api_get("/monitoring/drift")
    if drift:
        st.subheader(f"{STATUS_COLOR.get(drift['status'], '')} Drift status: {drift['status'].upper()}")
        if drift["status"] == "insufficient_data":
            st.info(f"Need at least {drift['min_samples']} predictions (have {drift['window_size']}).")
        else:
            st.write(drift["recommendation"])
            rows = [{"feature": k, **v} for k, v in drift["data_drift"].items()]
            rows.append({"feature": "predicted class mix", **drift["prediction_drift"]})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.caption("PSI < 0.1 stable · 0.1–0.25 moderate shift · > 0.25 significant shift")

    recent = api_get("/monitoring/recent?n=500") or []
    if recent:
        df = pd.DataFrame(recent).sort_values("id")
        st.subheader("Confidence over time")
        st.line_chart(df.set_index("id")["confidence"])
        st.subheader("Predicted class distribution")
        st.bar_chart(df["label"].value_counts())
        st.subheader("Recent requests")
        st.dataframe(df.sort_values("id", ascending=False)
                     [["ts", "filename", "label", "confidence", "latency_ms", "feedback_label"]].head(50),
                     use_container_width=True)
