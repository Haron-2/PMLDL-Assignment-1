"""Stage 5 - Streamlit frontend for the Wine Quality Prediction API.

Collects 11 physicochemical features + wine type, sends them to the FastAPI
backend (Stage 4) and displays the prediction with class probability.

Environment:
    API_URL  backend base URL (default http://localhost:8000;
             set to http://api:8000 inside Docker Compose)

Run locally (from project root; requires the API running on :8000):
    .venv/Scripts/python.exe -m streamlit run code/deployment/app/app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

# label -> (min, max, default, step); defaults are typical red-wine values
NUMERIC_FEATURES = {
    "fixed acidity":        (3.8, 15.9, 7.2, 0.1),
    "volatile acidity":     (0.08, 1.58, 0.50, 0.01),
    "citric acid":          (0.0, 1.66, 0.30, 0.01),
    "residual sugar":       (0.6, 65.8, 5.0, 0.1),
    "chlorides":            (0.009, 0.6, 0.08, 0.001),
    "free sulfur dioxide":  (2.0, 289.0, 25.0, 1.0),
    "total sulfur dioxide": (9.0, 440.0, 100.0, 1.0),
    "density":              (0.9871, 1.0390, 0.9960, 0.0001),
    "pH":                   (2.72, 4.01, 3.30, 0.01),
    "sulphates":            (0.22, 2.0, 0.55, 0.01),
    "alcohol":              (8.0, 14.9, 10.5, 0.1),
}

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")
st.title("🍷 Wine Quality Predictor")


def api_health() -> tuple[bool, str]:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200 and r.json().get("model_loaded"):
            return True, f"model loaded ({r.json().get('model_type', '?')})"
        return False, f"unexpected health response: HTTP {r.status_code}"
    except requests.RequestException as exc:
        return False, str(exc)


healthy, _ = api_health()

# ---------------- inputs ----------------
wine_type = st.selectbox("wine type", ["red", "white"])

payload: dict = {"wine_type": wine_type}
features = list(NUMERIC_FEATURES.items())
for i in range(0, len(features), 3):
    cols = st.columns(3)
    for col, (label, (lo, hi, default, step)) in zip(cols, features[i : i + 3]):
        with col:
            payload[label] = st.slider(label, min_value=float(lo), max_value=float(hi),
                                       value=float(default), step=float(step),
                                       format="%.4f" if step < 0.01 else "%.2f")

predict_clicked = st.button("Predict quality", type="primary", use_container_width=True)

# ---------------- result ----------------
if predict_clicked:
    if not healthy:
        st.error("API is not healthy or unreachable — start the backend and try again.")
    else:
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
        else:
            if resp.status_code == 200:
                body = resp.json()
                prediction_col, probability_col = st.columns(2)
                with prediction_col:
                    st.metric("Prediction", body["label"])
                with probability_col:
                    st.metric("Probability of Good quality",
                              f"{body['probability_good']:.1%}")
            elif resp.status_code == 422:
                st.error("Input validation failed (HTTP 422):")
                for err in resp.json().get("detail", []):
                    loc = " → ".join(str(p) for p in err.get("loc", []))
                    st.write(f"- `{loc}`: {err.get('msg')}")
            else:
                st.error(f"API error HTTP {resp.status_code}: {resp.text}")
