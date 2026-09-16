# Stage 5 — Streamlit Frontend Report

**Project:** Wine Quality Prediction MLOps Pipeline (PMLDL Assignment 1)
**Stage:** 5 of 8 — `code/deployment/app/app.py`
**Date:** 2026-09-16
**Status:** ✅ Completed — headless smoke test passed (health 200 `ok`, page 200)

---

## 1. Objective

Provide a UI so a user can set wine physicochemical properties, send them to
the Stage 4 FastAPI endpoint and see the predicted quality class with its
probability.

## 2. Implementation

- `API_URL` from the environment (default `http://localhost:8000`;
  `http://api:8000` inside Docker Compose) — the UI never hardcodes hosts.
- 11 numeric inputs as `st.slider`s with ranges taken from the Stage 1
  dataset statistics (min/max) and defaults typical for red wine; step sizes
  matched to feature granularity (e.g. `density` step 0.0001).
- `wine_type` via `st.selectbox` (`red`/`white`).
- Sidebar performs a live `/health` check and shows backend status +
  request-flow description.
- "Predict quality" button → `POST /predict` with dataset-named JSON keys
  (exactly the Stage 4 contract).
- Response handling: 200 → prediction banner + `P(Good)` metric + probability
  bar; 422 → per-field validation messages; connection/other errors →
  human-readable `st.error` messages (never a raw traceback).

## 3. Validation Results (local, headless)

| Check | Result |
|-------|--------|
| `GET /_stcore/health` | 200, body `ok` |
| `GET /` | 200 (Streamlit SPA shell served; UI renders client-side) |

The true end-to-end UI→API interaction is verified in Stage 6, where a Python
client **inside the app container** posts the exact payload the UI builds to
`http://api:8000/predict` over the Compose network.

## 4. Security Considerations

- No secrets, no local persistence; the app is a thin stateless client.
- All input sanitation is delegated to the API's Pydantic schemas (single
  validation point).
- Server binds to `0.0.0.0` only inside the container; host access is mapped
  by Docker Compose.

## 5. Limitations & Notes

- Slider ranges cap at the observed dataset min/max — values outside the
  historical range must be supplied via the API directly (documented).
- No session history/plotting (kept intentionally minimal per assignment).
