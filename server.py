"""
server.py — AstanaSim ML inference server
==========================================
FastAPI microservice that loads the trained XGBoost models and exposes:
  POST /predict       — next-year forecast for a given city-state
  GET  /importances   — SHAP feature importances (the ML insight panel)
  GET  /meta          — training diagnostics
  GET  /health        — liveness check

Run:
    python server.py             # port 8000 (default)
    python server.py --port 8080

Frontend calls this via mlClient.ts (add the URL as VITE_ML_API_URL env var).
Deploying: any Python host works — Railway.app free tier is one click.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from xgboost import XGBRegressor

app = FastAPI(title="AstanaSim ML", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock this down to your lovable.app domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models at startup ────────────────────────────────────────────────────
MODELS: dict[str, XGBRegressor] = {}
MODEL_DIR = Path(__file__).parent / "models"

def load_models():
    for name in ["aqi", "grpPerCapita", "population", "inflation"]:
        path = MODEL_DIR / f"{name}_model.json"
        if path.exists():
            m = XGBRegressor()
            m.load_model(str(path))
            MODELS[name] = m
    print(f"Loaded {len(MODELS)} models: {list(MODELS.keys())}")

@app.on_event("startup")
async def startup():
    load_models()

# ── Feature builder (mirrors train_model.py exactly) ─────────────────────────
FEATURE_ORDER = [
    "city_id", "year", "lag_population", "lag_grp_usd", "lag_aqi",
    "lag_inflation", "lag_unemployment", "lag_pm25", "lag_net_migration_k",
    "lag_housing_m2", "transit_score", "heating_coal_share",
    "pop_density_proxy", "grp_growth_lag", "coal_x_winter", "transit_x_pop",
]

CITY_IDS = {"astana": 0, "almaty": 1, "shymkent": 2, "aktobe": 3}

def build_features(req: "PredictRequest") -> np.ndarray:
    city_id = CITY_IDS.get(req.city.lower(), 0)
    features = [
        city_id,
        req.current_year,
        req.population,
        req.grp_usd,
        req.aqi,
        req.inflation,
        req.unemployment,
        req.pm25,
        req.net_migration_k,
        req.housing_m2,
        req.transit_score,
        req.heating_coal_share,
        req.population / 810.0,                        # pop_density_proxy
        req.grp_growth_lag,
        req.heating_coal_share * req.pm25,             # coal_x_winter interaction
        req.transit_score * req.population / 1e6,      # transit_x_pop interaction
    ]
    return np.array(features).reshape(1, -1)

# ── Request / response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    city: str = "astana"
    current_year: int
    population: float
    grp_usd: float
    aqi: float
    inflation: float
    unemployment: float
    pm25: float
    net_migration_k: float = 0.0
    housing_m2: float = 2.0
    transit_score: float = 26.0
    heating_coal_share: float = 0.81
    grp_growth_lag: float = 0.0

class ForecastResult(BaseModel):
    next_year: int
    predicted_aqi: float
    predicted_grp_usd: float
    predicted_population: float
    predicted_inflation: float
    # Confidence ranges (±1 sigma from LOO-CV RMSE)
    aqi_range: list[float]
    grp_range: list[float]
    population_range: list[float]
    inflation_range: list[float]

# LOO-CV RMSE from training — used to produce confidence ranges
RMSE = {
    "aqi": 9.25,
    "grpPerCapita": 897.19,
    "population": 66540.62,
    "inflation": 0.56,
}

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(MODELS.keys())}

@app.get("/meta")
def meta():
    path = MODEL_DIR / "model_meta.json"
    if not path.exists():
        raise HTTPException(404, "model_meta.json not found — run train_model.py first")
    return json.loads(path.read_text())

@app.get("/importances")
def importances():
    path = MODEL_DIR / "feature_importances.json"
    if not path.exists():
        raise HTTPException(404, "feature_importances.json not found — run train_model.py first")
    return json.loads(path.read_text())

@app.post("/predict", response_model=ForecastResult)
def predict(req: PredictRequest):
    if not MODELS:
        raise HTTPException(503, "Models not loaded — run train_model.py first")

    X = build_features(req)
    preds = {name: float(model.predict(X)[0]) for name, model in MODELS.items()}

    return ForecastResult(
        next_year=req.current_year + 1,
        predicted_aqi=round(preds["aqi"], 1),
        predicted_grp_usd=round(preds["grpPerCapita"], 0),
        predicted_population=round(preds["population"], 0),
        predicted_inflation=round(preds["inflation"], 2),
        aqi_range=[round(preds["aqi"] - RMSE["aqi"], 1), round(preds["aqi"] + RMSE["aqi"], 1)],
        grp_range=[round(preds["grpPerCapita"] - RMSE["grpPerCapita"], 0), round(preds["grpPerCapita"] + RMSE["grpPerCapita"], 0)],
        population_range=[round(preds["population"] - RMSE["population"], 0), round(preds["population"] + RMSE["population"], 0)],
        inflation_range=[round(preds["inflation"] - RMSE["inflation"], 2), round(preds["inflation"] + RMSE["inflation"], 2)],
    )

@app.post("/predict_horizon")
def predict_horizon(req: PredictRequest, years: int = 5):
    """Multi-step forecast by chaining single-step predictions."""
    if not MODELS:
        raise HTTPException(503, "Models not loaded")
    if years > 14:
        raise HTTPException(400, "Max 14 years")

    X = build_features(req)
    results = []

    state = req.model_copy()
    for step in range(years):
        X = build_features(state)
        preds = {name: float(model.predict(X)[0]) for name, model in MODELS.items()}

        grp_growth = (preds["grpPerCapita"] - state.grp_usd) / max(1, state.grp_usd) * 100

        results.append({
            "year": state.current_year + 1,
            "aqi": round(preds["aqi"], 1),
            "grp_usd": round(preds["grpPerCapita"], 0),
            "population": round(preds["population"], 0),
            "inflation": round(preds["inflation"], 2),
            "uncertainty_multiplier": round(1 + step * 0.3, 2),  # grows with horizon
        })

        # Advance state for next step
        state.current_year += 1
        state.aqi = preds["aqi"]
        state.grp_usd = preds["grpPerCapita"]
        state.population = preds["population"]
        state.inflation = preds["inflation"]
        state.grp_growth_lag = grp_growth
        state.pm25 = preds["aqi"] / 3.8  # rough AQI→PM2.5 inversion

    return {"city": req.city, "horizon": results}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("server:app", host="0.0.0.0", port=args.port, reload=True)
