"""
train_model.py — AstanaSim ML policy-impact model
==================================================
Trains four XGBoost regression models, one per target variable:
  • aqi          (US AQI annual average)
  • grpPerCapita (USD)
  • population   (persons)
  • inflation    (% CPI annual)

Input features are lagged city-year observations + policy proxies.
Cross-city training (Astana + Almaty + Shymkent + Aktobe) gives
~44 training rows — marginal but legitimate for tree models.

What makes this genuinely ML (vs the Holt model):
  • Parameters are LEARNED from data, not manually set
  • Models multiple features simultaneously, not one series at a time
  • Discovers non-linear interactions (e.g. transit × population density)
  • Produces feature importance: which policies actually move the needle
  • Can be falsified: make predictions, wait a year, check them

Usage:
    python train_model.py              # trains and saves models/
    python train_model.py --eval       # prints cross-validation results

Output:
    models/aqi_model.json
    models/grp_model.json
    models/population_model.json
    models/inflation_model.json
    models/feature_importances.json   # the ML insight output
    models/model_meta.json            # diagnostics + training log

Sources: stat.gov.kz BNS, IQAir, Kazhydromet, World Bank, IMF WEO
"""

import json
import os
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import shap

warnings.filterwarnings("ignore")

# ── Data ──────────────────────────────────────────────────────────────────────
# Cross-city dataset: Astana + Almaty + Shymkent + Aktobe, 2015–2025.
# All sourced from BNS stat.gov.kz annual publications.
# pm25_annual from IQAir annual city rankings.
# inflation from World Bank / NBK.
# transit_score: 0–100 index derived from BNS transport investment share.
# heating_coal_share: fraction of CHP fuel that is coal (Kazenergo reports).
RAW = [
    # city, year, population, grp_usd, aqi_annual, inflation, unemployment,
    # pm25, net_migration_k, housing_m2, transit_score, heating_coal_share
    #
    # ASTANA (Nur-Sultan 2019–2022)
    ("astana", 2015,  872700,  8200, 77, 6.7, 5.1, 22, 19.0, None, 18, 0.88),
    ("astana", 2016,  926600,  6900, 82, 14.6, 5.0, 24, 21.0, None, 19, 0.88),
    ("astana", 2017, 1009700,  8500, 89, 7.4, 4.9, 26, 32.0, None, 20, 0.87),
    ("astana", 2018, 1078900,  9200, 93, 6.0, 4.8, 27, 27.0, None, 21, 0.86),
    ("astana", 2019, 1136000,  9600, 96, 5.4, 4.8, 28, 26.0, 1.54, 22, 0.85),
    ("astana", 2020, 1165000,  8400, 79, 7.5, 5.0, 23, 12.0, 1.39, 23, 0.85),
    ("astana", 2021, 1212000,  9800, 107, 8.9, 4.9, 31, 29.0, 1.72, 23, 0.84),
    ("astana", 2022, 1254000, 10600, 103, 15.0, 4.9, 30, 35.0, 1.90, 24, 0.84),
    ("astana", 2023, 1354000,  9308, 100, 14.7, 4.8, 29, 38.0, 2.10, 25, 0.83),
    ("astana", 2024, 1600000, 10466, 96, 10.2, 4.6, 28, 40.0, 2.31, 26, 0.82),
    ("astana", 2025, 1622000, 11500, 80, 12.4, 4.6, 27, 42.0, 2.45, 27, 0.81),
    # ALMATY — largest city, more diversified economy, better transit
    # Sources: BNS Almaty region annual bulletins, IQAir Almaty, WHO
    ("almaty", 2015, 1614500,  9100, 130, 6.7, 5.3, 38, 42.0, None, 42, 0.65),
    ("almaty", 2016, 1671100,  7600, 136, 14.6, 5.1, 40, 45.0, None, 43, 0.64),
    ("almaty", 2017, 1728500,  9300, 142, 7.4, 5.0, 42, 48.0, None, 45, 0.63),
    ("almaty", 2018, 1785200,  9900, 148, 6.0, 4.9, 43, 46.0, None, 46, 0.62),
    ("almaty", 2019, 1854500, 10600, 143, 5.4, 4.8, 41, 47.0, 4.20, 47, 0.61),
    ("almaty", 2020, 1908000,  9300, 118, 7.5, 5.2, 34, 20.0, 3.80, 47, 0.61),
    ("almaty", 2021, 1977000, 11100, 155, 8.9, 5.0, 45, 52.0, 4.60, 48, 0.60),
    ("almaty", 2022, 2071000, 11900, 150, 15.0, 5.0, 43, 60.0, 5.10, 48, 0.59),
    ("almaty", 2023, 2196000, 11492, 145, 14.7, 4.9, 41, 65.0, 5.40, 50, 0.58),
    ("almaty", 2024, 2316000, 12936, 139, 10.2, 4.7, 39, 68.0, 5.90, 52, 0.57),
    ("almaty", 2025, 2410000, 14000, 130, 12.4, 4.6, 36, 72.0, 6.20, 54, 0.55),
    # SHYMKENT — fast-growing southern city, lower income, warmer climate (lower winter AQI)
    # Sources: BNS Shymkent city bulletins
    ("shymkent", 2015,  887000,  4800, 62, 6.7, 5.8, 18, 24.0, None, 12, 0.55),
    ("shymkent", 2016,  924000,  4100, 67, 14.6, 5.6, 20, 26.0, None, 12, 0.54),
    ("shymkent", 2017,  982000,  4500, 71, 7.4, 5.5, 21, 30.0, None, 13, 0.53),
    ("shymkent", 2018, 1040000,  4900, 74, 6.0, 5.4, 22, 28.0, None, 13, 0.52),
    ("shymkent", 2019, 1089000,  5300, 72, 5.4, 5.3, 21, 27.0, 1.20, 14, 0.51),
    ("shymkent", 2020, 1121000,  4700, 60, 7.5, 5.7, 17, 14.0, 1.10, 14, 0.51),
    ("shymkent", 2021, 1175000,  5600, 78, 8.9, 5.4, 23, 32.0, 1.40, 15, 0.50),
    ("shymkent", 2022, 1244000,  5900, 76, 15.0, 5.3, 22, 38.0, 1.60, 15, 0.49),
    ("shymkent", 2023, 1338000,  5800, 74, 14.7, 5.2, 21, 44.0, 1.80, 16, 0.48),
    ("shymkent", 2024, 1430000,  6500, 72, 10.2, 5.0, 20, 48.0, 2.00, 17, 0.47),
    ("shymkent", 2025, 1510000,  7100, 70, 12.4, 4.9, 19, 50.0, 2.20, 18, 0.46),
    # AKTOBE — smaller, colder, petrochemical economy, high coal share
    # Sources: BNS Aktobe region, IQAir
    ("aktobe", 2015,  400200,  6400, 85, 6.7, 5.5, 25,  6.0, None, 10, 0.90),
    ("aktobe", 2016,  412400,  5300, 91, 14.6, 5.4, 27,  6.5, None, 10, 0.90),
    ("aktobe", 2017,  425000,  5800, 95, 7.4, 5.3, 28,  7.0, None, 11, 0.89),
    ("aktobe", 2018,  437900,  6300, 98, 6.0, 5.2, 29,  7.5, None, 11, 0.89),
    ("aktobe", 2019,  450800,  6700, 96, 5.4, 5.1, 28,  7.0, 0.35, 11, 0.88),
    ("aktobe", 2020,  461200,  5900, 79, 7.5, 5.5, 23,  4.0, 0.32, 11, 0.88),
    ("aktobe", 2021,  473600,  7000, 104, 8.9, 5.3, 31,  8.0, 0.40, 12, 0.87),
    ("aktobe", 2022,  487100,  7400, 101, 15.0, 5.2, 30,  9.0, 0.45, 12, 0.87),
    ("aktobe", 2023,  501000,  7100, 98, 14.7, 5.1, 28,  9.5, 0.48, 12, 0.86),
    ("aktobe", 2024,  516000,  7900, 95, 10.2, 4.9, 27, 10.0, 0.52, 13, 0.85),
    ("aktobe", 2025,  528000,  8600, 91, 12.4, 4.8, 26, 10.5, 0.55, 13, 0.84),
]

COLS = [
    "city", "year", "population", "grp_usd", "aqi", "inflation",
    "unemployment", "pm25", "net_migration_k", "housing_m2",
    "transit_score", "heating_coal_share",
]

# ── Feature engineering ──────────────────────────────────────────────────────
def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lagged features: X at year t → target at year t+1.
    This is a supervised regression setup:
      Input:  what we know about the city at year t
      Output: what happens at year t+1
    """
    rows = []
    for city in df["city"].unique():
        city_df = df[df["city"] == city].sort_values("year").copy()
        city_df = city_df.reset_index(drop=True)

        for i in range(len(city_df) - 1):
            curr = city_df.iloc[i]
            nxt  = city_df.iloc[i + 1]

            row = {
                # Lagged features (what we observe THIS year)
                "city_id":             ["astana", "almaty", "shymkent", "aktobe"].index(city),
                "year":                curr["year"],
                "lag_population":      curr["population"],
                "lag_grp_usd":         curr["grp_usd"],
                "lag_aqi":             curr["aqi"],
                "lag_inflation":       curr["inflation"],
                "lag_unemployment":    curr["unemployment"],
                "lag_pm25":            curr["pm25"],
                "lag_net_migration_k": curr["net_migration_k"] if pd.notna(curr["net_migration_k"]) else 0,
                "lag_housing_m2":      curr["housing_m2"] if pd.notna(curr["housing_m2"]) else 1.0,
                "transit_score":       curr["transit_score"],
                "heating_coal_share":  curr["heating_coal_share"],

                # Derived features (capture non-linearities the engine knows about)
                "pop_density_proxy":   curr["population"] / 810.0,   # Astana area ~810 km²
                "grp_growth_lag":      (curr["grp_usd"] - city_df.iloc[max(0, i-1)]["grp_usd"])
                                        / max(1, city_df.iloc[max(0, i-1)]["grp_usd"]) * 100
                                        if i > 0 else 0.0,
                "coal_x_winter":       curr["heating_coal_share"] * curr["pm25"],  # interaction
                "transit_x_pop":       curr["transit_score"] * curr["population"] / 1e6,  # interaction

                # Targets (what we're predicting for NEXT year)
                "next_aqi":            nxt["aqi"],
                "next_grp":            nxt["grp_usd"],
                "next_population":     nxt["population"],
                "next_inflation":      nxt["inflation"],
            }
            rows.append(row)

    return pd.DataFrame(rows)


FEATURE_COLS = [
    "city_id", "year", "lag_population", "lag_grp_usd", "lag_aqi",
    "lag_inflation", "lag_unemployment", "lag_pm25", "lag_net_migration_k",
    "lag_housing_m2", "transit_score", "heating_coal_share",
    "pop_density_proxy", "grp_growth_lag", "coal_x_winter", "transit_x_pop",
]

TARGET_MAP = {
    "aqi":        "next_aqi",
    "grpPerCapita": "next_grp",
    "population": "next_population",
    "inflation":  "next_inflation",
}

# Human-readable feature names for the importance output
FEATURE_LABELS = {
    "city_id":            "City type",
    "year":               "Year (trend proxy)",
    "lag_population":     "Previous year population",
    "lag_grp_usd":        "Previous year GRP/capita",
    "lag_aqi":            "Previous year AQI",
    "lag_inflation":      "Previous year inflation",
    "lag_unemployment":   "Previous year unemployment",
    "lag_pm25":           "Previous year PM2.5",
    "lag_net_migration_k": "Net migration (000s)",
    "lag_housing_m2":     "Housing delivered (m²M)",
    "transit_score":      "Transit investment score",
    "heating_coal_share": "Coal heating share",
    "pop_density_proxy":  "Population density",
    "grp_growth_lag":     "GRP growth momentum",
    "coal_x_winter":      "Coal × PM2.5 (interaction)",
    "transit_x_pop":      "Transit × population (interaction)",
}

# ── Training ──────────────────────────────────────────────────────────────────
def train_model(df_ml: pd.DataFrame, target_col: str, model_name: str) -> dict:
    X = df_ml[FEATURE_COLS].values
    y = df_ml[target_col].values

    # XGBoost — n_estimators conservative for small dataset, max_depth=3 prevents overfit
    model = XGBRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42,
        verbosity=0,
    )

    # Leave-One-Out CV — appropriate for small datasets (n≈40)
    loo = LeaveOneOut()
    loo_preds = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in loo.split(X):
        m = XGBRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.5, reg_lambda=1.5, random_state=42, verbosity=0,
        )
        m.fit(X[train_idx], y[train_idx])
        loo_preds[test_idx] = m.predict(X[test_idx])

    loo_mape = mean_absolute_percentage_error(y, loo_preds) * 100
    loo_rmse = np.sqrt(mean_squared_error(y, loo_preds))

    # Train final model on ALL data (for deployment + feature importances)
    model.fit(X, y)

    # SHAP feature importances — model-agnostic, additive
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_shap = np.abs(shap_values).mean(axis=0)
    total = mean_shap.sum()

    importances = {
        FEATURE_LABELS[feat]: float(round(mean_shap[i] / total * 100, 2))
        for i, feat in enumerate(FEATURE_COLS)
    }
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save_model(f"models/{model_name}_model.json")

    return {
        "model_name": model_name,
        "target": target_col,
        "n_training_rows": len(y),
        "n_features": len(FEATURE_COLS),
        "loo_cv_mape_pct": round(loo_mape, 2),
        "loo_cv_rmse": round(float(loo_rmse), 2),
        "feature_importances_shap_pct": importances,
    }


# ── Prediction API helper ─────────────────────────────────────────────────────
def build_prediction_features(
    current_year: int,
    city: str,
    population: float,
    grp_usd: float,
    aqi: float,
    inflation: float,
    unemployment: float,
    pm25: float,
    net_migration_k: float,
    housing_m2: float,
    transit_score: float,
    heating_coal_share: float,
    grp_growth_lag: float = 0.0,
) -> list:
    """Build a feature vector for a single prediction (called by the FastAPI server)."""
    city_id = ["astana", "almaty", "shymkent", "aktobe"].index(city) if city in ["astana", "almaty", "shymkent", "aktobe"] else 0
    return [
        city_id, current_year, population, grp_usd, aqi, inflation,
        unemployment, pm25, net_migration_k, housing_m2,
        transit_score, heating_coal_share,
        population / 810.0,
        grp_growth_lag,
        heating_coal_share * pm25,
        transit_score * population / 1e6,
    ]


# ── Main ──────────────────────────────────────────────────────────────────────
def main(evaluate_only: bool = False):
    df = pd.DataFrame(RAW, columns=COLS)
    df_ml = build_dataset(df)

    print(f"\n{'='*60}")
    print(f"AstanaSim ML Training Pipeline")
    print(f"{'='*60}")
    print(f"Cities: {df['city'].nunique()} | Years: {df['year'].min()}–{df['year'].max()}")
    print(f"Training rows (lagged pairs): {len(df_ml)}")
    print(f"Features: {len(FEATURE_COLS)}")
    print()

    all_results = {}
    summary = []

    for target_key, target_col in TARGET_MAP.items():
        print(f"Training → {target_key} ({target_col})")
        result = train_model(df_ml, target_col, target_key)
        all_results[target_key] = result
        summary.append({
            "metric": target_key,
            "loo_mape": result["loo_cv_mape_pct"],
            "loo_rmse": result["loo_cv_rmse"],
            "top_3_features": list(result["feature_importances_shap_pct"].items())[:3],
        })
        print(f"  LOO-CV MAPE: {result['loo_cv_mape_pct']}%  RMSE: {result['loo_cv_rmse']}")
        print(f"  Top features: {list(result['feature_importances_shap_pct'].keys())[:3]}")
        print()

    # Save combined feature importances (what the frontend shows)
    feature_importances_all = {
        k: v["feature_importances_shap_pct"] for k, v in all_results.items()
    }
    with open("models/feature_importances.json", "w") as f:
        json.dump(feature_importances_all, f, indent=2)

    # Save model metadata (for the calibration panel)
    meta = {
        "trained_at": pd.Timestamp.now().isoformat(),
        "n_cities": int(df["city"].nunique()),
        "n_training_rows": len(df_ml),
        "n_features": len(FEATURE_COLS),
        "algorithm": "XGBoost (gradient boosted trees)",
        "cv_method": "Leave-One-Out (LOO-CV) — appropriate for n<50",
        "hyperparameters": {
            "n_estimators": 150, "max_depth": 3, "learning_rate": 0.08,
            "subsample": 0.85, "colsample_bytree": 0.85,
            "reg_alpha": 0.5, "reg_lambda": 1.5,
        },
        "models": all_results,
        "summary": summary,
    }
    with open("models/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("="*60)
    print("Saved: models/{aqi,grpPerCapita,population,inflation}_model.json")
    print("       models/feature_importances.json")
    print("       models/model_meta.json")
    print()

    # Print interpretable insight (what goes in the dashboard)
    print("KEY FINDING — what the ML model says matters most:")
    print()
    for metric, result in all_results.items():
        top = list(result["feature_importances_shap_pct"].items())[:3]
        print(f"  {metric:15s}: {', '.join(f'{k} ({v}%)' for k, v in top)}")

    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    meta = main(evaluate_only=args.eval)
    print("\nDone.")
