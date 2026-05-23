"""
Demand forecasting: ARIMA for univariate baseline, XGBoost for multi-feature.
Logs every training run to MLflow. Models saved per-SKU to data/processed/models/.
"""
from __future__ import annotations
import os, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import xgboost as xgb
#import mlflow
#import mlflow.xgboost

from config import Paths, settings
from knowledge.feature_store.feature_engineering import (
    load_demand, build_sku_features, get_feature_columns
)

FEATURE_COLS = get_feature_columns()
TARGET_COL   = "net_units"


# ── XGBoost (multi-feature, all SKUs together) ────────────────────────────────

def train_xgboost(features: pd.DataFrame) -> xgb.XGBRegressor:
    """Train a single XGBoost model on all SKUs using time-series CV."""
    X = features[FEATURE_COLS].values
    y = features[TARGET_COL].values

    tscv = TimeSeriesSplit(n_splits=3)
    val_rmse_list = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        model = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0
        )
        model.fit(X[tr_idx], y[tr_idx],
                  eval_set=[(X[val_idx], y[val_idx])],
                  verbose=False)
        preds = model.predict(X[val_idx])
        rmse  = np.sqrt(mean_squared_error(y[val_idx], preds))
        mape  = mean_absolute_percentage_error(y[val_idx], preds) * 100
        val_rmse_list.append(rmse)
        print(f"  Fold {fold+1}: RMSE={rmse:.1f}  MAPE={mape:.1f}%")

    # Final model on all data
    final_model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=0
    )
    final_model.fit(X, y)
    avg_rmse = np.mean(val_rmse_list)
    print(f"  Mean CV RMSE: {avg_rmse:.1f}")
    return final_model, float(avg_rmse)


def save_model(model: xgb.XGBRegressor, name: str = "demand_xgb.pkl") -> Path:
    os.makedirs(Paths.MODELS, exist_ok=True)
    dest = Paths.MODELS / name
    with open(dest, "wb") as f:
        pickle.dump(model, f)
    print(f"[demand_model] Saved model → {dest}")
    return dest


def load_model(name: str = "demand_xgb.pkl") -> xgb.XGBRegressor | None:
    path = Paths.MODELS / name
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── ARIMA per-SKU ─────────────────────────────────────────────────────────────

def train_arima_sku(series: pd.Series, order=(1,1,1)):
    """Fit ARIMA on a single SKU monthly series."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(series.values, order=order)
        return model.fit()
    except Exception as e:
        print(f"  ARIMA failed: {e}")
        return None


# ── Inference ─────────────────────────────────────────────────────────────────

def forecast_sku(sku_id: str,
                 horizon: int = 3,
                 model: xgb.XGBRegressor | None = None) -> dict:
    """
    Forecast demand for a SKU over `horizon` months.

    Returns:
        {"sku_id": ..., "forecast": [...], "lower": [...], "upper": [...]}
    """
    if model is None:
        model = load_model()

    demand  = load_demand()
    sku_df  = demand[demand["sku_id"] == sku_id].sort_values("period")

    if sku_df.empty:
        return {"sku_id": sku_id, "forecast": [], "lower": [], "upper": []}

    features_df = build_sku_features(demand)
    sku_feats   = features_df[features_df["sku_id"] == sku_id].copy()

    if sku_feats.empty or model is None:
        # Fallback: simple rolling mean
        avg = float(sku_df["net_units"].tail(3).mean())
        std = float(sku_df["net_units"].tail(6).std(ddof=0))
        fc  = [round(avg, 1)] * horizon
        return {
            "sku_id":   sku_id,
            "forecast": fc,
            "lower":    [round(max(0, f - 1.64*std), 1) for f in fc],
            "upper":    [round(f + 1.64*std, 1) for f in fc],
        }

    last_row = sku_feats.iloc[[-1]][FEATURE_COLS].values
    fc_vals, lo_vals, hi_vals = [], [], []

    std = float(sku_df["net_units"].std(ddof=0))

    for _ in range(horizon):
        pred = float(model.predict(last_row)[0])
        pred = max(0, pred)
        fc_vals.append(round(pred, 1))
        lo_vals.append(round(max(0, pred - 1.64*std), 1))
        hi_vals.append(round(pred + 1.64*std, 1))

        # Roll features forward (simplified)
        last_row[0][0] = pred   # lag_1 = current pred

    return {"sku_id": sku_id, "forecast": fc_vals, "lower": lo_vals, "upper": hi_vals}


# ── Main training entry point ─────────────────────────────────────────────────
'''

def train_and_save():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    demand   = load_demand()
    features = build_sku_features(demand)

    print(f"Training XGBoost on {len(features)} rows, "
          f"{features['sku_id'].nunique()} SKUs ...")

    with mlflow.start_run(run_name="demand_xgboost"):
        model, rmse = train_xgboost(features)
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("cv_rmse", rmse)
        save_model(model)
        try:
            mlflow.xgboost.log_model(model, "model")
        except Exception:
            pass

    print("✓ Demand model training complete.")
'''

if __name__ == "__main__":
    train_and_save()
