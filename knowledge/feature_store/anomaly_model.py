"""
Isolation Forest anomaly detection on inventory + demand patterns.
Flags unusual stock levels, demand spikes, or lead time outliers.
"""
from __future__ import annotations
import os, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import Paths
from knowledge.feature_store.feature_engineering import load_demand, load_inventory


FEATURE_COLS = ["net_units", "lag_1", "roll_mean_3", "roll_std_3", "is_high_season"]


def _build_anomaly_features(demand: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix for anomaly detection from demand history."""
    from knowledge.feature_store.feature_engineering import build_sku_features
    feats = build_sku_features(demand)
    # Keep only columns that exist in feats
    cols = [c for c in FEATURE_COLS if c in feats.columns]
    X = feats[cols].dropna().copy()
    X["sku_id"]  = feats.loc[X.index, "sku_id"]
    X["period"]  = feats.loc[X.index, "period"]
    X["product_name"] = feats.loc[X.index, "product_name"]
    return X


def train_isolation_forest(contamination: float = 0.05) -> tuple:
    """Train Isolation Forest + scaler on demand features."""
    demand = load_demand()
    df     = _build_anomaly_features(demand)
    num_cols = [c for c in FEATURE_COLS if c in df.columns]
    X_raw    = df[num_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Evaluate: flag count
    preds  = model.predict(X_scaled)
    n_anom = int((preds == -1).sum())
    print(f"[anomaly_model] Flagged {n_anom}/{len(preds)} rows as anomalies "
          f"({n_anom/len(preds)*100:.1f}%)")
    return model, scaler


def save_anomaly_model(model: IsolationForest, scaler: StandardScaler):
    os.makedirs(Paths.MODELS, exist_ok=True)
    with open(Paths.MODELS / "anomaly_iso.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print(f"[anomaly_model] Saved → {Paths.MODELS / 'anomaly_iso.pkl'}")


def load_anomaly_model() -> tuple[IsolationForest, StandardScaler] | tuple[None, None]:
    path = Paths.MODELS / "anomaly_iso.pkl"
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["scaler"]


def detect_anomalies(sku_id: str | None = None) -> list[dict]:
    """
    Run anomaly detection on demand data.

    Args:
        sku_id: If provided, filter to a single SKU. Otherwise all SKUs.

    Returns:
        List of anomaly dicts for flagged rows.
    """
    model, scaler = load_anomaly_model()
    demand = load_demand()

    if sku_id:
        demand = demand[demand["sku_id"] == sku_id]

    if demand.empty:
        return []

    df = _build_anomaly_features(demand)
    if df.empty:
        return []

    num_cols = [c for c in FEATURE_COLS if c in df.columns]
    X_scaled = scaler.transform(df[num_cols].values) if scaler else df[num_cols].values

    if model:
        scores = model.decision_function(X_scaled)   # lower = more anomalous
        preds  = model.predict(X_scaled)              # -1 = anomaly
    else:
        # Fallback: z-score based
        z = np.abs((df[num_cols].values - df[num_cols].values.mean(0)) /
                   (df[num_cols].values.std(0) + 1e-8))
        preds  = np.where(z.max(1) > 2.5, -1, 1)
        scores = -z.max(1)

    df["anomaly_score"] = scores
    df["is_anomaly"]    = (preds == -1)

    anomalies = df[df["is_anomaly"]][[
        "sku_id", "product_name", "period", "net_units",
        "roll_mean_3", "anomaly_score"
    ]].copy()

    return anomalies.to_dict("records")


if __name__ == "__main__":
    model, scaler = train_isolation_forest()
    save_anomaly_model(model, scaler)
    results = detect_anomalies()
    print(f"\nTop anomalies:\n{pd.DataFrame(results).head(10)}")
