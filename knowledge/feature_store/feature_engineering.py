"""
Build ML feature matrix from raw inventory CSVs.
Creates lag features, rolling stats, calendar flags for each SKU.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

from config import Paths


def load_demand() -> pd.DataFrame:
    """Load and clean demand_history.csv."""
    df = pd.read_csv(Paths.DATA_RAW / "demand_history.csv")
    df["period"] = pd.to_datetime(df["period"], format="%Y-%m")
    df = df.sort_values(["sku_id", "period"]).reset_index(drop=True)
    return df


def load_inventory() -> pd.DataFrame:
    """Load inventory_history.csv."""
    df = pd.read_csv(Paths.DATA_RAW / "inventory_history.csv")
    return df


def load_supplier_terms() -> pd.DataFrame:
    return pd.read_csv(Paths.DATA_RAW / "supplier_terms.csv")


def build_sku_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature matrix for each SKU time-series row.

    Features created per SKU:
        - lag_1, lag_3, lag_6     : units sold N months ago
        - roll_mean_3, roll_std_3  : 3-month rolling stats
        - roll_mean_6              : 6-month rolling mean
        - yoy_growth               : year-over-year growth rate
        - month, quarter           : calendar flags
        - abc_class_enc            : A=3, B=2, C=1
        - is_high_season           : months 10,11,12,1 (festive + New Year)
    """
    results = []
    abc_map = {"A": 3, "B": 2, "C": 1}

    for sku_id, grp in df.groupby("sku_id"):
        grp = grp.sort_values("period").copy()
        grp["lag_1"]      = grp["net_units"].shift(1)
        grp["lag_3"]      = grp["net_units"].shift(3)
        grp["lag_6"]      = grp["net_units"].shift(6)
        grp["roll_mean_3"]= grp["net_units"].shift(1).rolling(3).mean()
        grp["roll_std_3"] = grp["net_units"].shift(1).rolling(3).std().fillna(0)
        grp["roll_mean_6"]= grp["net_units"].shift(1).rolling(6).mean()
        grp["yoy_growth"] = grp["net_units"].pct_change(12).fillna(0)
        grp["month"]      = grp["period"].dt.month
        grp["quarter"]    = grp["period"].dt.quarter
        grp["abc_enc"]    = grp["abc_class"].map(abc_map).fillna(1)
        grp["is_high_season"] = grp["month"].isin([10, 11, 12, 1]).astype(int)
        results.append(grp)

    features = pd.concat(results).dropna(subset=["lag_1"]).reset_index(drop=True)
    return features


def get_feature_columns() -> list[str]:
    return [
        "lag_1", "lag_3", "lag_6",
        "roll_mean_3", "roll_std_3", "roll_mean_6",
        "yoy_growth", "month", "quarter",
        "abc_enc", "is_high_season",
    ]


def save_features(features: pd.DataFrame) -> Path:
    dest = Paths.DATA_PROCESSED / "features.parquet"
    features.to_parquet(dest, index=False)
    print(f"[feature_engineering] Saved {len(features)} rows → {dest}")
    return dest


if __name__ == "__main__":
    demand = load_demand()
    feats  = build_sku_features(demand)
    save_features(feats)
    print(feats[["sku_id", "period", "net_units"] + get_feature_columns()].tail(10))
