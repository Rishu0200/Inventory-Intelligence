"""
scripts/train_models.py — Train all ML models.
Runs: feature engineering → XGBoost demand model → Isolation Forest anomaly model
Usage: python scripts/train_models.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    start = time.time()
    print("\n🤖  Model Training Pipeline — Uninox Houseware")
    print("=" * 55)

    # ── Feature Engineering ───────────────────────────────────────────────────
    print("\n[1/3] Building feature matrix...")
    try:
        from knowledge.feature_store.feature_engineering import (
            load_demand, build_sku_features, save_features
        )
        demand   = load_demand()
        features = build_sku_features(demand)
        save_features(features)
        print(f"  ✓ Features: {len(features)} rows, {features['sku_id'].nunique()} SKUs")
    except Exception as e:
        print(f"  ✗ Feature engineering failed: {e}")
        return

    # ── Demand Model (XGBoost) ────────────────────────────────────────────────
    print("\n[2/3] Training XGBoost demand model...")
    try:
        from knowledge.feature_store.demand_model import train_and_save
        train_and_save()
        print("  ✓ XGBoost demand model trained and saved.")
    except Exception as e:
        print(f"  ✗ Demand model training failed: {e}")
        print("    Make sure MLflow is running: docker-compose up mlflow")

    # ── Anomaly Model (Isolation Forest) ──────────────────────────────────────
    print("\n[3/3] Training Isolation Forest anomaly model...")
    try:
        from knowledge.feature_store.anomaly_model import (
            train_isolation_forest, save_anomaly_model
        )
        model, scaler = train_isolation_forest(contamination=0.05)
        save_anomaly_model(model, scaler)
        print("  ✓ Isolation Forest trained and saved.")
    except Exception as e:
        print(f"  ✗ Anomaly model training failed: {e}")

    elapsed = time.time() - start
    print(f"\n{'=' * 55}")
    print(f"  ✅ All models trained in {elapsed:.1f}s")
    print(f"{'=' * 55}")
    print("  Models saved to: data/processed/models/")
    print("  - demand_xgb.pkl")
    print("  - anomaly_iso.pkl\n")


if __name__ == "__main__":
    main()
