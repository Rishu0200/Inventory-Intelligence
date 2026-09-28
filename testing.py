from knowledge.feature_store.demand_model import load_model, forecast_sku
from knowledge.feature_store.anomaly_model import load_anomaly_model

model = load_model()
print("XGBoost loaded from Storage:", model is not None)

anomaly_model, scaler = load_anomaly_model()
print("Isolation Forest loaded from Storage:", anomaly_model is not None)

result = forecast_sku("RSH-001", horizon=3, model=model)
print("Forecast:", result["forecast"])