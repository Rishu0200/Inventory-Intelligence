from knowledge.feature_store.demand_model import load_model, forecast_sku
from knowledge.feature_store.anomaly_model import load_anomaly_model
import requests

"""
for i in range(25):
    r = requests.post("http://localhost:8000/api/auth/login",
                      data={"username": "nobody@example.com", "password": "wrong"})
    print(i + 1, r.status_code)
"""
from cache.redis_client import rate_limit_check
print([rate_limit_check("debug-test", 3, 60) for _ in range(5)])