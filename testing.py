import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from knowledge.feature_store.feature_engineering import load_demand, build_sku_features, get_feature_columns

cols = get_feature_columns()
f = build_sku_features(load_demand()).sort_values(["period", "sku_id"]).reset_index(drop=True)
X, y, naive = f[cols].values, f["net_units"].values, f["lag_1"].values

model_rmse, naive_rmse = [], []
for tr, va in TimeSeriesSplit(n_splits=3).split(X):
    m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    m.fit(X[tr], y[tr])
    model_rmse.append(np.sqrt(mean_squared_error(y[va], m.predict(X[va]))))
    naive_rmse.append(np.sqrt(mean_squared_error(y[va], naive[va])))

print("model RMSE per fold:", [round(v, 1) for v in model_rmse])
print("naive RMSE per fold:", [round(v, 1) for v in naive_rmse])
print("model mean:", round(np.mean(model_rmse), 1), " naive mean:", round(np.mean(naive_rmse), 1))
print("improvement:", round((1 - np.mean(model_rmse) / np.mean(naive_rmse)) * 100, 1), "%")