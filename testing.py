from knowledge.feature_store.feature_engineering import load_demand, build_sku_features; 
d = load_demand() 
f = build_sku_features(d) 
print(len(f), 'rows')