from sklearn.ensemble import RandomForestRegressor

def get_model(n_estimators=100, random_state=42):
    return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)