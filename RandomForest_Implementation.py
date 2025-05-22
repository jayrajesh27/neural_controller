import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


FEATURES_CSV = '/Users/jayrajesh/jayrajesh.github.io/Files/processed/combined_features.csv'
LABELS_CSV   = '/Users/jayrajesh/Downloads/combined_labels.csv'

df_X = pd.read_csv(FEATURES_CSV)
df_y = pd.read_csv(LABELS_CSV)
X = df_X.values                 # shape: (n_samples, n_features)
y = df_y.values                 # shape: (n_samples, 2)

print(f"Features shape: {X.shape}")
print(f"Labels   shape: {y.shape}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# @param
params = {
    'n_estimators': 900,
    'max_depth': None,
    'min_samples_split': 10,
    'min_samples_leaf': 2,
    'max_features': 'log2',
    'bootstrap': False,
    'n_jobs': -1
}

# Run 10 Times and Collect Metrics
mse_list, rmse_list, mae_list, r2_list = [], [], [], []

for seed in range(10):
    model = RandomForestRegressor(random_state=seed, **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    mse_list.append(mse)
    rmse_list.append(rmse)
    mae_list.append(mae)
    r2_list.append(r2)

    print(f"Run {seed+1}/10 — MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# Summary Statistics
print("\nAverage performance over 10 runs:")
print(f"MSE:  {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}")
print(f"RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
print(f"MAE:  {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
print(f"R2:   {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")





