# Author: Jay Rajesh

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

FEATURES_CSV = '/Users/jayrajesh/jayrajesh.github.io/Files/processed/combined_features.csv'
LABELS_CSV   = '/Users/jayrajesh/Downloads/combined_labels.csv'

df_X = pd.read_csv(FEATURES_CSV)
df_y = pd.read_csv(LABELS_CSV)

# Convert to NumPy arrays
X = df_X.values    # shape (n_samples, n_features)
y = df_y.values    # shape (n_samples, 2)

print(f"Features shape: {X.shape}")
print(f"Labels   shape: {y.shape}")


# First split out test set (20%)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

# Then split train/val 75/25 of the remaining 80% → 60/20 overall
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42)

print(f"Training samples:   {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Test samples:       {X_test.shape[0]}")

# Train XGBoost Regressor
model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',      # fast histogram-based
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,
    verbose=True
)

# Evaluate on Test Set
y_pred = model.predict(X_test)

# Compute metrics for each of the two outputs separately
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nTest MSE:  {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE:  {mae:.4f}")
print(f"Test R²:   {r2:.4f}")

# (Optional) Feature Importance
try:
    importances = model.feature_importances_
    idxs = np.argsort(importances)[-20:]
    plt.figure(figsize=(8,6))
    plt.barh(range(len(idxs)), importances[idxs], align='center')
    plt.yticks(range(len(idxs)), [df_X.columns[i] for i in idxs])
    plt.title("Top 20 feature importances")
    plt.tight_layout()
    plt.show()
except Exception:
    pass


