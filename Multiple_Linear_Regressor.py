
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FEATURES_CSV = '/Users/jayrajesh/jayrajesh.github.io/Files/processed/combined_features.csv'
LABELS_CSV   = '/Users/jayrajesh/Downloads/combined_labels.csv'

X = pd.read_csv(FEATURES_CSV).values
y = pd.read_csv(LABELS_CSV).values

print(f"Features shape: {X.shape}")
print(f"Labels   shape: {y.shape}")


# 1. Train/Val/Test Split
X_trval, X_test, y_trval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_trval, y_trval, test_size=0.25, random_state=42)

print(f"Train samples: {X_train.shape[0]}")
print(f" Val samples: {X_val.shape[0]}")
print(f" Test samples: {X_test.shape[0]}")

# 2. Build & Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 3. Evaluate on Test Set
y_pred = model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\nLinear Regression Test set performance:")
print(f"  MSE : {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE : {mae:.4f}")
print(f"  R²  : {r2:.4f}")


