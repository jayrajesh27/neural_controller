import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, initializers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# -------------------------------
# 1. Data Preparation
# -------------------------------

# Load the dataset (ensure the file path is correct)
data_file = 'merged_navigation_data.csv'
df = pd.read_csv(data_file)

# Check for missing values
print("Missing values per column before cleaning:")
print(df.isnull().sum())

# Drop any rows with missing values
df = df.dropna()
print("Missing values per column after cleaning:")
print(df.isnull().sum())

print("Columns in dataset:", df.columns.tolist())

target_columns = ['odom_linear_x', 'odom_angular_z']
for col in target_columns:
    if col not in df.columns:
        raise ValueError(f"Target column '{col}' not found. Available columns: {df.columns.tolist()}")

X = df.drop(target_columns + ['timestamp'], axis=1).values
y = df[target_columns].values

# Split data into train+validation (80%) and test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Further split train_val into training (70% of total data) and validation (10% of total data)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.125, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("Data splitting complete.")
print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# -------------------------------
# 2. Model Architecture (3-Layer ANN)
# -------------------------------

n_features = X_train.shape[1]
model = models.Sequential([
    layers.Dense(256, activation='relu',
                 input_shape=(n_features,),
                 kernel_initializer=initializers.HeNormal(),
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(256, activation='relu',
                 kernel_initializer=initializers.HeNormal(),
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(256, activation='relu',
                 kernel_initializer=initializers.HeNormal(),
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(2, activation='linear')
])

model.summary()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# -------------------------------
# 3. Training Configuration
# -------------------------------

# Early stopping to prevent overfitting
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# -------------------------------
# 4. Model Training
# -------------------------------

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=1)

print("History keys:", history.history.keys())

# -------------------------------
# 5. Evaluation & Reporting
# -------------------------------

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss (MSE):", test_loss)
print("Test MAE:", test_mae)

# Compute RMSE on the test set
test_rmse = np.sqrt(test_loss)
print("Test RMSE:", test_rmse)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Compute R² score for regression accuracy
r2 = r2_score(y_test, y_pred)
print("Test R² Score:", r2)

# Training and validation loss over epochs
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.title('Model Loss Over Epochs')
plt.show()

# Training and validation MAE over epochs
mae_key = 'mae'
val_mae_key = 'val_mae'
if 'mean_absolute_error' in history.history:
    mae_key = 'mean_absolute_error'
    val_mae_key = 'val_mean_absolute_error'

plt.figure()
plt.plot(history.history[mae_key], label='Training MAE')
plt.plot(history.history[val_mae_key], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.title('Model MAE Over Epochs')
plt.show()

# Combined plot
plt.figure()
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.plot(history.history[mae_key], label='Training MAE')
plt.plot(history.history[val_mae_key], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.title('Training vs. Validation: Loss and MAE')
plt.show()

plt.figure(figsize=(12, 5))

# Scatter plot for odom_linear_x
plt.subplot(1, 2, 1)
plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
plt.plot([y_test[:, 0].min(), y_test[:, 0].max()],
         [y_test[:, 0].min(), y_test[:, 0].max()], 'r--')
plt.xlabel('True odom_linear_x')
plt.ylabel('Predicted odom_linear_x')
plt.title('Prediction: Odom Linear X')

# Scatter plot for odom_angular_z
plt.subplot(1, 2, 2)
plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5)
plt.plot([y_test[:, 1].min(), y_test[:, 1].max()],
         [y_test[:, 1].min(), y_test[:, 1].max()], 'r--')
plt.xlabel('True odom_angular_z')
plt.ylabel('Predicted odom_angular_z')
plt.title('Prediction: Odom Angular Z')

plt.tight_layout()
plt.show()

# Then, check for overfitting/underfitting by comparing final training and validation loss
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
print("Final Training Loss:", final_train_loss)
print("Final Validation Loss:", final_val_loss)
if final_train_loss < final_val_loss * 0.9:
    print("The model might be overfitting.")
elif final_train_loss > final_val_loss * 1.1:
    print("The model might be underfitting.")
else:
    print("The model performance appears balanced.")