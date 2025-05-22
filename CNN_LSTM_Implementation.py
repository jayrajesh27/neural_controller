import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks, initializers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

FEATURES_CSV = '/Users/jayrajesh/jayrajesh.github.io/Files/processed/combined_features.csv'
LABELS_CSV   = '/Users/jayrajesh/Downloads/combined_labels.csv'

df_feat = pd.read_csv(FEATURES_CSV)
df_lab  = pd.read_csv(LABELS_CSV)

X_raw = df_feat.values  
y_raw = df_lab.values  

SEQ_LEN = 10
def make_seqs(X, y, L):
    Xs, ys = [], []
    for i in range(L-1, len(X)):
        Xs.append(X[i-L+1 : i+1])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = make_seqs(X_raw, y_raw, SEQ_LEN)
print("Sequence shapes:", X_seq.shape, y_seq.shape)

# Chronological split: 70% train, 10% val, 20% test
n = len(X_seq)
i70, i80 = int(0.7*n), int(0.8*n)
X_tr, y_tr = X_seq[:i70],      y_seq[:i70]
X_va, y_va = X_seq[i70:i80],    y_seq[i70:i80]
X_te, y_te = X_seq[i80:],       y_seq[i80:]

# Normalize Features
scaler = StandardScaler().fit(X_tr.reshape(-1, X_tr.shape[-1]))
def scale(X):
    b, t, f = X.shape
    X2 = scaler.transform(X.reshape(-1, f))
    return X2.reshape(b, t, f)

X_tr = scale(X_tr)
X_va = scale(X_va)
X_te = scale(X_te)

# Pretrain 1D-CNN Autoencoder
n_feat = X_tr.shape[-1]
ae_in = layers.Input(shape=(n_feat, 1))

# Encoder
x = layers.Conv1D(64, 5, padding='same', activation='relu')(ae_in)
x = layers.MaxPool1D(2, padding='same')(x)
x = layers.Conv1D(32, 5, padding='same', activation='relu')(x)
encoded = layers.MaxPool1D(2, padding='same')(x)

# Decoder
x = layers.Conv1DTranspose(32, 5, strides=2, padding='same', activation='relu')(encoded)
x = layers.Conv1DTranspose(64, 5, strides=2, padding='same', activation='relu')(x)
x = layers.Conv1D(1, 5, padding='same', activation='linear')(x)
cropped = layers.Cropping1D((0, 3))(x)  # ensure output length matches input

autoencoder = models.Model(ae_in, cropped, name='ae')
autoencoder.compile(optimizer='adam', loss='mse')

ae_X = X_tr[:, -1, :].reshape(-1, n_feat, 1)
autoencoder.fit(
    ae_X, ae_X,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    callbacks=[callbacks.EarlyStopping('val_loss', patience=3, restore_best_weights=True)],
    verbose=1
)

# Extract the encoder submodel
encoder = models.Model(ae_in, encoded, name='encoder')

# Build CNN+LSTM with Regularization
encoder.trainable = False  # freeze encoder for initial training

seq_in = layers.Input(shape=(SEQ_LEN, n_feat))
x = layers.GaussianNoise(0.05)(seq_in)
x = layers.Reshape((SEQ_LEN, n_feat, 1))(x)
x = layers.TimeDistributed(encoder)(x)
x = layers.TimeDistributed(layers.Flatten())(x)
x = layers.Dropout(0.3)(x)

# LSTM with dropout and L2
x = layers.LSTM(
    128,
    activation='tanh',
    dropout=0.2,
    recurrent_dropout=0.2,
    kernel_regularizer=regularizers.l2(1e-3),
    recurrent_regularizer=regularizers.l2(1e-3)
)(x)

# Dense head with batchnorm & dropout
x = layers.Dense(
    64,
    activation='relu',
    kernel_initializer=initializers.HeNormal(),
    kernel_regularizer=regularizers.l2(1e-3)
)(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

out = layers.Dense(
    2,
    activation='linear',
    kernel_regularizer=regularizers.l2(1e-3)
)(x)

model = models.Model(seq_in, out)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Stage‑1 Training (Encoder Frozen)
hist1 = model.fit(
    X_tr, y_tr,
    validation_data=(X_va, y_va),
    epochs=30,
    batch_size=32,
    callbacks=[callbacks.EarlyStopping('val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)

# Stage‑2 Fine‑Tuning (Unfreeze Encoder)
encoder.trainable = True
for layer in encoder.layers:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='mse',
    metrics=['mae']
)

hist2 = model.fit(
    X_tr, y_tr,
    validation_data=(X_va, y_va),
    epochs=20,
    batch_size=32,
    callbacks=[callbacks.EarlyStopping('val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)

# Final Eval
loss, mae = model.evaluate(X_te, y_te, verbose=0)
rmse = np.sqrt(loss)
print(f"\nTest MSE: {loss:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

y_pred = model.predict(X_te)
print("Test R²:", r2_score(y_te, y_pred))

plt.plot(hist1.history['loss'] + hist2.history['loss'], label='Train MSE')
plt.plot(hist1.history['val_loss'] + hist2.history['val_loss'], label='Val MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()