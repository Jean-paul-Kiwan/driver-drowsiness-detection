import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("drowsiness_log.csv")

print("Columns in dataset:")
print(df.columns)
print("Dataset size:", len(df))

# =========================
# FEATURES & LABEL
# =========================
FEATURES = [
    "EAR", "MAR",
    "yaw_delta_deg", "pitch_delta_deg", "roll_delta_deg",
    "blink_rate_per_min",
    "yawning", "head_down", "head_pose_warning"
]

X = df[FEATURES].values
y = (df["status"] == "DROWSY").astype(int).values

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# CLASS WEIGHTS (important!)
# =========================
# Handle imbalance: few DROWSY samples
neg, pos = np.bincount(y_train)
total = neg + pos

class_weight = {
    0: total / (2 * neg),
    1: total / (2 * pos)
}

# =========================
# MLP MODEL
# =========================
model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =========================
# TRAIN
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1
)

# =========================
# EVALUATION
# =========================
y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n=== MLP RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["AWAKE", "DROWSY"]))

# =========================
# SAVE MODEL
# =========================
model.save("mlp_model.h5")
print("\nModel saved as: mlp_model.h5")
