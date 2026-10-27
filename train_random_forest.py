import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# =========================
# CONFIG
# =========================
CSV_FILE = "drowsiness_log.csv"
MODEL_OUT = "random_forest_model.pkl"

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv(CSV_FILE)

print("Columns in dataset:")
print(df.columns)

# =========================
# FEATURE SELECTION
# =========================
FEATURES = [
    "EAR",
    "MAR",
    "blink_rate_per_min",
    "yaw_delta_deg",
    "pitch_delta_deg",
    "roll_delta_deg"
]

LABEL = "status"

# Keep only required columns
df = df[FEATURES + [LABEL]]

# =========================
# CLEAN DATA
# =========================
df = df.dropna()

# Convert labels to binary
# AWAKE = 0, DROWSY = 1
df[LABEL] = df[LABEL].map({"AWAKE": 0, "DROWSY": 1})

# Remove any leftover invalid rows
df = df.dropna()

X = df[FEATURES].values
y = df[LABEL].values

print(f"Dataset size: {len(X)} samples")

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =========================
# SCALE FEATURES
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# RANDOM FOREST MODEL
# =========================
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n=== RANDOM FOREST RESULTS ===")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["AWAKE", "DROWSY"]))

# =========================
# SAVE MODEL
# =========================
joblib.dump({
    "model": rf,
    "scaler": scaler,
    "features": FEATURES
}, MODEL_OUT)

print(f"\nModel saved as: {MODEL_OUT}")
