import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# =========================
# LOAD DATASET
# =========================
CSV_FILE = "drowsiness_log.csv"

df = pd.read_csv(CSV_FILE)

print("Columns in dataset:")
print(df.columns)
print(f"Dataset size: {len(df)} samples\n")

# =========================
# SELECT FEATURES
# =========================
FEATURES = [
    "EAR",
    "MAR",
    "blink_rate_per_min",
    "yaw_delta_deg",
    "pitch_delta_deg",
    "roll_delta_deg"
]

X = df[FEATURES]

# Target label
y = df["status"].map({"AWAKE": 0, "DROWSY": 1})

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
# FEATURE SCALING (IMPORTANT FOR SVM)
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# =========================
# TRAIN SVM
# =========================
svm_model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)

# =========================
# EVALUATION
# =========================
y_pred = svm_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["AWAKE", "DROWSY"])

print("=== SVM RESULTS ===")
print(f"Accuracy: {accuracy * 100:.2f}%\n")

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

# =========================
# SAVE MODEL
# =========================
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(scaler, "svm_scaler.pkl")

print("Model saved as: svm_model.pkl")
print("Scaler saved as: svm_scaler.pkl")
