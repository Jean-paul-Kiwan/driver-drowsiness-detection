import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# -------- Fake / simulated dataset --------
# [EAR, blinks_per_min, eye_closed_time]
X = np.array([
    [0.28, 18, 0.2],
    [0.30, 15, 0.1],
    [0.27, 20, 0.3],
    [0.18, 10, 1.8],
    [0.16, 8, 2.1],
    [0.17, 9, 1.6],
    [0.29, 16, 0.2],
    [0.15, 7, 2.3],
])

# Labels: 0 = Awake, 1 = Drowsy
y = np.array([0, 0, 0, 1, 1, 1, 0, 1])

# -------- Model --------
model = Sequential([
    Dense(8, activation='relu', input_shape=(3,)),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------- Train --------
model.fit(X, y, epochs=100, verbose=1)

# -------- Save model --------
model.save("drowsiness_model.h5")

print("✅ Model trained and saved as drowsiness_model.h5")
