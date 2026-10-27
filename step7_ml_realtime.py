import cv2
import mediapipe as mp
import math
import time
import numpy as np
import tensorflow as tf
import winsound
import threading

# -------- Load model --------
model = tf.keras.models.load_model("drowsiness_model.h5")

# -------- MediaPipe --------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def dist(p1, p2):
    return math.dist(p1, p2)

def ear(eye):
    return (dist(eye[1], eye[5]) + dist(eye[2], eye[4])) / (2 * dist(eye[0], eye[3]))

# -------- State --------
blink_counter = 0
total_blinks = 0
start_time = time.time()
closed_start = None
alarm_on = False

def alarm():
    global alarm_on
    while alarm_on:
        winsound.Beep(2500, 500)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0]
        le, re = [], []

        for i in LEFT_EYE:
            le.append((int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)))
        for i in RIGHT_EYE:
            re.append((int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)))

        ear_val = (ear(le) + ear(re)) / 2

        if ear_val < 0.20:
            blink_counter += 1
            if closed_start is None:
                closed_start = time.time()
        else:
            if blink_counter >= 2:
                total_blinks += 1
            blink_counter = 0
            closed_start = None

        elapsed = time.time() - start_time
        bpm = (total_blinks / elapsed) * 60 if elapsed > 0 else 0
        closed_time = time.time() - closed_start if closed_start else 0

        # -------- ML Prediction --------
        X = np.array([[ear_val, bpm, closed_time]])
        pred = model.predict(X, verbose=0)[0][0]

        drowsy = pred > 0.5

        if drowsy and not alarm_on:
            alarm_on = True
            threading.Thread(target=alarm, daemon=True).start()
        elif not drowsy:
            alarm_on = False

        status = "DROWSY (ML)" if drowsy else "AWAKE (ML)"
        color = (0,0,255) if drowsy else (0,255,0)

        cv2.putText(frame, status, (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Driver Drowsiness - ML", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

alarm_on = False
cap.release()
cv2.destroyAllWindows()
