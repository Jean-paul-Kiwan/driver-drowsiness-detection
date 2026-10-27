import cv2
import mediapipe as mp
import math
import time
import winsound
import threading

# -------- MediaPipe setup --------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean_distance(p1, p2):
    return math.dist(p1, p2)

def eye_aspect_ratio(eye):
    A = euclidean_distance(eye[1], eye[5])
    B = euclidean_distance(eye[2], eye[4])
    C = euclidean_distance(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# -------- Parameters --------
EAR_THRESHOLD = 0.20
DROWSY_TIME = 1.5

# -------- State variables --------
closed_start_time = None
blink_counter = 0
total_blinks = 0
start_time = time.time()

drowsy = False
alarm_on = False
drowsy_events = 0

# -------- Alarm --------
def alarm_sound():
    global alarm_on
    while alarm_on:
        winsound.Beep(2500, 500)

# -------- Webcam --------
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
        for face_landmarks in results.multi_face_landmarks:

            left_eye, right_eye = [], []

            for idx in LEFT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                left_eye.append((x, y))

            for idx in RIGHT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                right_eye.append((x, y))

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

            # -------- Blink detection --------
            if ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= 2:
                    total_blinks += 1
                blink_counter = 0

            # -------- Drowsiness detection --------
            if ear < EAR_THRESHOLD:
                if closed_start_time is None:
                    closed_start_time = time.time()
                elif time.time() - closed_start_time >= DROWSY_TIME:
                    if not drowsy:
                        drowsy_events += 1
                    drowsy = True
            else:
                closed_start_time = None
                drowsy = False

            # -------- Alarm control --------
            if drowsy and not alarm_on:
                alarm_on = True
                threading.Thread(target=alarm_sound, daemon=True).start()
            elif not drowsy:
                alarm_on = False

            # -------- Statistics --------
            elapsed_time = time.time() - start_time
            bpm = (total_blinks / elapsed_time) * 60 if elapsed_time > 0 else 0

            # -------- Display --------
            color = (0, 0, 255) if drowsy else (0, 255, 0)
            status = "DROWSY!" if drowsy else "AWAKE"

            cv2.putText(frame, status, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, f"Blinks: {total_blinks}", (30, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, f"Blinks/min: {bpm:.1f}", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, f"Drowsy events: {drowsy_events}", (30, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Driver Drowsiness - Step 5", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

alarm_on = False
cap.release()
cv2.destroyAllWindows()
