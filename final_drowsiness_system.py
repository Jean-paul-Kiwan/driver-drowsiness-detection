import cv2
import time
import csv
import numpy as np
import mediapipe as mp
import tensorflow as tf
import winsound

# ============================================================
# CONFIGURATION
# ============================================================
EAR_THRESHOLD = 0.21
EAR_DROWSY_TIME = 2.5          # seconds eyes must stay closed

MAR_THRESHOLD = 0.60
YAWN_TIME_THRESHOLD = 1.2      # seconds mouth must stay open

HEAD_BASELINE_TIME = 3.0       # seconds to calibrate head pose

# Head-down (based on pitch delta)
HEAD_DOWN_DELTA_DEG = 10.0     # degrees downward from baseline pitch
HEAD_DOWN_TIME_THRESHOLD = 2.0 # seconds sustained head down before alert

# Head pose "range" warning (based on deltas vs baseline)
HEAD_PITCH_LIMIT = 25.0        # degrees (delta from baseline)
HEAD_YAW_LIMIT   = 35.0        # degrees (delta from baseline)
HEAD_ROLL_LIMIT  = 30.0        # degrees (delta from baseline)
HEAD_POSE_TIME_THRESHOLD = 1.5 # must be bad pose for this long before warning

ML_THRESHOLD = 0.70            # strong confidence only

BEEP_COOLDOWN = 2.0
HEAD_BEEP_COOLDOWN = 1.5       # separate cooldown for head warning beep

CSV_FILE = "drowsiness_log.csv"
MODEL_FILE = "drowsiness_model.h5"

# ============================================================
# INITIALIZATION
# ============================================================
model = tf.keras.models.load_model(MODEL_FILE)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True, max_num_faces=1)

cap = cv2.VideoCapture(0)

start_time = time.time()
prev_time = start_time
last_beep = 0.0
last_head_beep = 0.0

blink_count = 0
blink_start = False

eye_closed_time = 0.0
mouth_open_time = 0.0
head_down_time = 0.0
head_pose_time = 0.0  # sustained bad head pose timer

awake_time = 0.0
drowsy_time = 0.0

# Head pose baseline samples
pitch_samples, yaw_samples, roll_samples = [], [], []
pitch_baseline = None
yaw_baseline = None
roll_baseline = None

# ============================================================
# CSV SETUP
# ============================================================
csv_file = open(CSV_FILE, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow([
    "time_s",
    "EAR", "MAR",
    "yaw_deg", "pitch_deg", "roll_deg",
    "yaw_delta_deg", "pitch_delta_deg", "roll_delta_deg",
    "blinks", "blink_rate_per_min",
    "ml_score", "status",
    "awake_time_s", "drowsy_time_s",
    "yawning", "head_down",
    "head_pose_warning"
])

# ============================================================
# LANDMARK INDICES
# ============================================================
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Face points for head pose (2D from MediaPipe)
POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]  # nose, chin, left eye, right eye, left mouth, right mouth

# Approximate 3D face model (mm)
MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),     # Nose tip
    (0.0,  -63.6,  -12.5),     # Chin
    (-43.3,  32.7,  -26.0),    # Left eye outer corner
    (43.3,  32.7,  -26.0),     # Right eye outer corner
    (-28.9, -28.9,  -24.1),    # Left mouth corner
    (28.9,  -28.9,  -24.1)     # Right mouth corner
], dtype=np.float64)

# ============================================================
# FUNCTIONS
# ============================================================
def distance(a, b):
    return np.linalg.norm(a - b)

def eye_aspect_ratio(landmarks, idxs, w, h):
    pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in idxs]
    v = distance(pts[1], pts[5]) + distance(pts[2], pts[4])
    h_ = distance(pts[0], pts[3])
    return v / (2 * h_) if h_ != 0 else 0.0

def mouth_aspect_ratio(landmarks, w, h):
    up = np.array([landmarks[13].x * w, landmarks[13].y * h])
    lo = np.array([landmarks[14].x * w, landmarks[14].y * h])
    l  = np.array([landmarks[61].x * w, landmarks[61].y * h])
    r  = np.array([landmarks[291].x * w, landmarks[291].y * h])
    horiz = distance(l, r)
    return distance(up, lo) / horiz if horiz != 0 else 0.0

def rotation_matrix_to_euler_degrees(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])  # roll
        y = np.arctan2(-R[2, 0], sy)      # pitch
        z = np.arctan2(R[1, 0], R[0, 0])  # yaw
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    roll  = np.degrees(x)
    pitch = np.degrees(y)
    yaw   = np.degrees(z)
    return yaw, pitch, roll

def estimate_head_pose(landmarks, w, h):
    image_points = []
    for idx in POSE_LANDMARKS:
        p = landmarks[idx]
        image_points.append((p.x * w, p.y * h))
    image_points = np.array(image_points, dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    R, _ = cv2.Rodrigues(rvec)
    return rotation_matrix_to_euler_degrees(R)

def angle_diff(a, b):
    d = a - b
    return (d + 180) % 360 - 180
# ============================
# UI COLORS (LIGHT THEME)
# ============================
PANEL_BG = (0, 0, 0)     # light background
TEXT_MAIN = (255, 255, 255)       # dark text
TEXT_TITLE = (180, 255, 255)       # dark text
TEXT_MUTED = (90, 90, 90)
GOOD = (0, 160, 0)
BAD = (0, 0, 200)
WARN_YELLOW = (0, 140, 255)


# ============================================================
# UI HELPERS (SIDEBAR LAYOUT)
# ============================================================
def draw_panel(img, x, y, w, h, alpha=0.45):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), PANEL_BG, -1)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def put(img, text, x, y, scale=0.45, color=None, thickness=0.5):
    if color is None:
        color = TEXT_MAIN
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )

def label_value(img, label, value, x, y, good_color=TEXT_MAIN, bad=False):

    color = (0, 0, 255) if bad else good_color
    put(img, f"{label}: {value}", x, y, 0.58, color, 2)


# ============================================================
# MAIN LOOP
# ============================================================
WINDOW_NAME = "Driver Drowsiness Detection - FINAL"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]

    # Layout config
    SIDEBAR_W = 380
    GAP = 12
    # Make a big canvas: left sidebar + camera feed
    canvas = np.zeros((frame_h, frame_w + SIDEBAR_W + GAP, 3), dtype=np.uint8)

    # Place camera feed on the right
    cam_x = SIDEBAR_W + GAP
    canvas[:, cam_x:cam_x + frame_w] = frame

    # For all drawings (dots/text) we draw on the camera region inside canvas
    cam_view = canvas[:, cam_x:cam_x + frame_w]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    now = time.time()
    dt = now - prev_time
    prev_time = now
    elapsed = now - start_time

    status = "NO FACE"
    ear = mar = 0.0
    yaw_deg = pitch_deg = roll_deg = 0.0
    yaw_delta = pitch_delta = roll_delta = 0.0
    ml_score = 0.0
    yawning = False
    head_drowsy = False
    head_pose_warning = False

    # For warning box list
    warnings = []

    # ========================================================
    # DETECTION PIPELINE
    # ========================================================
    if results.multi_face_landmarks:
        status = "AWAKE"
        lm = results.multi_face_landmarks[0].landmark

        # EAR / MAR
        ear = (eye_aspect_ratio(lm, LEFT_EYE, frame_w, frame_h) +
               eye_aspect_ratio(lm, RIGHT_EYE, frame_w, frame_h)) / 2.0
        mar = mouth_aspect_ratio(lm, frame_w, frame_h)

        # VISUAL DOTS (keep them)
        for i in LEFT_EYE + RIGHT_EYE:
            p = lm[i]
            cv2.circle(cam_view, (int(p.x * frame_w), int(p.y * frame_h)), 2, (0, 255, 0), -1)

        for i in [13, 14, 61, 291]:
            p = lm[i]
            cv2.circle(cam_view, (int(p.x * frame_w), int(p.y * frame_h)), 2, (0, 255, 255), -1)

        for i in POSE_LANDMARKS:
            p = lm[i]
            cv2.circle(cam_view, (int(p.x * frame_w), int(p.y * frame_h)), 3, (255, 0, 255), -1)

        # Blink count
        if ear < EAR_THRESHOLD:
            if not blink_start:
                blink_start = True
        else:
            if blink_start:
                blink_count += 1
                blink_start = False

        blink_rate = (blink_count / elapsed) * 60 if elapsed > 0 else 0.0

        # Eye-closed timer
        if ear < EAR_THRESHOLD:
            eye_closed_time += dt
        else:
            eye_closed_time = 0.0

        if eye_closed_time > EAR_DROWSY_TIME:
            warnings.append("⚠ EYES CLOSED TOO LONG")

        # Yawning
        if mar > MAR_THRESHOLD:
            mouth_open_time += dt
        else:
            mouth_open_time = 0.0
        yawning = mouth_open_time > YAWN_TIME_THRESHOLD
        if yawning:
            warnings.append("⚠ YAWNING DETECTED")

        # Head pose
        pose = None
        try:
            pose = estimate_head_pose(lm, frame_w, frame_h)
        except Exception:
            pose = None

        if pose is not None:
            yaw_deg, pitch_deg, roll_deg = pose

            # Calibration baseline
            if pitch_baseline is None:
                pitch_samples.append(pitch_deg)
                yaw_samples.append(yaw_deg)
                roll_samples.append(roll_deg)

                if elapsed > HEAD_BASELINE_TIME:
                    pitch_baseline = float(np.mean(pitch_samples))
                    yaw_baseline = float(np.mean(yaw_samples))
                    roll_baseline = float(np.mean(roll_samples))
            else:
                pitch_delta = pitch_deg - pitch_baseline
                yaw_delta = angle_diff(yaw_deg, yaw_baseline)
                roll_delta = angle_diff(roll_deg, roll_baseline)

                out_of_range = (
                    abs(pitch_delta) > HEAD_PITCH_LIMIT or
                    abs(yaw_delta)   > HEAD_YAW_LIMIT or
                    abs(roll_delta)  > HEAD_ROLL_LIMIT
                )

                if out_of_range:
                    head_pose_time += dt
                else:
                    head_pose_time = 0.0

                head_pose_warning = head_pose_time > HEAD_POSE_TIME_THRESHOLD
                if head_pose_warning:
                    warnings.append("⚠ HEAD POSE OUT OF RANGE")

                if pitch_delta > HEAD_DOWN_DELTA_DEG:
                    head_down_time += dt
                else:
                    head_down_time = 0.0

                head_drowsy = head_down_time > HEAD_DOWN_TIME_THRESHOLD
                if head_drowsy:
                    warnings.append("⚠ HEAD DOWN (SUSTAINED)")

        # Head warning beep (only after calibration + sustained)
        if (pitch_baseline is not None) and head_pose_warning and (now - last_head_beep) > HEAD_BEEP_COOLDOWN:
            winsound.Beep(1800, 600)
            last_head_beep = now

        # ML score
        features = np.array([[ear, mar, blink_rate]], dtype=np.float32)
        ml_score = float(model.predict(features, verbose=0)[0][0])

        # Final decision
        drowsy_rule = (eye_closed_time > EAR_DROWSY_TIME) or yawning or head_drowsy
        ml_rule = (ml_score > ML_THRESHOLD)
        if ml_score > ML_THRESHOLD:
           warnings.append("⚠ ML MODEL: HIGH DROWSINESS CONFIDENCE")

        if drowsy_rule or ml_rule:
            status = "DROWSY"
            drowsy_time += dt
        else:
            status = "AWAKE"
            awake_time += dt

        # Drowsy beep
        if status == "DROWSY" and (now - last_beep) > BEEP_COOLDOWN:
            winsound.Beep(2500, 700)
            last_beep = now

        # CSV
        writer.writerow([
            round(elapsed, 2),
            round(ear, 4), round(mar, 4),
            round(yaw_deg, 2), round(pitch_deg, 2), round(roll_deg, 2),
            round(yaw_delta, 2), round(pitch_delta, 2), round(roll_delta, 2),
            blink_count, round(blink_rate, 2),
            round(ml_score, 4), status,
            round(awake_time, 2), round(drowsy_time, 2),
            int(yawning), int(head_drowsy),
            int(head_pose_warning)
        ])

    # ========================================================
    # DRAW UI (LEFT SIDEBAR + TOP BAR)
    # ========================================================
    # Sidebar background
    draw_panel(canvas, 0, 0, SIDEBAR_W, frame_h, alpha=0.55)

    # Top status bar (over camera area too)
    draw_panel(canvas, 0, 0, SIDEBAR_W + GAP + frame_w, 60, alpha=0.55)

    status_color = (0, 255, 0) if status == "AWAKE" else ((0, 0, 255) if status == "DROWSY" else (0, 255, 255))
    put(canvas, f"SYSTEM STATUS: {status}", 18, 42, 1.05, status_color, 3)

    # Calibration hint
    if results.multi_face_landmarks and pitch_baseline is None:
        put(canvas, f"Calibrating head pose... keep neutral ({HEAD_BASELINE_TIME:.0f}s)", 390, 42, 0.8, (255, 255, 0), 2)

    # Sidebar sections
    y = 75
    put(canvas, "DRIVER METRICS", 18, y, 0.60, TEXT_TITLE, 2)
    y += 30

    label_value(canvas, "EAR", f"{ear:.2f}", 18, y, bad=(ear > 0 and ear < EAR_THRESHOLD))
    y += 20
    label_value(canvas, "MAR", f"{mar:.2f}", 18, y, bad=(mar > MAR_THRESHOLD))
    y += 20

    blink_rate_display = (blink_count / elapsed) * 60 if elapsed > 0 else 0.0
    label_value(canvas, "Blinks", f"{blink_count}", 18, y)
    y += 20
    label_value(canvas, "Blink rate", f"{blink_rate_display:.1f} /min", 18, y)
    y += 20

    label_value(canvas, "ML score", f"{ml_score:.2f}", 18, y, bad=(ml_score > ML_THRESHOLD))
    y += 20

    label_value(canvas, "Awake time", f"{awake_time:.1f}s", 18, y)
    y += 20
    label_value(canvas, "Drowsy time", f"{drowsy_time:.1f}s", 18, y)
    y += 28

    put(canvas, "HEAD ORIENTATION", 18, y, 0.60, TEXT_TITLE, 2)
    y += 30

    label_value(canvas, "Yaw", f"{yaw_deg:+.1f}°", 18, y)
    y += 20
    label_value(canvas, "Pitch", f"{pitch_deg:+.1f}°", 18, y)
    y += 20
    label_value(canvas, "Roll", f"{roll_deg:+.1f}°", 18, y)
    y += 30

    # Deltas (highlight if exceed limits)
    if pitch_baseline is not None:
        label_value(canvas, "ΔYaw", f"{yaw_delta:+.1f}°", 18, y, bad=(abs(yaw_delta) > HEAD_YAW_LIMIT))
        y += 20
        label_value(canvas, "ΔPitch", f"{pitch_delta:+.1f}°", 18, y, bad=(abs(pitch_delta) > HEAD_PITCH_LIMIT))
        y += 20
        label_value(canvas, "ΔRoll", f"{roll_delta:+.1f}°", 18, y, bad=(abs(roll_delta) > HEAD_ROLL_LIMIT))
        y += 28
    else:
        put(canvas, "ΔYaw / ΔPitch / ΔRoll: waiting baseline...", 18, y, 0.58, (200, 200, 200), 2)
        y += 28

    put(canvas, "WARNINGS", 18, y, 0.60, TEXT_TITLE, 2)
    y += 30

    if len(warnings) == 0:
        put(canvas, "No active warnings.", 18, y, 0.62, (180, 180, 180), 2)
    else:
        for wtxt in warnings[:4]:
            put(canvas, wtxt, 18, y, 0.62, (0, 0, 255), 2)
            y += 20

    # ========================================================
    # OPTIONAL: minimal on-camera warning banner
    # ========================================================
    if status == "DROWSY":
        cv2.rectangle(cam_view, (0, frame_h - 55), (frame_w, frame_h), (0, 0, 255), -1)
        put(cam_view, "DROWSINESS ALERT! Please take a break.", 14, frame_h - 18, 0.8, (255, 255, 255), 2)

    # Show everything
    cv2.imshow(WINDOW_NAME, canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

# ============================================================
# CLEANUP
# ============================================================
csv_file.close()
cap.release()
cv2.destroyAllWindows()

# python final_drowsiness_system.py