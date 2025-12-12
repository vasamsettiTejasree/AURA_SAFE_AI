# aura_safe_ai.py
import os
import time
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = r"C:\Users\prath\OneDrive\Documents\Desktop\emotion detection\emotion_cnn_model.h5"
INCIDENTS_DIR = r"C:\Users\prath\OneDrive\Documents\Desktop\emotion detection\incidents"
BLUR_KERNEL = (55, 55)      # Larger -> stronger blur
BUFFER_SIZE = 5             # smoothing buffer for emotion
FALL_DROP_THRESH = 0.12     # hip y drop threshold (tweakable)
FALL_ANGLE_THRESH = 0.30    # shoulder-hip vertical difference threshold (tweakable)
FALL_HOLD_SECONDS = 3       # how long alert stays before reset

# Create incidents folder if not exists
os.makedirs(INCIDENTS_DIR, exist_ok=True)

# ---------------------------
# Load Emotion Model
# ---------------------------
print("Loading emotion model...")
model = load_model(MODEL_PATH)  # ensure this path exists
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ---------------------------
# Initialize Pose (MediaPipe)
# ---------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---------------------------
# Webcam
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

# ---------------------------
# State
# ---------------------------
emotion_buffer = deque(maxlen=BUFFER_SIZE)
last_hip_y = None
fall_detected = False
fall_start_time = 0

# Optional: sound alert (Windows)
try:
    import winsound

    def beep_alert():
        # short beep; you can customize frequency/duration
        winsound.Beep(1000, 400)
except Exception:
    def beep_alert():
        # fallback: print
        print("[ALERT] Fall detected (no sound available).")

# ---------------------------
# Helper Functions
# ---------------------------
def preprocess_face(gray_face):
    face = cv2.resize(gray_face, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=(0, -1))
    return face

def save_incident(frame):
    t = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(INCIDENTS_DIR, f"fall_{t}.jpg")
    cv2.imwrite(path, frame)
    print("Saved incident image to:", path)

# ---------------------------
# Main Loop
# ---------------------------
print("Starting Aura Safe AI (press 'q' to quit)...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_frame = frame.copy()
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- Face detection (Haar) ----
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Default values shown on UI
    ui_emotion = "N/A"
    ui_fall = "No"

    # Predict emotion for each face, blur faces
    for (x, y, fw, fh) in faces:
        # crop and preprocess
        roi_gray = gray[y:y+fh, x:x+fw]
        if roi_gray.size == 0:
            continue
        face_input = preprocess_face(roi_gray)

        # predict
        preds = model.predict(face_input, verbose=0)
        emotion = labels[int(np.argmax(preds))]
        emotion_buffer.append(emotion)
        # smoothing
        ui_emotion = Counter(emotion_buffer).most_common(1)[0][0]

        # blur face region for privacy
        face_region = frame[y:y+fh, x:x+fw]
        try:
            blurred_face = cv2.GaussianBlur(face_region, BLUR_KERNEL, 30)
        except Exception:
            # fallback smaller kernel
            blurred_face = cv2.GaussianBlur(face_region, (23, 23), 10)
        frame[y:y+fh, x:x+fw] = blurred_face

        # Put emotion text above blurred face
        cv2.putText(frame, ui_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    # ---- Pose detection (MediaPipe) ----
    rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Mediapipe uses normalized coords; compute average hip and shoulder
        left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
        hip_y = (left_hip.y + right_hip.y) / 2.0

        # Convert to pixel coordinates if needed
        shoulder_px_y = shoulder_y * h
        hip_px_y = hip_y * h

        # compute "drop" (positive means hip moved downward on screen)
        if last_hip_y is not None:
            drop = hip_y - last_hip_y  # normalized units (0..1)
            # body orientation: vertical difference between shoulders and hips
            body_vert_diff = abs(shoulder_y - hip_y)

            # Simple fall condition: sudden downward hip movement AND body is roughly horizontal
            if (drop > FALL_DROP_THRESH) and (body_vert_diff < FALL_ANGLE_THRESH):
                if not fall_detected:
                    fall_detected = True
                    fall_start_time = time.time()
                    ui_fall = "YES"
                    # alert: beep and save image
                    beep_alert()
                    save_incident(frame)
            # if fall not detected yet, keep ui_fall No
            else:
                # If fall currently active, check hold time
                if fall_detected:
                    elapsed = time.time() - fall_start_time
                    if elapsed > FALL_HOLD_SECONDS:
                        fall_detected = False
                ui_fall = "YES" if fall_detected else "No"

        last_hip_y = hip_y

        # draw small pose skeleton for debug (optional)
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )

    # Overlay UI text: top-left
    cv2.rectangle(frame, (0, 0), (300, 70), (0, 0, 0), -1)  # background
    cv2.putText(frame, f"Emotion: {ui_emotion}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Fall: {ui_fall}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if ui_fall == "YES" else (255, 255, 255), 2)

    cv2.imshow("AURA SAFE AI (Emotion + Privacy Blur + Simple Fall)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
pose.close()
