import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --------------------------
# Load your trained model
# --------------------------
model_path = r"C:\Users\prath\OneDrive\Documents\Desktop\emotion detection\emotion_cnn_model.h5"
model = load_model(model_path)

# Emotion labels
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --------------------------
# Load face detector
# --------------------------
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --------------------------
# Start Webcam
# --------------------------
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # --------------------------
        # Extract face for emotion model
        # --------------------------
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float32") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Predict emotion
        preds = model.predict(roi_gray)
        emotion = labels[np.argmax(preds)]

        # --------------------------
        # Blur the face for privacy
        # --------------------------
        face_region = frame[y:y + h, x:x + w]
        blurred_face = cv2.GaussianBlur(face_region, (55, 55), 30)
        frame[y:y + h, x:x + w] = blurred_face

        # --------------------------
        # Write emotion above blurred face
        # --------------------------
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection (Blurring Enabled)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
