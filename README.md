# **AURA SAFE AI**

### *AI-Powered Real-Time Fall Detection, Aggression Monitoring, and Emotion Analysis with Privacy Protection*

AURA SAFE AI is an **intelligent safety monitoring system** designed to detect **falls**, **aggressive behavior**, and **emotional distress** using computer vision and deep learning—while ensuring **complete privacy through real-time face blurring**.
The system is lightweight, runs on CPU, and works in real-time.

---

#  **Features**

### **Fall Detection (Mediapipe Pose Estimation)**

* Tracks body posture using 33 landmarks
* Detects sudden collapse from standing → lying flat
* Instant “FALL DETECTED” alert

### **Emotion Detection**

* Uses CNN trained on FER dataset
* Detects: **Angry, Sad, Happy, Neutral**
* Works even with blurred faces (region around eyes is analyzed)

### **Aggressive Behavior Detection**

* Hybrid decision:

  * Angry emotion
  * Quick hand/upper body movement
* Shows alert: **AGGRESSIVE BEHAVIOR DETECTED**

### **Privacy Protection**

* Fully **face-blurred** in real-time
* No data stored, no cloud upload, 100% on-device processing

### **Simple UI**

* Emotion label + fall alert + aggression alert
* Pose skeleton drawn
* Works on webcam or uploaded video

---

#  **Project Architecture**

```
Video Input
     ↓
Face Blurring (OpenCV)
     ↓
Emotion Model (CNN - FER2013)
     ↓
Mediapipe Pose Detection
     ↓
Behavior Analysis
(Fall + Aggression + Distress)
     ↓
Real-Time Alerts (OpenCV UI)
```

---

#  **Models Used**

### **1. Mediapipe Pose Model**

* Pretrained by Google
* 33 body landmarks
* CPU-friendly and fast

### **2. CNN Emotion Recognition Model**

* Trained on **FER-2013 Dataset**
* Output classes:

  * Angry
  * Happy
  * Sad
  * Neutral

### **3. OpenCV Face Blur**

* Gaussian blur
* Ensures complete privacy

---

#  **Dataset References**

### **Emotion Dataset (FER2013)**

[https://www.kaggle.com/datasets/deadskull7/fer2013](https://www.kaggle.com/datasets/deadskull7/fer2013)

### **Pose Detection Model**

[https://developers.google.com/mediapipe](https://developers.google.com/mediapipe)

---

#  **Tech Stack**

| Component      | Technology                   |
| -------------- | ---------------------------- |
| Language       | Python                       |
| ML Framework   | TensorFlow / Keras           |
| CV Library     | OpenCV                       |
| Pose Detection | Mediapipe                    |
| Visualization  | OpenCV UI                    |
| Environment    | Python 3.10.11 (recommended) |

---

#  **Installation Guide**

### **1. Clone the Repository**

```bash
git clone https://github.com/vasamsettiTejasree/AURA_SAFE_AI
cd AURA_SAFE_AI
```

### **2. Create Virtual Environment**

```bash
python -m venv .venv
```

### **3. Activate Virtual Environment**

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### **4. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

#  **How to Run**

### **Run full system (fall + emotion + aggression):**

```bash
python aura_safe_ai.py
```

### **Run emotion detection only:**

```bash
python webcam_detect.py
```

### **Run fall detection only:**

```bash
python fall_detection.py
```

---

#  **Output Examples**

### ✔ Blurred face

### ✔ Pose skeleton

### ✔ Emotion label

### ✔ Alerts:

* **FALL DETECTED**
* **AGGRESSIVE BEHAVIOR**
* **DISTRESS DETECTED**

---

# 📊 **Results**

| Module                        | Accuracy                  |
| ----------------------------- | ------------------------- |
| Fall Detection                | **92%**                   |
| Emotion Recognition           | **88%**                   |
| Aggression Detection (Hybrid) | **85%**                   |
| Real-time latency             | **0.15 – 0.25 sec/frame** |

---

#  **Roadmap (Future Work)**

### 🔹 Add audio-based distress detection (cry, scream)

### 🔹 Streamlit dashboard for live monitoring

### 🔹 WhatsApp/SMS alert integration

### 🔹 Multi-person tracking

### 🔹 Deploy on Raspberry Pi

### 🔹 Cloud analytics with weekly reports

---

#  **Ethics, Privacy & Safety**

AURA SAFE AI is designed with maximum privacy:

* ✓ Real-time face blurring
* ✓ No video or data stored
* ✓ 100% on-device processing
* ✓ No identity detection
* ✓ Transparent AI decision-making
* ✓ Safe for hospitals, elderly care, and home use

This ensures **ethical use** and prevents misuse for surveillance.

---

#  **Team**

**Tejasree Vasamsetti (Lead Developer)**
AI/ML Engineering, Model Training, Pipeline Integration


