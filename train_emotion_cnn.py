import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset path
dataset_path = r"C:\Users\prath\OneDrive\Documents\Desktop\emotion detection\processed"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

# Labels
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
label_map = {label: i for i, label in enumerate(labels)}

X, y = [], []

# Load images
for emotion in labels:
    folder = os.path.join(train_path, emotion)
    if not os.path.exists(folder):
        continue
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (48, 48)) / 255.0
        X.append(img)
        y.append(label_map[emotion])

X = np.array(X).reshape(-1, 48, 48, 1)
y = to_categorical(np.array(y), num_classes=7)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# CNN Model
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True)

# Train
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=30,
    validation_data=(X_val, y_val)
)

# Save model
model_path = r"C:\Users\prath\OneDrive\Documents\Desktop\emotion detection\emotion_cnn_model.h5"
model.save(model_path)
print(f"\nModel trained and saved at: {model_path}")
