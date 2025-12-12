import os

# 📌 CHANGE THIS TO YOUR CORRECT PATH
base_path = r"C:\Users\prath\OneDrive\Documents\Desktop\emotion detection\processed"

for folder in ["train", "test"]:
    folder_path = os.path.join(base_path, folder)
    total_images = 0

    if not os.path.exists(folder_path):
        print(f"{folder} folder NOT found!")
        continue

    print(f"\nChecking folder: {folder}")

    # Loop inside each emotion folder
    for emotion in os.listdir(folder_path):
        emotion_path = os.path.join(folder_path, emotion)

        if os.path.isdir(emotion_path):
            # Count only image files
            images = [
                f for f in os.listdir(emotion_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            print(f"  {emotion}: {len(images)} images")

            total_images += len(images)

    print(f"\n👉 Total images in {folder}: {total_images}")


# Checking folder: train
#   angry: 3995 images
#   disgust: 436 images
#   fear: 4097 images
#   happy: 7215 images
#   neutral: 4965 images
#   sad: 4830 images
#   surprise: 3171 images

# 👉 Total images in train: 28709

# Checking folder: test
#   angry: 958 images
#   disgust: 111 images
#   fear: 1024 images
#   happy: 1774 images
#   neutral: 1233 images
#   sad: 1247 images
#   surprise: 831 images

# 👉 Total images in test: 7178