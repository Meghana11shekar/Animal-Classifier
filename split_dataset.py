import os
import shutil
import random

source_folder = "C:\\Users\\Sarka\\animal_classifier\\raw-img"
classes = ["cat", "dog", "horse", "sheep", "cow", "elephant", "butterfly", "chicken", "spider", "squirrel"]
base_dir = "C:\\Users\\Sarka\\animal_classifier\\dataset"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
    
    path = os.path.join(source_folder, cls)
    images = os.listdir(path)
    random.shuffle(images)

    split_idx = int(0.8 * len(images))
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    for img in train_images:
        src = os.path.join(path, img)
        dst = os.path.join(train_dir, cls, img)
        shutil.copy(src, dst)

    for img in test_images:
        src = os.path.join(path, img)
        dst = os.path.join(test_dir, cls, img)
        shutil.copy(src, dst)

print("✅ Dataset split complete.")
