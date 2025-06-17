import os
import shutil

SOURCE_DIR = "detection/recognition_dataset_unorganized"
DEST_DIR = "detection/recognition_dataset"
os.makedirs(DEST_DIR, exist_ok=True)

for file in os.listdir(SOURCE_DIR):
    if not file.endswith(".jpg"):
        continue
    label = file.split(".")[0]
    label_dir = os.path.join(DEST_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    src = os.path.join(SOURCE_DIR, file)
    dst = os.path.join(label_dir, file)
    shutil.copy2(src, dst)

print("✅ Dataset reorganized by class.")
