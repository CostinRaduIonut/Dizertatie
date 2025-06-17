import os
import cv2
from collections import defaultdict
from uuid import uuid4

SOURCE_DIR = "detection/recognition_dataset_unorganized"
AUGMENTED_COUNT_PER_IMAGE = 2  # You can increase if needed
THRESHOLD = 20  # Classes with fewer than this count will be augmented

AUGMENTED_SUFFIXES = ["_aug1", "_aug2"]

def apply_augmentations(img):
    """Returns a list of augmented versions of the image."""
    aug1 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    aug2 = cv2.convertScaleAbs(img, alpha=1.1, beta=30)  # Brightness/contrast boost
    return [aug1, aug2]

# Step 1: Count occurrences of each class
class_counts = defaultdict(list)
for fname in os.listdir(SOURCE_DIR):
    if fname.endswith(".jpg") and "." in fname:
        label = fname.split(".")[0]
        class_counts[label].append(fname)

# Step 2: Augment rare classes
augmented = 0
for label, files in class_counts.items():
    if len(files) >= THRESHOLD:
        continue

    print(f"[Augmenting] Class '{label}' has only {len(files)} images.")

    for fname in files:
        path = os.path.join(SOURCE_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠ Could not read {path}")
            continue

        aug_imgs = apply_augmentations(img)
        for i, aug in enumerate(aug_imgs):
            new_fname = f"{label}.{str(uuid4()).split('-')[0]}{AUGMENTED_SUFFIXES[i]}.jpg"
            cv2.imwrite(os.path.join(SOURCE_DIR, new_fname), aug)
            augmented += 1

print(f"\n✅ Augmentation complete: {augmented} new images created.")