import os
import shutil
import random

# Directory where your images are located
source_dir = "./temp"

# Target directory
target_dir = r"detection/recognition_dataset"

# Create the train/validation split directories
train_dir = os.path.join(target_dir, 'train')
val_dir = os.path.join(target_dir, 'validation')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get all image filenames
all_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
print(all_files)

# Group files by class
class_files = {}
for file in all_files:
    class_label = file.split('.')[0][0]  # Extract the letter from the file name (e.g., a1.JPG0dim -> a)
    if class_label not in class_files:
        class_files[class_label] = []
    class_files[class_label].append(file)

# Create class folders and split the files
for class_label, files in class_files.items():
    random.shuffle(files)
    split_index = int(len(files) * 0.75)
    train_files = files[:split_index]
    val_files = files[split_index:]

    # Create directories for this class
    os.makedirs(os.path.join(train_dir, class_label), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_label), exist_ok=True)

    # Copy files
    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, class_label, file))
    for file in val_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(val_dir, class_label, file))

print("Dataset split complete!")
