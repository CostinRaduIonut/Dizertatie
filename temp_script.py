import os
import shutil

# Directory where your images are located
source_dir = "./temp"

# Output directory
output_dir = "./detection/recognition_dataset"

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all files in the source directory
all_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for filename in all_files:
    # Extract the class name prefix (e.g., "a1" or "b1")
    class_code = filename.split('.')[0]  # removes extension and everything after
    if not class_code:
        continue
    letter = class_code[0].lower()  # first character → 'a', 'b', etc.

    # Destination folder
    class_folder = os.path.join(output_dir, letter)
    os.makedirs(class_folder, exist_ok=True)

    # Move file
    src_path = os.path.join(source_dir, filename)
    dst_path = os.path.join(class_folder, filename)
    shutil.move(src_path, dst_path)

print("Done. Files have been sorted into class folders.")
