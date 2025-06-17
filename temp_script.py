import cv2
import os

# Set your input and output directories
input_dir = 'images_output/'
output_dir = 'braille_detectat/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# Process all files in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(image_extensions):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Read and rotate the image
        img = cv2.imread(input_path)
        if img is None:
            print(f"[Warning] Failed to load image: {filename}")
            continue

        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        # Save the rotated image
        cv2.imwrite(output_path, rotated_img)
        print(f"Saved rotated image: {output_path}")
