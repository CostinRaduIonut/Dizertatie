import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# --- Constants ---
IMAGE_SIZE = (28, 28)
LABELS = list("abcdefghijklmnopqrstuvwxyz")  # 26 classes only
MODEL_PATH = "nn_braille"  # Adjust if needed

# --- Load Model ---
def load_braille_model(path=MODEL_PATH):
    return load_model(path)

# --- Preprocess Image ---
def preprocess_image(img_path):
    img = image.load_img(img_path, color_mode="grayscale", target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)  # Shape: (1, 28, 28, 1)

# --- Predict Single Image ---
def predict_braille_character(model, img_path):
    img_tensor = preprocess_image(img_path)
    predictions = model.predict(img_tensor)
    predicted_index = np.argmax(predictions[0])
    return LABELS[predicted_index], predicted_index

# --- Predict All in Folder (Optional Batch) ---
def predict_folder(model, folder_path):
    for file in os.listdir(folder_path):
        if file.lower().endswith(".jpg"):
            full_path = os.path.join(folder_path, file)
            label, index = predict_braille_character(model, full_path)
            print(f"{file}: {label} (class index {index})")

