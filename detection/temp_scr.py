import os
import cv2 as cv
import numpy as np
import string
from tkinter import Tk, Label, Canvas
from PIL import Image, ImageTk
from ultralytics import YOLO
from uuid import uuid4
import json

# Paths
MODEL_PATH = "runs/detect/train6_good/weights/best.pt"
SOURCE_DIR = "braille_detectat"
DEST_DIR = "detection/recognition_dataset_unorganized"
os.makedirs(DEST_DIR, exist_ok=True)

# Grid recognizer constants
KERNEL = np.ones((3, 3), np.uint8)
BRAILLE_DICT = {
    "100000": "a", "110000": "b", "100100": "c", "100110": "d", "100010": "e",
    "110100": "f", "110110": "g", "110010": "h", "010100": "i", "010110": "j",
    "101000": "k", "111000": "l", "101100": "m", "101110": "n", "101010": "o",
    "111100": "p", "111110": "q", "111010": "r", "011100": "s", "011110": "t",
    "101001": "u", "111001": "v", "010111": "w", "101101": "x", "101111": "y",
    "101011": "z"
}

# --- Grid-based recognizer functions ---

def extract_circles(img_braille, img_area):
    dilated = cv.dilate(img_braille, KERNEL, iterations=1)
    _, thresh = cv.threshold(dilated, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        (x, y), radius = cv.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        rect_x, rect_y, w, h = cv.boundingRect(cnt)
        rect_area = w * h
        if radius > 2 and (rect_area / img_area) < 0.95 and center[0] > 2 and center[1] > 2:
            candidates.append((center, radius))
    return candidates

def decode_braille(candidates, width, height):
    points_normalized = [(x / width, y / height) for (x, y), _ in candidates]
    temp = []
    for x, y in points_normalized:
        col = 0 if x < .5 else 1
        if y < 0.4: row = 0
        elif y < 0.75: row = 1
        else: row = 2
        temp.append((col, row))

    temp.reverse()
    coordinates = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    encoding = "".join("1" if (c, r) in temp else "0" for c, r in coordinates)
    return BRAILLE_DICT.get(encoding, "*")

# --- Main Auto Annotator GUI ---

class AutoAnnotator:
    def __init__(self, root):
        self.root = root
        self.model = YOLO(MODEL_PATH)
        self.image_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.png'))]
        self.current_file_index = 0
        self.checkpoint_file = "checkpoint.json"
        self.load_checkpoint()
        self.boxes = []
        self.box_index = 0
        self.current_image = None
        self.history = []

        self.canvas = Canvas(root, width=300, height=300)
        self.canvas.pack()

        self.label = Label(root, text="Auto-labeling Braille cells...")
        self.label.pack()

        self.load_next_image()

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r") as f:
                data = json.load(f)
                self.current_file_index = data.get("current_file_index", 0)

    def save_checkpoint(self):
        data = {"current_file_index": self.current_file_index}
        with open(self.checkpoint_file, "w") as f:
            json.dump(data, f)

    def load_next_image(self):
        while self.current_file_index < len(self.image_files):
            filename = self.image_files[self.current_file_index]
            filepath = os.path.join(SOURCE_DIR, filename)
            print(f"\n[Processing] {filename}")
            self.current_image = cv.imread(filepath)
            result = self.model.predict(source=self.current_image, imgsz=640, verbose=False)
            self.boxes = result[0].boxes.xyxy.tolist()
            self.box_index = 0
            if self.boxes:
                self.process_boxes()
            self.current_file_index += 1
            self.save_checkpoint()
        self.label.config(text="Done labeling all images.")
        print("\n[✔] Finished all available images.")
        self.root.quit()

    def process_boxes(self):
        for box in self.boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = self.current_image[y1:y2, x1:x2]
            gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
            img_area = gray.shape[0] * gray.shape[1]
            candidates = extract_circles(gray, img_area)
            label = decode_braille(candidates, gray.shape[1], gray.shape[0])
            if label == "*":
                label = "unknown"
            filename = f"{label}.{str(uuid4()).split('-')[0]}.jpg"
            save_path = os.path.join(DEST_DIR, filename)
            cv.imwrite(save_path, crop)
            with open("autolabel_log.csv", "a") as logf:
                logf.write(f"{filename},{label}\n")

            # Optional visual (can be disabled for speed)
            crop_vis = cv.resize(crop, (150, 150))
            crop_img = Image.fromarray(cv.cvtColor(crop_vis, cv.COLOR_BGR2RGB))
            self.tk_image = ImageTk.PhotoImage(crop_img)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
            self.label.config(text=f"Labeled: {label.upper()} ({self.box_index+1}/{len(self.boxes)})")
            self.root.update()
            self.box_index += 1

if __name__ == "__main__":
    root = Tk()
    root.title("Braille Auto Annotator")
    app = AutoAnnotator(root)
    root.mainloop()
