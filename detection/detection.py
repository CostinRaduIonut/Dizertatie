from ultralytics import YOLO
import cv2 as cv
import numpy as np
import os

# Constants
MODEL_PATH = "../runs/detect/train6_good/weights/best.pt"
BRAILLE_DIR = "braille_detectat/"
KERNEL = np.ones((3, 3), np.uint8)
model = YOLO(MODEL_PATH)
BRAILLE_DICT = {
    "100000": "a", "110000": "b", "100100": "c", "100110": "d", "100010": "e",
    "110100": "f", "110110": "g", "110010": "h", "010100": "i", "010110": "j",
    "101000": "k", "111000": "l", "101100": "m", "101110": "n", "101010": "o",
    "111100": "p", "111110": "q", "111010": "r", "011100": "s", "011110": "t",
    "101001": "u", "111001": "v", "010111": "w", "101101": "x", "101111": "y",
    "101011": "z"
}

def sort_boxes(boxes, y_threshold=15):
    if not boxes:
        return []
    boxes = np.array(boxes)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        print("[Warning] Detected boxes have invalid shape:", boxes.shape)
        return []

    boxes = boxes[boxes[:, 1].argsort()]  # Sort top-down
    rows, current_row = [], [boxes[0]]

    for box in boxes[1:]:
        if abs(box[1] - current_row[-1][1]) < y_threshold:
            current_row.append(box)
        else:
            rows.append(sorted(current_row, key=lambda b: b[0]))  # Sort left-right
            current_row = [box]
    rows.append(sorted(current_row, key=lambda b: b[0]))

    # Convert to nested list of lists
    return [[b.tolist() for b in row] for row in rows]


def extract_circles(img_braille, img_area):
    # Preprocess
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
        if radius > 2 and (rect_area / img_area) < 0.95 and center[0] > 4 and center[1] > 4:
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

def process_image(img_array, model):
    img = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)  # Convert to grayscale
    img_display = cv.cvtColor(img_array.copy(), cv.COLOR_BGR2RGB)  # For annotation

    result = model.predict(source=img_array, imgsz=640, verbose=False)  # Direct image input
    boxes = result[0].boxes.xyxy.tolist()
    if not boxes:
        print(f"[Info] No Braille boxes detected.")
        return None

    rows = sort_boxes(boxes)
    full_text = ""

    for row in rows:
        row_text = ""
        prev_x2 = None
        for box in row:
            x1, y1, x2, y2 = map(int, box)
            braille_cell = img[y1:y2, x1:x2]
            img_area = braille_cell.shape[0] * braille_cell.shape[1]
            cv.rectangle(img_display, (x1, y1), (x2, y2), (255, 0, 0), 2)
            candidates = extract_circles(braille_cell, img_area)

            for (cx, cy), r in candidates:
                cv.circle(img_display, (int(cx + x1), int(cy + y1)), r, (0, 255, 0), 2)

            if prev_x2 is not None and (x1 - prev_x2) > ((x2 - x1) * 1.5):
                row_text += " "

            char = decode_braille(candidates, braille_cell.shape[1], braille_cell.shape[0])
            row_text += char
            cv.putText(img_display, char.upper(), (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            prev_x2 = x2

        full_text += row_text + "\n"

    return full_text.strip(), img_display


