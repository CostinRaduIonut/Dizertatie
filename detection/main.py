from ultralytics import YOLO
import cv2 as cv
import numpy as np
import os
import scipy

model = YOLO("runs/detect/train6_good/weights/best.pt")

tolerance = 0.4
kernel = np.ones((3, 3), np.uint8)  # Slightly smaller kernel for dilation

# Load images
filenames = os.listdir("detection/detection_dataset/test/images/")
filenames = ["0e74f47f-ccec-4ee5-af4a-9d8334be3573_png.rf.3638872fa1aef478828dddaa0785b877.jpg"]  # Test single image

for filename in filenames:
    fullpath = f"detection/detection_dataset/test/images/{filename}"
    result = model.predict(source=fullpath)
    boxes_xyxy = result[0].boxes.xyxy.tolist()
    img = cv.imread(fullpath, cv.IMREAD_GRAYSCALE)
    im_height, im_width = img.shape
    
    print(f"\nProcessing {fullpath}")

    for box in boxes_xyxy:
        x1, y1, x2, y2 = map(int, box)
        img_braille = img[y1:y2, x1:x2]
        img_area = img_braille.shape[0] * img_braille.shape[1]

        # Preprocess
        img_braille = cv.dilate(img_braille, kernel, iterations=1)  # Helps merge dots if fragmented
        b_height, b_width = img_braille.shape
        cimg = cv.cvtColor(img_braille, cv.COLOR_GRAY2BGR)
        _, thresh = cv.threshold(img_braille, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        cnt_data = []

        for cnt in contours:
            (x, y), radius = cv.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            area = np.pi * (radius ** 2)  # Fixed area formula
            rect_x, rect_y, w, h = cv.boundingRect(cnt)
            rect_area = w * h
            aspect_ratio = w / h if h != 0 else 0
            extent = cv.contourArea(cnt) / rect_area if rect_area != 0 else 0
            hull = cv.convexHull(cnt)
            hull_area = cv.contourArea(hull)
            solidity = cv.contourArea(cnt) / hull_area if hull_area != 0 else 0
            cnt_data.append((center, radius, area, rect_area, aspect_ratio, extent, solidity))

        # Area filtering
        areas = np.array([circle[2] for circle in cnt_data])
        avg_area = np.mean(areas)
        print(f"Average detected area: {avg_area:.2f}")

        # First filter: reasonable size
        circle_candidates = []
        for circle in cnt_data:
            radius = circle[1]
            rect_area = circle[3]
            if radius > 3 and (rect_area / img_area) < 0.95:
                circle_candidates.append(circle)

        # Refined filter: aspect ratio, extent, solidity, area tolerance
        filtered = []
        for circle in circle_candidates:
            area = circle[2]
            aspect_ratio = circle[4]
            extent = circle[5]
            solidity = circle[6]
            if ((1 - tolerance) * avg_area <= area <= (1 + tolerance) * avg_area and
                0.6 <= aspect_ratio <= 1.4 and
                extent >= 0.6 and
                solidity >= 0.8):
                filtered.append(circle)
        points = list(map(lambda c: (c[0][0], c[0][1]), circle_candidates))
        points_normalized = list(map(lambda p: (p[0] / b_width, p[1] / b_height), points))
        
        temp = []
        
        for p in points_normalized:
            x = p[0]
            y = p[1]
            col = 0 if x < .5 else 1 
            if y < 0.33:
                row = 0
            elif y > 0.66:
                row = 2
            else:
                row = 1
            temp.append((col, row))
        print(temp)
