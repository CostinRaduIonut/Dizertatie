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
filenames = [filenames[8]]  # Test single image

for filename in filenames:
    fullpath = f"detection/detection_dataset/test/images/{filename}"
    result = model.predict(source=fullpath)
    boxes_xyxy = result[0].boxes.xyxy.tolist()
    img = cv.imread(fullpath, cv.IMREAD_GRAYSCALE)
    
    print(f"\nProcessing {fullpath}")

    for box in boxes_xyxy:
        x1, y1, x2, y2 = map(int, box)
        img_braille = img[y1:y2, x1:x2]
        img_area = img_braille.shape[0] * img_braille.shape[1]

        # Preprocess
        img_braille = cv.dilate(img_braille, kernel, iterations=1)  # Helps merge dots if fragmented
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

        # Draw debug visuals
        for circle in circle_candidates:
            cv.circle(cimg, circle[0], circle[1], (255, 0, 0), 1)  # Blue: all candidates
        for circle in filtered:
            cv.circle(cimg, circle[0], circle[1], (0, 255, 0), 1)  # Green: filtered
        
        print(f"Candidates: {len(circle_candidates)} | Filtered: {len(filtered)}")
        cv.imshow("Braille Dots", cimg)
        cv.waitKey(0)
        cv.destroyAllWindows()
