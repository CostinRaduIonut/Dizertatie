from ultralytics import YOLO
import cv2 as cv
import numpy as np
import os
import scipy

model = YOLO("runs/detect/train6_good/weights/best.pt")

tolerance = 0.4
kernel = np.ones((3, 3), np.uint8)  # Slightly smaller kernel for dilation
braille_dict = {
    "100000" : "a",
    "110000" : "b",
    "100100" : "c",
    "100110" : "d",
    "100010" : "e",
    "110100" : "f",
    "110110" : "g",
    "110010" : "h",
    "010100" : "i",
    "010110" : "j",
    "101000" : "k",
    "111000" : "l",
    "101100" : "m",
    "101110" : "n",
    "101010" : "o",
    "111100" : "p",
    "111110" : "q",
    "111010" : "r",
    "011100" : "s",
    "011110" : "t",
    "101001" : "u",
    "111001" : "v",
    "010111" : "w",
    "101101" : "x",
    "101111" : "y",
    "101011" : "z"
    
}

def sort_boxes(boxes, y_threshold=15):
    """
    Sorts boxes top-to-bottom, then left-to-right within each row.
    boxes: list of [x1, y1, x2, y2]
    Returns: sorted list of boxes or empty list if no boxes
    """
    if not boxes:
        return []

    boxes = np.array(boxes)

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        print("[Warning] Detected boxes have invalid shape:", boxes.shape)
        return []

    # Sort by y1 (top coordinate)
    boxes = boxes[boxes[:, 1].argsort()]

    rows = []
    current_row = [boxes[0]]
    for box in boxes[1:]:
        if abs(box[1] - current_row[-1][1]) < y_threshold:
            current_row.append(box)
        else:
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [box]
    rows.append(sorted(current_row, key=lambda b: b[0]))  # last row

    return [box.tolist() for row in rows for box in row]

# Load images
filenames = os.listdir("braille_detectat/")
# filenames = ["0a5e1b87-1554-4b50-8e67-4bae748b2ab8_png.rf.7ba2be5e0bed500ddb267b4ede3ef6d1.jpg"]
# filenames = ["00c68b61-95cd-4a93-862f-a12b89f8be36_png.rf.1ec9431b21b23834d94a7bf1370b997f.jpg"]  # Test single image
# filenames = ["0b999ef4-8c76-4313-9b65-4197c3c318de_png.rf.76f81a72c4d7871a31c5b89ca95d1a27.jpg"]
# filenames = ["0d7bef26-e53e-4ce5-bc03-ab9f759566df_png.rf.2e92df16554cc8d90266100bbd05b364.jpg"]
# filenames = ["0aeaec4c-975f-4f81-83ce-9fbd6b546258_png.rf.2099a10504bef021af0c6d128391ab3e.jpg"]

for filename in filenames:
    fullpath = f"braille_detectat/{filename}"
    result = model.predict(source=fullpath)
    boxes_xyxy = result[0].boxes.xyxy.tolist()
    if not boxes_xyxy:
        print(f"[Info] No Braille boxes detected in {filename}. Skipping.")
        continue
    boxes_xyxy = sort_boxes(boxes_xyxy)
    img = cv.imread(fullpath, cv.IMREAD_GRAYSCALE)
    im_height, im_width = img.shape
    img_display = cv.cvtColor(img.copy(), cv.COLOR_GRAY2BGR)
    print(f"\nProcessing {fullpath}")
    text = ""

    for box in boxes_xyxy:
        x1, y1, x2, y2 = map(int, box)
        img_braille = img[y1:y2, x1:x2]
        img_area = img_braille.shape[0] * img_braille.shape[1]
        cv.rectangle(img_display, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

        # Preprocess
        img_braille = cv.dilate(img_braille, kernel, iterations=1)  # Helps merge dots if fragmented
        b_height, b_width = img_braille.shape
        print(f"dim = {b_width}, {b_height}")
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

        circle_candidates = []
        for circle in cnt_data:
            radius = circle[1]
            rect_area = circle[3]
            center = circle[0]
            percent = rect_area / img_area
            if radius > 2 and percent < 0.95 and center[0] > 4 and center[1] > 4:
                circle_candidates.append(circle)

        for c in circle_candidates:
            px = int(c[0][0] + box[0])
            py = int(c[0][1] + box[1])
            cv.circle(img_display, (px, py), c[1], color=(0, 255, 0), thickness=2)
        
        points = list(map(lambda c: (c[0][0], c[0][1]), circle_candidates))
        points_normalized = list(map(lambda p: (p[0] / b_width, p[1] / b_height), points))
        
        temp = []
        
        for p in points_normalized:
            x = p[0]
            y = p[1]
            col = 0 if x < .5 else 1 
            
            row = 0 if y < .5 else 1 
            if y > 0 and y < 0.4:
                row = 0
            elif y >= 0.4 and y < 0.75:
                row = 1
            elif y >= 0.75:
                row = 2
                
            
            temp.append((col, row))
        braille_encoding = ""
        coordinates = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        temp.reverse()
        for x in coordinates:
            col = x[0]
            row = x[1]
            braille_encoding += "1" if (col, row) in temp else "0"
        if braille_encoding in braille_dict.keys():
            text += braille_dict[braille_encoding]
            cv.putText(img_display, braille_dict[braille_encoding].capitalize(), (int(box[0]), int(box[1] - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        else:
            cv.putText(img_display, "*", (int(box[0]), int(box[1] - 10)), cv.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 0), 2)
            
    
            
    print(text)
    cv.imshow("a", img_display)
    cv.waitKey(0)
    cv.destroyAllWindows()