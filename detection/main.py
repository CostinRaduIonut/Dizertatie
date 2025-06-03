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
filenames = ["00c68b61-95cd-4a93-862f-a12b89f8be36_png.rf.1ec9431b21b23834d94a7bf1370b997f.jpg"]  # Test single image

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
        cv.imshow("a", img_braille)
        cv.waitKey(0)
        cv.destroyAllWindows()