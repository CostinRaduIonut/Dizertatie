from ultralytics import YOLO
import cv2 as cv 
import numpy as np 
import torch
from torchvision.ops import nms
import os 

# model = YOLO("detection/yolo11n.pt")  # load a pretrained model (recommended for training)

# results = model.train(data="detection/detection_dataset/data.yaml", epochs=30, imgsz=640)

# success = model.export(format="onnx")


model = YOLO("runs/detect/train6/weights/best.pt")  # load a pretrained model (recommended for training)


import cv2
import numpy as np

def is_close(box1, box2, threshold=5):
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2
    return abs(x1a - x1b) < threshold and abs(y1a - y1b) < threshold and \
           abs(x2a - x2b) < threshold and abs(y2a - y2b) < threshold

def recover_missing_boxes_with_nms(image_path, yolo_boxes):
    image = cv2.imread(image_path)

    # Convert YOLO boxes to Python tuples
    yolo_boxes_np = [tuple(map(int, box.cpu().numpy())) for box in yolo_boxes]

    # Calculate average area of YOLO-detected boxes
    areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in yolo_boxes_np]
    avg_area = np.mean(areas)
    area_tolerance = 0.3  # 30% area tolerance

    # Extract all contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    recovered_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        contour_area = w * h

        if avg_area * (1 - area_tolerance) < contour_area < avg_area * (1 + area_tolerance):
            already_detected = any(is_close((x, y, x + w, y + h), existing_box) for existing_box in yolo_boxes_np)
            if not already_detected:
                recovered_boxes.append((x, y, x + w, y + h))

    # Combine YOLO and recovered boxes
    all_boxes = yolo_boxes_np + recovered_boxes

    # Prepare boxes and scores for NMS
    all_boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
    scores = torch.cat([
        torch.ones(len(yolo_boxes_np)),                      # High confidence for YOLO boxes
        torch.full((len(recovered_boxes),), 0.5)             # Lower confidence for recovered boxes
    ])

    # Apply NMS to remove redundant boxes
    iou_threshold = 0.4
    nms_indices = nms(all_boxes_tensor, scores, iou_threshold)
    nms_boxes = all_boxes_tensor[nms_indices].int().tolist()

    # Draw NMS filtered boxes on the image
    for x1, y1, x2, y2 in nms_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("NMS Filtered Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return nms_boxes



# Path to the test image
image_path = "braille_detectat/"

# Run recovery
# final_boxes = recover_missing_boxes_with_nms(image_path, valid_boxes)

r = model.predict(source='braille_detectat/', save=True)

listdr = os.listdir("runs/detect/predict")

for i, result in enumerate(r):
    recover_missing_boxes_with_nms("braille_detectat/" + listdr[i], result.cpu)

