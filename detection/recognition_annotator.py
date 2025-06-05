from ultralytics import YOLO
import cv2 as cv
import numpy as np
import os
import string 
from uuid import uuid4

DEST_DIR = "detection/recognition_dataset_unorganized"
SOURCE_DIR = "braille_detectat/"
MODEL_PATH = "runs/detect/train6_good/weights/best.pt"
letters = string.ascii_letters.lower()

def start_annotation():
    model = YOLO(MODEL_PATH)
    filenames = os.listdir(SOURCE_DIR)
    for filename in filenames:
        filepath = f"{SOURCE_DIR}{filename}"
        result = model.predict(source=filepath)
        boxes = result[0].boxes.xyxy.tolist()
        img = cv.imread(filepath)
        img_display = img.copy()
        
        for i,box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            braille_cell = img[y1:y2, x1:x2]
            cv.imshow(f"{filename} -> {i}", braille_cell)
            key = cv.waitKey(0) & 0xff
            cv.destroyAllWindows()        
            if key == ord(' '):
                continue
            elif key == 27:
                return  
            elif key == 13:
                break 
            else:
                cell_filename = f"{chr(key)}.{str(uuid4()).split('-')[0]}.jpg"
                cv.imwrite(f"{DEST_DIR}/{cell_filename}", braille_cell)
            
            
        
        
start_annotation()
        
    
    