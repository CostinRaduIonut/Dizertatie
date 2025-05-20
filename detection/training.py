from ultralytics import YOLO
import cv2 as cv 
import numpy as np 
import torch
from torchvision.ops import nms

def train():
    model = YOLO("detection/yolo11s.pt")  # load a pretrained model (recommended for training)

    results = model.train(data="detection/detection_dataset/data.yaml", epochs=80, imgsz=640, dfl=1.5, cls=0.5, box=7.0, device=0, batch=16)

    success = model.export(format="onnx")
    
if __name__ == '__main__':
    train()
