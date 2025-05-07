import numpy as np
import cv2 as cv
import uuid
import os 
from threading import Thread

# before you run the code, create two separate folders:
# images_output -> images are generated here
# video_inputs -> insert videos here
 
def video_to_images(filepath:str):
    cap = cv.VideoCapture(filepath)
    totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    i = 0
    while cap.isOpened() and i < totalFrames // 2:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read current frame. Exiting...")
            break 
        cv.imwrite(f"./images_output/{str(uuid.uuid4())}.png", frame)
        i += 1
    cap.release()
    cv.destroyAllWindows()
    

def generate_images(image_files):
    for file in image_files:
        p = f"./video_inputs/{file}"
        video_to_images(p)
    

image_files = os.listdir("./video_inputs")
length = len(image_files) - 1
x = image_files[:length // 2]
y = image_files[length // 2:]

th1 = Thread(target=generate_images, args=(x,))
th2 = Thread(target=generate_images, args=(y,))

th1.start()
th2.start()
