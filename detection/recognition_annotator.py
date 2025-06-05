
import os
import cv2
import string
from tkinter import Tk, Button, Label, Frame, Canvas, filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
from uuid import uuid4

MODEL_PATH = "runs/detect/train6_good/weights/best.pt"
SOURCE_DIR = "braille_detectat"
DEST_DIR = "detection/recognition_dataset_unorganized"

class AnnotatorGUI:
    def __init__(self, root):
        self.root = root
        self.model = YOLO(MODEL_PATH)
        self.image_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.png'))]
        self.current_file_index = 0
        self.boxes = []
        self.box_index = 0
        self.history = []

        self.canvas = Canvas(root, width=300, height=300)

        # Load Braille reference image
        ref_img_path = os.path.join(os.path.dirname(__file__), "braille_reference.png")
        if os.path.exists(ref_img_path):
            pil_ref = Image.open(ref_img_path).resize((300, 150))
            self.tk_ref_img = ImageTk.PhotoImage(pil_ref)
            self.ref_canvas = Canvas(root, width=300, height=150)
            self.ref_canvas.pack()
            self.ref_canvas.create_image(0, 0, anchor="nw", image=self.tk_ref_img)
            Label(root, text="Braille Reference (a–z)").pack()
        self.canvas.pack()

        self.label = Label(root, text="Press a button or key to annotate")
        self.label.pack()

        button_frame = Frame(root)
        button_frame.pack()
        self.buttons = {}
        for i, char in enumerate(string.ascii_lowercase):
            btn = Button(button_frame, text=char, width=2, command=lambda c=char: self.save_label(c))
            btn.grid(row=i//13, column=i%13)
            self.buttons[char] = btn

        control_frame = Frame(root)
        control_frame.pack()

        Button(control_frame, text="Skip", command=self.skip).pack(side="left")
        Button(control_frame, text="Undo", command=self.undo).pack(side="left")
        Button(control_frame, text="Exit", command=root.quit).pack(side="left")

        self.root.bind("<Key>", self.key_press)
        self.load_next_image()

    def load_next_image(self):
        while self.current_file_index < len(self.image_files):
            filename = self.image_files[self.current_file_index]
            filepath = os.path.join(SOURCE_DIR, filename)
            result = self.model.predict(source=filepath)
            self.boxes = result[0].boxes.xyxy.tolist()
            self.box_index = 0
            self.current_image = cv2.imread(filepath)
            if self.boxes:
                self.show_box()
                return
            else:
                self.current_file_index += 1
        self.label.config(text="No more images.")

    def show_box(self):
        if self.box_index >= len(self.boxes):
            self.current_file_index += 1
            self.load_next_image()
            return
        x1, y1, x2, y2 = map(int, self.boxes[self.box_index])
        self.current_crop = self.current_image[y1:y2, x1:x2]
        cv2.imwrite("temp.jpg", self.current_crop)
        pil_image = Image.open("temp.jpg").resize((300, 300))
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.label.config(text=f"Image {self.current_file_index+1}/{len(self.image_files)}, Cell {self.box_index+1}/{len(self.boxes)}")

    def save_label(self, label):
        filename = f"{label}.{str(uuid4()).split('-')[0]}.jpg"
        save_path = os.path.join(DEST_DIR, filename)
        cv2.imwrite(save_path, self.current_crop)
        self.history.append((save_path, self.box_index, self.current_file_index))
        self.box_index += 1
        self.show_box()

    def skip(self):
        self.box_index += 1
        self.show_box()

    def undo(self):
        if not self.history:
            return
        last_path, last_box_index, last_file_index = self.history.pop()
        if os.path.exists(last_path):
            os.remove(last_path)
        self.box_index = last_box_index
        self.current_file_index = last_file_index
        self.show_box()

    def key_press(self, event):
        key = event.char.lower()
        if key in string.ascii_lowercase:
            self.save_label(key)
        elif event.keysym == "space":
            self.skip()
        elif event.keysym == "Escape":
            self.root.quit()

if __name__ == "__main__":
    os.makedirs(DEST_DIR, exist_ok=True)
    root = Tk()
    root.title("Braille Annotator")
    app = AnnotatorGUI(root)
    root.mainloop()
