import argparse
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from siamese_network import Siamese


class AttendanceApp:
  data_dir = Path("data")
  weights_dir = data_dir / "weights"
  assets_dir = data_dir / "assets"
  records_dir = data_dir / "attendance"
  known_dir = data_dir / "known_faces"

  def __init__(self, root, args):
    self.root = root
    self.root.title("Attendance Recorder")
    self.root.geometry("700x600")

    self.left_frame = ttk.Frame(root, height=720)
    self.left_frame.grid(row=0, column=0, padx=0, pady=0)

    self.video_label = ttk.Label(root)
    self.video_label.grid(row=0, column=1, padx=0, pady=0, sticky='')
    self.display_frame(np.zeros((720, 720, 3)).astype(np.uint8))

    self.start_button = ttk.Button(self.left_frame, text="Start Recording", command=self.start_recording)
    self.start_button.grid(row=0, column=0, padx=0, pady=0)

    self.stop_button = ttk.Button(self.left_frame, text="Stop Recording", command=self.stop_recording)
    self.stop_button.grid(row=1, column=0, padx=0, pady=0)

    self.cap = None
    self.timer = None
    self.overlay = cv2.imread(str(self.assets_dir/"overlay.png"), cv2.IMREAD_UNCHANGED)
    self.font = cv2.FONT_HERSHEY_SIMPLEX

    # Database setup
    self.load_database()

    # Initialize face recognizer
    self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
    self.face_recognizer = Siamese()
    self.threshold = args.threshold
    if not args.weight:
      weights = [x for x in self.weights_dir.iterdir() if x.suffix == ".h5"]
      assert len(weights) > 0
      args.weight = weights[0]
    self.face_recognizer.set_weight(args.weight)

    # Load known faces
    self.load_known_faces()

  def load_database(self):
    today = datetime.now().date().isoformat()
    self.db_file = self.records_dir / f"{today}.csv"
    self.attendance_df = pd.read_csv(self.db_file) if self.db_file.exists() else pd.DataFrame(columns=['name', 'timestamp'])

  def load_known_faces(self):
    self.labels = []
    self.known_people = []
    for person_file in self.known_dir.iterdir():
      if not person_file.is_file():
        continue
      if person_file.suffix.casefold() != ".jpg":
        continue
      self.labels.append(".".join(person_file.name.split('.')[:-1]))
      self.known_people.append(cv2.imread(str(person_file)))
    self.nb_known_people = len(self.known_people)

  def start_recording(self):
    self.detected = False
    self.text = ""
    self.frame_count = 0
    self.cap = cv2.VideoCapture(0)
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self.width = self.height = self.frame_height
    self.detection_box = (self.width // np.array([3.6, 14.4, 2.33, 1.8])).astype(int) #xywh
    self.update_timer()

  def stop_recording(self):
    if self.timer is not None:
      self.root.after_cancel(self.timer)
    if self.cap is not None:
      self.cap.release()
    self.display_frame(np.zeros((720, 720, 3)).astype(np.uint8))

  def get_input(self, image):
    return [self.known_people, [image]*len(self.known_people)]

  def update_timer(self):
    self.timer = self.root.after(30, self.update_frame)

  def display_frame(self, frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((600, 600))
    imgtk = ImageTk.PhotoImage(image=image)
    self.video_label.imgtk = imgtk
    self.video_label.config(image=imgtk)

  def detect_faces(self, frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

  def recognize_face(self, frame):
    return self.face_recognizer.predict(
      self.get_input(frame),
      preprocess=True,
      batch_size=min(self.nb_known_people, 8))

  def put_bbox(self, frame, face):
    x, y, w, h = face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

  def update_text(self, frame, text):
    textsize = cv2.getTextSize(text, self.font, 0.7, 2)[0]
    cv2.putText(frame, text, (self.width//2 - (textsize[0]//2), self.height - self.height//60), self.font, 0.7, (36, 255, 12), 2)

  def add_overlay(self, frame):
    overlay = cv2.resize(self.overlay, (self.height, self.width))
    alpha = overlay[:, :, 3] / 255
    colors = overlay[:, :, :3]
    alpha_mask = np.dstack((alpha, alpha, alpha))
    frame[:] = frame[:]*(1 - alpha_mask) + colors*alpha_mask

  def post_update(self, frame):
    self.update_text(frame, self.text)
    self.add_overlay(frame)
    self.display_frame(frame)
    self.update_timer()

  def update_frame(self):
    ret, frame = self.cap.read()
    if not ret:
      self.update_timer()
      return

    offset = (self.frame_width-self.width)//2
    frame = frame[:, offset:self.width+offset, :]
    if self.detected:
      self.post_update(frame)
      self.frame_count += 1
      if self.frame_count > 20:
        self.stop_recording()
      return

    self.text = "Align your face"
    img = frame.copy()
    faces = self.detect_faces(img)
    if len(faces) != 1:
      self.post_update(frame)
      return

    face = faces[0]
    face_iou = iou(xywh2xyxy(self.detection_box), xywh2xyxy(face))
    self.text = "Align your face properly"
    if face_iou < 0.7:
      self.post_update(frame)
      return

    self.put_bbox(frame, face)
    self.text = "Who are you?"
    result = self.recognize_face(img)
    confidence, label = result.max(), result.argmax()
    if confidence > self.threshold:
      name = self.labels[label]
      self.detected = True
      self.record_attendance(name)
      self.text = f"Welcome {name}. Your attendance has been recorded"
    self.post_update(frame)

  def record_attendance(self, name):
    new_record = {'name': name, 'timestamp': pd.Timestamp.now()}
    self.attendance_df.loc[len(self.attendance_df)] = new_record
    self.attendance_df.to_csv(self.db_file, index=False)


def xywh2xyxy(box):
  b = box.copy()
  if len(b.shape) == 1:
    b[2] += b[0]
    b[3] += b[1]
  elif len(b.shape) == 2:
    b[:, 2] += b[:, 0]
    b[:, 3] += b[:, 1]
  return b


def iou(boxA, boxB):
  boxes = np.vstack([boxA, boxB])
  xA, yA = boxes[:, :2].max(axis=0)
  xB, yB = boxes[:, 2:].min(axis=0)
  inter_area = max(0, xB-xA+1) * max(0, yB-yA+1)
  boxes_area = (boxes[:, 2]-boxes[:, 0]+1) * (boxes[:, 3]-boxes[:, 1]+1)
  return inter_area / (boxes_area.sum() - inter_area)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Attendance Recorder using Face Recognition.")
  parser.add_argument("-w", "--weight", type=str, default="", help="Filename of the weight file to be used for the Siamese network. This file must be placed inside the 'data/weights' directory. If not specified, the first '.h5' file found in the directory will be used.")
  parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Threshold for face recognition confidence. If the recognition confidence exceeds this threshold, the face is considered recognized. Default value is 0.5.")
  args = parser.parse_args()

  root = tk.Tk()
  app = AttendanceApp(root, args)
  root.mainloop()
