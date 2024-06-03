import argparse
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from siamese_network import Siamese
from utils import iou, load_known_faces, setup_logger


class AttendanceApp:
  data_dir = Path("data")
  weights_dir = data_dir / "weights"
  assets_dir = data_dir / "assets"
  records_dir = data_dir / "attendance"
  known_dir = data_dir / "known_faces"

  def __init__(self, root, args):
    self.logger = setup_logger(10 if args.verbose else 30)
    self.setup_backend(args)
    self.setup_gui(root)

  def setup_gui(self, root):
    self.root = root
    self.root.title("Attendance Recorder")
    self.root.geometry("900x640")
    self.root.minsize(width=900, height=640)

    self.date_time = ttk.Label(root, font=("", 14), background='#333', foreground='#ffa500', anchor='center')
    self.date_time.pack(pady=5, fill='x')
    self.update_date_time()

    self.main_frame = ttk.Frame(root)
    self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

    self.status_bar = ttk.Label(root, text=f"Registered Users: {self.nb_known_people}", relief=tk.SUNKEN, anchor='w')
    self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    self.setup_left_frame()
    self.setup_right_frame()

  def setup_left_frame(self):
    self.left_frame = ttk.Frame(self.main_frame)
    self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    self.record_button = ttk.Button(self.left_frame, text="Take Attendance", command=self.toggle_recording)
    self.record_button.pack(pady=5)
    self.is_recording = False

    self.register_button = ttk.Button(self.left_frame, text="Add New User", command=self.toggle_new_user)
    self.register_button.pack(pady=5)

    self.db_frame = ttk.Frame(self.left_frame)
    self.db_frame.pack(pady=5, fill="both", expand=True)

    self.db_tree = ttk.Treeview(self.db_frame, columns=list(self.attendance_df), show="headings")
    for i, col in enumerate(list(self.attendance_df)):
      self.db_tree.heading(col, text=col)
      self.db_tree.column(col, width=(i+1)*100)
    self.db_tree.pack(side="left", fill="both", expand=True)

    for row in self.attendance_df.to_numpy().tolist():
      self.db_tree.insert("", "end", values=row)

    self.db_scrollbar = ttk.Scrollbar(self.db_frame, orient=tk.VERTICAL, command=self.db_tree.yview)
    self.db_scrollbar.pack(side="right", fill="y")
    self.db_tree.configure(yscrollcommand=self.db_scrollbar.set)

    self.quit_button = ttk.Button(self.left_frame, text="Quit", command=self.root.quit)
    self.quit_button.pack(pady=5)

  def setup_right_frame(self):
    self.right_frame = ttk.Frame(self.main_frame)
    self.right_frame.grid(row=0, column=1, padx=10, pady=0, sticky="nsew")

    self.video_label = ttk.Label(self.right_frame)
    self.video_label.pack(expand=True, fill="both")

    self.text_label = ttk.Label(self.right_frame, anchor="center", font=("", 14))
    self.text_label.pack(side="left", expand=True, fill="x")

    self.take_img_button = ttk.Button(self.right_frame, text="Take Image", command=self.take_image)
    self.take_img_button["state"] = "disabled"

    self.display_frame(np.zeros((720, 720, 3)).astype(np.uint8))

  def update_date_time(self):
    self.date_time.config(text=f"{datetime.now():%d-%B-%Y  |  %H:%M:%S}")
    self.root.after(1000, self.update_date_time)

  def setup_backend(self, args):
    self.cap = None
    self.timer = None

    if not (self.assets_dir/"overlay.png").exists():
      self.logger.error("Overlay image not found.")
      raise FileNotFoundError("Overlay image not found.")

    self.overlay = cv2.imread(str(self.assets_dir/"overlay.png"), cv2.IMREAD_UNCHANGED)
    self.font = cv2.FONT_HERSHEY_SIMPLEX

    self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
    if self.face_cascade.empty():
      self.logger.error("Failed to load face cascade classifier.")
      raise FileNotFoundError("Face cascade classifier not found.")

    self.face_recognizer = Siamese()
    self.threshold = args.threshold
    if not args.weight:
      weights = [x for x in self.weights_dir.iterdir() if x.suffix == ".h5"]
      if len(weights) == 0:
        self.logger.error("No weight files found in the weights directory.")
        raise FileNotFoundError("No weight files found in the weights directory.")
      weight = weights[0]
    else:
      weight = self.weights_dir / args.weight

    if not weight.exists():
      self.logger.error("Weight file not found.")
      raise FileNotFoundError(f"Weight file not found: {weight}")

    self.face_recognizer.set_weight(weight)
    self.logger.info(f"Loaded weight: {weight}")
    self.min_buffer_size = args.min_buffer_size
    self.max_buffer_size = 2 * self.min_buffer_size

    self.load_database()
    self.known_people, self.labels = load_known_faces(self.known_dir)
    self.nb_known_people = len(self.known_people)
    self.logger.info(f"{self.nb_known_people} people: {self.labels}")

  def load_database(self):
    today = datetime.now().date().isoformat()
    self.db_file = self.records_dir / f"{today}.csv"
    if self.db_file.exists():
      self.attendance_df = pd.read_csv(self.db_file)
      self.logger.info("Loaded existing db for the day")
    else:
      self.attendance_df = pd.DataFrame(columns=['name', 'timestamp'])
      self.logger.info("Created a new db for the day")

  def toggle_recording(self):
    if self.nb_known_people == 0:
      messagebox.showerror("Error", "No registered users found. Please register at least one user before starting taking attendance.")
      return

    self.is_detecting = True
    if self.is_recording:
      self.stop_recording()
      self.record_button.config(text="Take Attendance")
      self.register_button["state"] = "normal"
    else:
      self.start_recording()
      self.record_button.config(text="Stop")
      self.register_button["state"] = "disabled"
    self.is_recording = not self.is_recording

  def toggle_new_user(self):
    self.is_detecting = False
    if self.is_recording:
      self.stop_recording()
      self.register_button.config(text="Add New User")
      self.record_button["state"] = "normal"
      self.take_img_button.pack_forget()
    else:
      self.start_recording()
      self.register_button.config(text="Stop")
      self.record_button["state"] = "disabled"
      self.take_img_button.pack(side="right")
    self.is_recording = not self.is_recording

  def start_recording(self):
    self.prediction_buffer = []
    self.image = None
    self.is_detected = False
    self.is_failed = False
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
    self.logger.info(f"fw: {self.frame_width}, fh: {self.frame_height}")

  def stop_recording(self):
    if self.timer is not None:
      self.root.after_cancel(self.timer)
    if self.cap is not None:
      self.cap.release()
    self.display_frame(np.zeros((720, 720, 3)).astype(np.uint8))
    self.update_text("")

  def take_image(self):
    self.toggle_new_user()
    image = self.image.copy()
    while True:
      name = simpledialog.askstring("Input", "Enter your name:")
      if name is None:
        self.logger.info(f"Name input was cancelled")
        return
      if name not in self.labels:
        break
      messagebox.showerror("Error", "This name already exists. Please enter a different name.")

    face_path = self.known_dir / f"{name}.jpg"
    cv2.imwrite(str(face_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    self.known_people.append(image)
    self.labels.append(name)
    self.nb_known_people += 1
    self.logger.info(f"New user saved, {str(face_path)}")
    self.status_bar.config(text=f"Registered Users: {self.nb_known_people}")
    messagebox.showinfo("Success", f"New face for {name} successfully recorded.")

  def get_input(self, image):
    return [self.known_people, [image]*len(self.known_people)]

  def update_timer(self):
    self.timer = self.root.after(30, self.update_frame)

  def display_frame(self, frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((500, 500))
    imgtk = ImageTk.PhotoImage(image=image)
    self.video_label.imgtk = imgtk
    self.video_label.config(image=imgtk)

  def detect_faces(self, frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

  def recognize_face(self, frame):
    inputs = self.get_input(frame)
    result = self.face_recognizer.predict(inputs, preprocess=True, batch_size=min(self.nb_known_people, 8))
    confidence, label = result.max(), result.argmax()
    return confidence, label

  def smooth_predictions(self):
    if not self.prediction_buffer:
      return 0, -1
    confs, lbls = zip(*self.prediction_buffer)
    avg_conf = np.mean(confs)
    lbl = max(set(lbls), key=lbls.count)
    return avg_conf, lbl

  def put_bbox(self, frame, face):
    x, y, w, h = face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

  def update_text(self, text):
    self.text_label.config(text=text)

  def add_overlay(self, frame):
    overlay = cv2.resize(self.overlay, (self.height, self.width))
    alpha = overlay[:, :, 3] / 255
    colors = overlay[:, :, :3]
    alpha_mask = np.dstack((alpha, alpha, alpha))
    frame[:] = frame[:]*(1 - alpha_mask) + colors*alpha_mask

  def post_update(self, frame):
    self.update_text(self.text)
    self.add_overlay(frame)
    self.display_frame(frame)
    self.update_timer()

  def post_detection(self, frame, failed=False):
    if self.frame_count == 0:
      self.logger.info(f"Subject{' failed to ' if failed else ' '}identified, stopping camera feed.")
    self.post_update(frame)
    self.frame_count += 1
    if self.frame_count > 10:
      self.toggle_recording()

  def update_frame(self):
    self.take_img_button["state"] = "disabled"
    ret, frame = self.cap.read()
    if not ret:
      self.update_timer()
      return

    offset = (self.frame_width-self.width)//2
    frame = frame[:, offset:self.width+offset, :]
    if self.is_detected:
      self.post_detection(frame, failed=self.is_failed)
      return

    self.process_frame(frame)

  def process_frame(self, frame):
    self.text = "Align your face"
    img = frame.copy()
    faces = self.detect_faces(img)
    if len(faces) != 1:
      self.post_update(frame)
      return

    face = faces[0]
    face_iou = iou(self.detection_box, face)
    self.text = "Align your face properly"
    self.logger.info(f"IoU: {face_iou:.4f}")
    if face_iou < 0.65:
      self.post_update(frame)
      return

    if not self.is_detecting:
      self.take_img_button["state"] = "normal"
      self.put_bbox(frame, face)
      self.text = "Press Take Image button"
      self.post_update(frame)
      self.image = img
      return

    self.recognize_and_record(frame, face)

  def recognize_and_record(self, frame, face):
    img = frame.copy()
    self.text = "Recognizing face..."
    self.put_bbox(frame, face)
    confidence, label = self.recognize_face(img)
    self.prediction_buffer.append((confidence, label))
    self.logger.info(f"Buffer size: {len(self.prediction_buffer)}")
    if len(self.prediction_buffer) < self.min_buffer_size:
      self.post_update(frame)
      return

    if len(self.prediction_buffer) == self.max_buffer_size:
      self.is_detected = True
      self.is_failed = True
      self.text = f"Recognition failed, please try again"
      self.post_update(frame)
      return

    avg_confidence, most_common_label = self.smooth_predictions()
    name = self.labels[most_common_label]
    if avg_confidence > self.threshold:
      self.is_detected = True
      self.record_attendance(name)
      self.text = f"Welcome {name}. Your attendance has been recorded"

    self.logger.info(f"Label: {name}, Confidence: {avg_confidence:.4f}")
    self.logger.debug(f"{self.prediction_buffer}")
    self.logger.debug(f"{self.labels}")
    self.post_update(frame)

  def record_attendance(self, name):
    new_record = {'name': name, 'timestamp': pd.Timestamp.now()}
    self.attendance_df.loc[len(self.attendance_df)] = new_record

    self.db_tree.insert("", "end", values=list(new_record.values()))
    self.attendance_df.to_csv(self.db_file, index=False)
    self.logger.info("New record saved")


def main():
  parser = argparse.ArgumentParser(description="Attendance Recorder using Face Recognition.")
  parser.add_argument("-w", "--weight", type=str, default="", help="Filename of the weight file to be used for the Siamese network. This file must be placed inside the 'data/weights' directory. If not specified, the first '.h5' file found in the directory will be used.")
  parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Threshold for face recognition confidence. If the recognition confidence exceeds this threshold, the face is considered recognized. Default value is 0.5.")
  parser.add_argument("-b", "--min-buffer-size", type=int, default=5, help="Minimum buffer size for face recognition. Default value is 10.")
  parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose command line output.")
  args = parser.parse_args()

  root = tk.Tk()
  app = AttendanceApp(root, args)
  root.mainloop()


if __name__ == "__main__":
  main()
