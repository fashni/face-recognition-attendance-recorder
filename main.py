import argparse
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from siamese_network_onnx import Siamese
from utils import get_weight_dir, iou, load_known_faces, setup_logger


class FaceRecognizer:
  """
  A class for recognizing faces in images using a pre-trained Siamese network.
  """
  def __init__(self, weight_path, threshold, known_faces, labels, logger):
    """
    Initializes the FaceRecognizer with a given weight path, threshold, and logger.

    Args:
        weight_path (str): Path to the pre-trained weights for the Siamese network.
        threshold (float): Threshold for face recognition confidence.
        known_faces (list[numpy.ndarray]): List of known faces to compare against.
        logger (Logger): Logger object for logging information.
    """
    self.logger = logger
    self.threshold = threshold
    self.nb_known_faces = 0
    self.face_recognizer = Siamese(weight_path)
    self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
    self.labels = labels
    self.known_embeddings = self.get_known_embedding(known_faces)

  def get_known_embedding(self, imgs, preprocess=False):
    self.nb_known_faces = len(imgs)
    return self.face_recognizer.get_embedding(imgs, batch_size=min(self.nb_known_faces, 32), preprocess=preprocess)

  def add_embedding(self, image, label, preprocess=True):
    embedding = self.face_recognizer.get_embedding(image, preprocess=preprocess)
    self.known_embeddings = np.vstack([self.known_embeddings, embedding])
    self.labels.append(label)
    self.nb_known_faces += 1

  def recognize_face(self, frame):
    """
    Recognizes faces in a given frame and compares them with known faces.

    Args:
        frame (numpy.ndarray): The frame in which to recognize faces.

    Returns:
        tuple: A tuple containing the maximum confidence score and the label of the recognized face.
    """
    result = self.face_recognizer.predict(frame, self.known_embeddings, preprocess=True)
    confidence, label = result.max(), result.argmax()
    return confidence, label


class DatabaseManager:
  """
  A class for managing a database of attendance records.
  """
  def __init__(self, records_dir, logger):
    """
    Initializes the DatabaseManager with directories for records and known faces, and a logger.

    Args:
        records_dir (Path): Directory where attendance records are stored.
        known_dir (Path): Directory where known faces are stored.
        logger (Logger): Logger object for logging information.
    """
    self.logger = logger
    self.records_dir = records_dir
    self.load_database()

  def load_database(self):
    """
    Loads the attendance database for the current day. If no database exists, creates a new one.
    """
    today = datetime.now().date().isoformat()
    self.db_file = self.records_dir / f"{today}.csv"
    if self.db_file.exists():
      self.attendance_df = pd.read_csv(self.db_file)
      self.logger.info("Loaded existing db for the day")
    else:
      self.attendance_df = pd.DataFrame(columns=['name', 'timestamp'])
      self.logger.info("Created a new db for the day")

  def record_attendance(self, record):
    """
    Records attendance by adding a new record to the database and saving it.

    Args:
        record (dict): Dictionary containing the attendance record with keys 'name' and 'timestamp'.
    """
    self.attendance_df.loc[len(self.attendance_df)] = record
    self.attendance_df.to_csv(self.db_file, index=False)
    self.logger.info("New record saved")


class UIManager:
  ASSETS_DIR = Path("data/assets")
  KNOWN_DIR = Path("data/known_faces")
  def __init__(self, root, face_recognizer, db_manager, logger, min_buffer_size=5):
    """
    Initializes the UIManager with the root window, face recognizer, database manager, logger, and buffer sizes.

    Args:
        root (tk.Tk): The root window for the Tkinter application.
        face_recognizer (FaceRecognizer): An instance of the FaceRecognizer class.
        db_manager (DatabaseManager): An instance of the DatabaseManager class.
        logger (Logger): Logger object for logging information.
        min_buffer_size (int): Minimum buffer size for smoothing predictions. Default is 5.
    """
    self.root = root
    self.face_recognizer = face_recognizer
    self.db_manager = db_manager
    self.logger = logger
    self.min_buffer_size = min_buffer_size
    self.max_buffer_size = 2 * min_buffer_size
    self.setup_gui()

  def setup_gui(self):
    """
    Sets up the graphical user interface for the application.
    """
    self.root.title("Attendance Recorder")
    self.root.geometry("900x640")
    self.root.minsize(width=900, height=640)

    self.date_time = ttk.Label(self.root, font=("", 14), background='#333', foreground='#ffa500', anchor='center')
    self.date_time.pack(pady=5, fill='x')
    self.update_date_time()

    self.main_frame = ttk.Frame(self.root)
    self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

    self.status_bar = ttk.Label(self.root, text=f"Registered Users: {self.face_recognizer.nb_known_faces}", relief=tk.SUNKEN, anchor='w')
    self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    self.setup_left_frame()
    self.setup_right_frame()
    self.setup_menu()

  def setup_left_frame(self):
    """
    Sets up the left frame of the GUI, containing buttons and a table for database records.
    """
    self.left_frame = ttk.Frame(self.main_frame)
    self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    self.record_button = ttk.Button(self.left_frame, text="Take Attendance", command=self.toggle_recording)
    self.record_button.pack(pady=5)
    self.is_recording = False

    self.register_button = ttk.Button(self.left_frame, text="Add New User", command=self.toggle_new_user)
    self.register_button.pack(pady=5)

    self.db_frame = ttk.Frame(self.left_frame)
    self.db_frame.pack(pady=5, fill="both", expand=True)

    self.db_tree = ttk.Treeview(self.db_frame, columns=list(self.db_manager.attendance_df), show="headings")
    for i, col in enumerate(list(self.db_manager.attendance_df)):
      self.db_tree.heading(col, text=col)
      self.db_tree.column(col, width=(i + 1) * 100)
    self.db_tree.pack(side="left", fill="both", expand=True)

    for row in self.db_manager.attendance_df.to_numpy().tolist():
      self.db_tree.insert("", "end", values=row)

    self.db_scrollbar = ttk.Scrollbar(self.db_frame, orient=tk.VERTICAL, command=self.db_tree.yview)
    self.db_scrollbar.pack(side="right", fill="y")
    self.db_tree.configure(yscrollcommand=self.db_scrollbar.set)

    self.quit_button = ttk.Button(self.left_frame, text="Quit", command=self.root.quit)
    self.quit_button.pack(pady=5)

  def setup_right_frame(self):
    """
    Sets up the right frame of the GUI, containing video display.
    """
    self.right_frame = ttk.Frame(self.main_frame)
    self.right_frame.grid(row=0, column=1, padx=10, pady=0, sticky="nsew")

    self.video_label = ttk.Label(self.right_frame)
    self.video_label.pack(expand=True, fill="both")
    self.overlay = cv2.imread(str(self.ASSETS_DIR/"overlay.png"), cv2.IMREAD_UNCHANGED)

    self.text_label = ttk.Label(self.right_frame, anchor="center", font=("", 14))
    self.text_label.pack(side="left", expand=True, fill="x")

    self.take_img_button = ttk.Button(self.right_frame, text="Take Image", command=self.take_image)
    self.take_img_button["state"] = "disabled"

    self.display_frame(np.zeros((720, 720, 3)).astype(np.uint8))

  def setup_menu(self):
    """
    Sets up the menu for the GUI, including settings options.
    """
    menubar = tk.Menu(self.root)
    self.root.config(menu=menubar)

    settings_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Settings", menu=settings_menu)

    settings_menu.add_command(label="Change Threshold", command=self.change_threshold)
    settings_menu.add_command(label="Change Min Buffer Size", command=self.change_min_buffer_size)
    settings_menu.add_command(label="Change Max Buffer Size", command=self.change_max_buffer_size)

  def change_threshold(self):
    """
    Changes the threshold for face recognition confidence.
    """
    new_threshold = simpledialog.askfloat("Change Threshold", "Enter new threshold value:",
                                          minvalue=0.0, maxvalue=1.0, initialvalue=self.face_recognizer.threshold)
    if new_threshold is not None:
      self.face_recognizer.threshold = new_threshold
      messagebox.showinfo("Threshold Changed", f"New threshold set to: {new_threshold}")

  def change_min_buffer_size(self):
    """
    Changes the minimum buffer size for smoothing predictions.
    """
    new_min_buffer_size = simpledialog.askinteger("Change Min Buffer Size", "Enter new minimum buffer size:",
                                                  minvalue=1, maxvalue=self.max_buffer_size, initialvalue=self.min_buffer_size)
    if new_min_buffer_size is not None:
      self.min_buffer_size = new_min_buffer_size
      messagebox.showinfo("Buffer Size Changed", f"New min. buffer size set to: {new_min_buffer_size}")

  def change_max_buffer_size(self):
    """
    Changes the maximum buffer size for smoothing predictions.
    """
    new_max_buffer_size = simpledialog.askinteger("Change Max Buffer Size", "Enter new maximum buffer size:",
                                                  minvalue=self.min_buffer_size, maxvalue=100, initialvalue=self.max_buffer_size)
    if new_max_buffer_size is not None:
      self.max_buffer_size = new_max_buffer_size
      messagebox.showinfo("Buffer Size Changed", f"New max. buffer size set to: {new_max_buffer_size}")

  def update_date_time(self):
    """
    Updates the date and time display on the GUI.
    """
    self.date_time.config(text=f"{datetime.now():%d %B %Y  |  %H:%M:%S}")
    self.root.after(1000, self.update_date_time)

  def toggle_recording(self):
    """
    Toggles the recording state for taking attendance.
    """
    if self.face_recognizer.nb_known_faces == 0:
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
    """
    Toggles the state for adding a new user.
    """
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
    """
    Starts the video recording for face recognition.
    """
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
    self.detection_box = (self.width // np.array([3.6, 14.4, 2.33, 1.8])).astype(int)  # xywh
    self.update_timer()
    self.logger.debug(f"fw: {self.frame_width}, fh: {self.frame_height}")

  def stop_recording(self):
    """
    Stops the video recording.
    """
    if self.timer is not None:
      self.root.after_cancel(self.timer)
    if self.cap is not None:
      self.cap.release()
    self.display_frame(np.zeros((720, 720, 3)).astype(np.uint8))
    self.update_text("")

  def take_image(self):
    """
    Takes an image for registering a new user.
    """
    self.toggle_new_user()
    image = self.image.copy()
    while True:
      name = simpledialog.askstring("Input", "Enter your name:")
      if name is None:
        self.logger.info("Name input was cancelled")
        return
      if name not in self.face_recognizer.labels:
        break
      messagebox.showerror("Error", "This name already exists. Please enter a different name.")

    face_path = self.KNOWN_DIR / f"{name}.jpg"
    cv2.imwrite(str(face_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    self.face_recognizer.add_embedding(image, name)
    self.logger.info(f"New user saved, {str(face_path)}")
    self.status_bar.config(text=f"Registered Users: {self.face_recognizer.nb_known_faces}")
    messagebox.showinfo("Success", f"New face for {name} successfully recorded.")

  def update_timer(self):
    """
    Updates the timer for frame update.
    """
    self.timer = self.root.after(30, self.update_frame)

  def display_frame(self, frame):
    """
    Displays a frame in the GUI.
    """
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((500, 500))
    imgtk = ImageTk.PhotoImage(image=image)
    self.video_label.imgtk = imgtk
    self.video_label.config(image=imgtk)

  def detect_faces(self, frame):
    """
    Detects faces in a frame.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = self.face_recognizer.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

  def recognize_face(self, frame):
    """
    Recognizes faces in a frame.
    """
    return self.face_recognizer.recognize_face(frame)

  def smooth_predictions(self):
    """
    Smoothes the predictions for face recognition.
    """
    if not self.prediction_buffer:
      return 0, -1
    confs, lbls = zip(*self.prediction_buffer)
    avg_conf = np.mean(confs)
    lbl = max(set(lbls), key=lbls.count)
    return avg_conf, lbl

  def put_bbox(self, frame, face):
    """
    Draws bounding boxes around detected faces.
    """
    x, y, w, h = face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

  def update_text(self, text):
    """
    Updates the text displayed in the GUI.
    """
    self.text_label.config(text=text)

  def add_overlay(self, frame):
    """
    Adds an overlay to the frame.
    """
    overlay = cv2.resize(self.overlay, (self.height, self.width))
    alpha = overlay[:, :, 3] / 255
    colors = overlay[:, :, :3]
    alpha_mask = np.dstack((alpha, alpha, alpha))
    frame[:] = frame[:] * (1 - alpha_mask) + colors * alpha_mask

  def post_update(self, frame):
    """
    Update various states after processing a frame, including displaying the frame.
    """
    self.update_text(self.text)
    self.add_overlay(frame)
    self.display_frame(frame)
    self.update_timer()

  def post_detection(self, frame, failed=False):
    """
    Update various states after recognizing a face in a frame and stop the recording.
    """
    if self.frame_count == 0:
      self.logger.info(f"Subject{' failed to ' if failed else ' '}identified, stopping camera feed.")
    self.post_update(frame)
    self.frame_count += 1
    if self.frame_count > 10:
      self.toggle_recording()

  def update_frame(self):
    """
    Main loop for frame processing.
    """
    self.take_img_button["state"] = "disabled"
    ret, frame = self.cap.read()
    if not ret:
      self.update_timer()
      return

    offset = (self.frame_width - self.width) // 2
    frame = frame[:, offset:self.width + offset, :]
    if self.is_detected:
      self.post_detection(frame, failed=self.is_failed)
      return

    self.process_frame(frame)

  def process_frame(self, frame):
    """
    Processes a frame for face recognition.
    """
    self.text = "Align your face"
    img = frame.copy()
    faces = self.detect_faces(img)
    if len(faces) != 1:
      self.post_update(frame)
      return

    face = faces[0]
    face_iou = iou(self.detection_box, face)
    self.text = "Align your face properly"
    self.logger.debug(f"IoU: {face_iou:.4f}")
    if face_iou < 0.65:
      self.post_update(frame)
      return

    # Adding a new user
    if not self.is_detecting:
      self.take_img_button["state"] = "normal"
      self.put_bbox(frame, face)
      self.text = "Press Take Image button"
      self.post_update(frame)
      self.image = img
      return

    self.recognize_and_record(frame, face)

  def recognize_and_record(self, frame, face):
    """
    Recognizes user and records attendance.
    """
    img = frame.copy()
    self.text = "Recognizing face..."
    self.put_bbox(frame, face)
    confidence, label = self.recognize_face(img)
    self.prediction_buffer.append((confidence, label))
    self.logger.debug(f"Buffer size: {len(self.prediction_buffer)}")
    if len(self.prediction_buffer) < self.min_buffer_size:
      self.post_update(frame)
      return

    if len(self.prediction_buffer) == self.max_buffer_size:
      self.is_detected = True
      self.is_failed = True
      self.text = "Recognition failed, please try again"
      self.post_update(frame)
      return

    avg_confidence, most_common_label = self.smooth_predictions()
    name = self.face_recognizer.labels[most_common_label]
    if avg_confidence > self.face_recognizer.threshold:
      self.is_detected = True
      new_record = {'name': name, 'timestamp': pd.Timestamp.now()}
      self.db_manager.record_attendance(new_record)
      self.db_tree.insert("", "end", values=list(new_record.values()))
      self.text = f"Welcome {name}. Your attendance has been recorded"

    self.logger.info(f"Label: {name}, Confidence: {avg_confidence:.4f}")
    self.logger.debug(f"{self.prediction_buffer}")
    self.logger.debug(f"{self.face_recognizer.labels}")
    self.post_update(frame)


def main():
  parser = argparse.ArgumentParser(description="Attendance Recorder using Face Recognition.")
  parser.add_argument("-w", "--weight", type=str, default="", help="Filename of the weight file to be used for the Siamese network. This file must be placed inside the 'data/weights' directory. If not specified, the first '.h5' file found in the directory will be used.")
  parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Threshold for face recognition confidence. If the recognition confidence exceeds this threshold, the face is considered recognized. Default value is 0.5.")
  parser.add_argument("-b", "--min-buffer-size", type=int, default=5, help="Minimum buffer size for face recognition. Default value is 5.")
  parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose command line output.")
  args = parser.parse_args()

  root = tk.Tk()
  logger = setup_logger(10 if args.verbose else 20)

  weight_file = get_weight_dir(args.weight, logger)
  known_imgs, labels = load_known_faces(Path("data/known_faces"), preprocess=True)
  db_manager = DatabaseManager(records_dir=Path("data/records"), logger=logger)
  face_recognizer = FaceRecognizer(weight_path=weight_file, threshold=args.threshold, known_faces=known_imgs, labels=labels, logger=logger)
  ui_manager = UIManager(root, face_recognizer, db_manager, logger, args.min_buffer_size)

  root.mainloop()


if __name__ == "__main__":
  main()
