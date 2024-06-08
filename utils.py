from pathlib import Path

import cv2
import numpy as np


def iou(boxA, boxB):
  """
  Compute the Intersection over Union (IoU) of two bounding boxes.

  Parameters:
  boxA (array-like): Coordinates of the first bounding box in the format [x, y, width, height].
  boxB (array-like): Coordinates of the second bounding box in the format [x, y, width, height].

  Returns:
  float: The IoU value between the two bounding boxes.
  """
  boxes = np.vstack([boxA, boxB])
  boxes[:, 2] += boxes[:, 0]
  boxes[:, 3] += boxes[:, 1]
  xA, yA = boxes[:, :2].max(axis=0)
  xB, yB = boxes[:, 2:].min(axis=0)
  inter_area = max(0, xB-xA+1) * max(0, yB-yA+1)
  boxes_area = (boxes[:, 2]-boxes[:, 0]+1) * (boxes[:, 3]-boxes[:, 1]+1)
  return inter_area / (boxes_area.sum() - inter_area)

def load_known_faces(known_dir, preprocess=True):
  """
  Load known face images from a directory.

  Parameters:
  known_dir (Path): Path to the directory containing known face images.

  Returns:
  tuple: A tuple containing a list of known face images and a list of corresponding labels.
  """
  labels = []
  known_ppl = []
  for person_file in known_dir.iterdir():
    if not person_file.is_file():
      continue
    if person_file.suffix.casefold() != ".jpg":
      continue
    labels.append(".".join(person_file.name.split('.')[:-1]))
    known_ppl.append(cv2.imread(str(person_file)))
  if preprocess:
    known_ppl = preprocess_images(known_ppl)
  return known_ppl, labels

def setup_logger(logging_level):
  """
  Set up a logger for the application.

  Parameters:
  logging_level (int): Logging level (e.g., logging.DEBUG, logging.INFO).

  Returns:
  logging.Logger: Configured logger instance.
  """
  import logging
  logger = logging.getLogger('AttendanceApp')
  logger.setLevel(logging_level)
  ch = logging.StreamHandler()
  ch.setLevel(logging_level)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  ch.setFormatter(formatter)
  logger.addHandler(ch)
  return logger

def get_weight_dir(weight_dir, suffix, logger):
  """
  Retrieve and validate the directory containing weight files for the model.

  Parameters:
  weight_dir (str): Directory name containing the weight files.
  suffix (str): The suffix/format of the weight files to be used.
  logger (logging.Logger): Logger instance for logging errors.

  Returns:
  Path: Path to the weight directory.

  Raises:
  SystemExit: If no valid weight directory is found.
  """
  root_dir = Path('data/weights')
  include_files = [f"cnn{suffix}", f"simscore{suffix}"]
  is_all_exist = lambda d: all((d / f).exists() for f in include_files)
  res = None
  if weight_dir:
    res = root_dir / weight_dir
  else:
    dirs = [item for item in root_dir.iterdir() if item.is_dir() and is_all_exist(item)]
    res = dirs[0] if dirs else None
  if res is None or not res.exists():
    logger.error("No valid weight directory found.")
    exit(1)
  return res

def preprocess_images(imgs, width=105, height=105, channel=1):
  if isinstance(imgs, np.ndarray) and len(imgs.shape) == 3:
    imgs = [imgs]
  res = []
  for image in imgs:
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (width, height))
    res.append(img.reshape(width, height, channel).astype(np.float32))
  return np.array(res)
