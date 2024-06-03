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

def load_known_faces(known_dir):
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

def get_weight_file(weight_fn, logger):
  """
  Retrieve the weight file for the model.

  Parameters:
  weight_fn (str): Filename of the weight file.
  logger (logging.Logger): Logger instance for logging errors.

  Returns:
  Path: Path to the weight file.

  Raises:
  SystemExit: If no valid weight file is found.
  """
  weight_dir = Path('data/weights')
  if weight_fn:
    weight_file = weight_dir / weight_fn
  else:
    weight_file = next(weight_dir.glob("*.h5"), None)
  if not weight_file or not weight_file.exists():
    logger.error("No valid weight file found.")
    exit(1)
  return weight_file
