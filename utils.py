import cv2
import numpy as np

def iou(boxA, boxB):
  boxes = np.vstack([boxA, boxB])
  boxes[:, 2] += boxes[:, 0]
  boxes[:, 3] += boxes[:, 1]
  xA, yA = boxes[:, :2].max(axis=0)
  xB, yB = boxes[:, 2:].min(axis=0)
  inter_area = max(0, xB-xA+1) * max(0, yB-yA+1)
  boxes_area = (boxes[:, 2]-boxes[:, 0]+1) * (boxes[:, 3]-boxes[:, 1]+1)
  return inter_area / (boxes_area.sum() - inter_area)

def load_known_faces(known_dir):
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
  import logging
  logger = logging.getLogger('AttendanceApp')
  logger.setLevel(logging_level)
  ch = logging.StreamHandler()
  ch.setLevel(logging_level)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  ch.setFormatter(formatter)
  logger.addHandler(ch)
  return logger
