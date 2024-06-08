import argparse
from pathlib import Path

from siamese_network import Siamese

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("weight_fn", type=str, help="File name of the weight (.h5). The file must be placed inside the 'data/weights' directory.")
  return parser.parse_args()

def main(args):
  WEIGHTS_DIR = Path("data/weights")
  weight_path = WEIGHTS_DIR / args.weight_fn
  assert weight_path.exists()
  assert weight_path.suffix.casefold() == ".h5"
  net = Siamese(weight_path)
  net.export_onnx()

if __name__ == "__main__":
  args = parse_args()
  main(args)
