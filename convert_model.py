import argparse
from pathlib import Path

import tensorflow as tf
import tf2onnx

from siamese_network import SiameseNetwork


VALID_FMT = ["onnx", "h5"]

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("weight_fn", type=str, help="File name of the weight (.h5). The file must be placed inside the 'data/weights' directory.")
  parser.add_argument("-f", "--formats", nargs="*", help=f"Output formats, accepted formats: {VALID_FMT}. Default to all formats.")
  return parser.parse_args()

def convert_model(model, output_path):
  model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=[tf.TensorSpec(inp.shape, dtype=inp.dtype, name=inp.name) for inp in model.inputs],
    output_path=output_path,
  )

def main(weight_path, formats=None):
  if formats is None:
    return
  if len(formats) == 0:
    return

  net =  SiameseNetwork()
  siamese_net = net.get_siamese_net()
  cnn = net.get_single_branch_model((105,105,1))
  simscore = net.get_similarity_model((cnn.output_shape[1],))

  siamese_net.load_weights(weight_path)
  cnn.set_weights(siamese_net.get_layer(index=2).get_weights())
  simscore.set_weights(siamese_net.get_layer(index=-1).get_weights())

  output_dir = weight_path.parent / weight_path.stem
  if "onnx" in formats:
    convert_model(cnn, output_dir / "cnn.onnx")
    convert_model(simscore, output_dir / "simscore.onnx")

  if "h5" in formats:
    cnn.save_weights(output_dir / "cnn.weights.h5")
    simscore.save_weights(output_dir / "simscore.weights.h5")


if __name__ == "__main__":
  args = parse_args()
  if args.formats is None:
    formats = VALID_FMT
  else:
    formats = [fmt for fmt in args.format if fmt in VALID_FMT]

  weight_path = Path("data/weights") / args.weight_fn
  main(weight_path, formats)
