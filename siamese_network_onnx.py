import numpy as np
import onnxruntime as ort

from utils import preprocess_images


class Siamese:
  def __init__(self, weight_dir):
    self.cnn_session = ort.InferenceSession(weight_dir / "single_net.onnx", providers=ort.get_available_providers())
    self.sim_session = ort.InferenceSession(weight_dir / "similarity_net.onnx", providers=ort.get_available_providers())
    self.get_io_details()

  def get_io_details(self):
    cnn_inputs = self.cnn_session.get_inputs()
    sim_inputs = self.sim_session.get_inputs()
    cnn_outputs = self.cnn_session.get_outputs()
    sim_outputs = self.sim_session.get_outputs()
    self.cnn_input_names = [inp.name for inp in cnn_inputs]
    self.sim_input_names = [inp.name for inp in sim_inputs]
    self.cnn_output_names = [out.name for out in cnn_outputs]
    self.sim_output_names = [out.name for out in sim_outputs]

  def predict(self, img, known_embeddings, batch_size=1, preprocess=False):
    embedding = self.get_embedding(img, batch_size=batch_size, preprocess=preprocess)
    diff = np.abs(known_embeddings - embedding)
    res = self.sim_session.run(self.sim_output_names, {self.sim_input_names[0]: diff})[0]
    return res[:, 0]

  def get_embedding(self, imgs, batch_size=1, preprocess=False):
    if preprocess:
      imgs = preprocess_images(imgs)
    return self.cnn_session.run(self.cnn_output_names, {self.cnn_input_names[0]: imgs})[0]
