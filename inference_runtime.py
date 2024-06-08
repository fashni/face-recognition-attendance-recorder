import numpy as np

from utils import preprocess_images


class SiameseTF:
  def __init__(self, weight_dir=None, width=105, height=105, cells=1):
    from siamese_network import SiameseNetwork
    net = SiameseNetwork()
    self.cnn = net.get_single_branch_model(input_shape=(width, height, cells))
    self.simscore = net.get_similarity_model(input_shape=(self.cnn.output_shape[1],))
    self.nb_known_faces = 0
    self.weight_dir = None
    if weight_dir is not None:
      self.set_weight(weight_dir)

  def set_weight(self, weight_dir):
    self.weight_dir = weight_dir
    self.cnn.load_weights(weight_dir / "cnn.weights.h5")
    self.simscore.load_weights(weight_dir / "simscore.weights.h5")

  def predict(self, img, known_embeddings, batch_size=1, preprocess=False):
    if known_embeddings is None:
      return
    embedding = self.get_embedding(img, batch_size=batch_size, preprocess=preprocess)
    diff = np.abs(known_embeddings - embedding)
    res = self.simscore.predict(diff, batch_size=min(self.nb_known_faces, 32))
    return res[:, 0]

  def get_embedding(self, imgs, batch_size=1, preprocess=False):
    if preprocess:
      imgs = preprocess_images(imgs)
    return self.cnn.predict(imgs, batch_size=batch_size)


class SiameseONNX:
  def __init__(self, weight_dir):
    import onnxruntime as ort
    self.cnn_session = ort.InferenceSession(weight_dir / "cnn.onnx", providers=ort.get_available_providers())
    self.sim_session = ort.InferenceSession(weight_dir / "simscore.onnx", providers=ort.get_available_providers())
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
