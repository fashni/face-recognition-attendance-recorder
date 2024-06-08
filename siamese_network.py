import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Sequential, Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2

from utils import preprocess_images


class SiameseNetwork(object):
  def __init__(self, seed=0):
    """
    Seed - The seed used to initialize the weights
    """
    K.clear_session()
    self.load_file = None
    self.initialize_seed(seed)

  def get_siamese_net(self, width=105, height=105, cells=1):
    """
    width, height, cells - used for defining the tensors used for the input images
    """
    # Define the matrices for the input images
    input_shape = (width, height, cells)
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Get the CNN architecture as presented in the paper (read the readme for more information)
    model = self._get_architecture(input_shape)
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a layer to combine the two CNNs
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_siamese_dist = L1_layer([encoded_l, encoded_r])

    # An output layer with Sigmoid activation function
    prediction = Dense(1, activation='sigmoid', bias_initializer=self.initialize_bias)(L1_siamese_dist)

    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
    return siamese_net

  def initialize_seed(self, seed):
    """
    Initialize seed all for environment
    """
    self.seed = seed
    os.environ['PYTHONHASHSEED'] = str(self.seed)
    random.seed(self.seed)
    np.random.seed(self.seed)
    tf.random.set_seed(self.seed)

  def initialize_weights(self, shape, dtype=None):
    """
    Called when initializing the weights of the siamese model, uses the random_normal function of keras to return a
    tensor with a normal distribution of weights.
    """
    return K.random_normal(shape, mean=0.0, stddev=0.01, dtype=dtype, seed=self.seed)

  def initialize_bias(self, shape, dtype=None):
    """
    Called when initializing the biases of the siamese model, uses the random_normal function of keras to return a
    tensor with a normal distribution of weights.
    """
    return K.random_normal(shape, mean=0.5, stddev=0.01, dtype=dtype, seed=self.seed)

  def _get_architecture(self, input_shape):
    """
    Returns a Convolutional Neural Network based on the input shape given of the images. This is the CNN network
    that is used inside the siamese model. Uses parameters from the siamese one shot paper.
    """
    model = Sequential()
    model.add(
      Conv2D(filters=64,
            kernel_size=(10, 10),
            input_shape=input_shape,
            kernel_initializer=self.initialize_weights,
            kernel_regularizer=l2(2e-4),
            name='Conv1'
            ))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D())

    model.add(
      Conv2D(filters=128,
            kernel_size=(7, 7),
            kernel_initializer=self.initialize_weights,
            bias_initializer=self.initialize_bias,
            kernel_regularizer=l2(2e-4),
            name='Conv2'
            ))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D())

    model.add(
      Conv2D(filters=128,
            kernel_size=(4, 4),
            kernel_initializer=self.initialize_weights,
            bias_initializer=self.initialize_bias,
            kernel_regularizer=l2(2e-4),
            name='Conv3'
            ))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D())

    model.add(
      Conv2D(filters=256,
            kernel_size=(4, 4),
            kernel_initializer=self.initialize_weights,
            bias_initializer=self.initialize_bias,
            kernel_regularizer=l2(2e-4),
            name='Conv4'
            ))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Flatten())
    model.add(
      Dense(4096,
            activation='sigmoid',
            kernel_initializer=self.initialize_weights,
            kernel_regularizer=l2(2e-3),
            bias_initializer=self.initialize_bias))
    return model

  def get_single_branch_model(self, input_shape):
    """
    Returns a model representing a single branch of the Siamese network.
    """
    model = self._get_architecture(input_shape)
    single_input = Input(shape=input_shape)
    single_output = model(single_input)
    single_branch_model = Model(inputs=single_input, outputs=single_output)
    return single_branch_model

  def get_similarity_model(self, input_shape):
    input_diff = Input(shape=input_shape)
    similarity_score = Dense(1, activation='sigmoid')(input_diff)
    similarity_model = Model(inputs=input_diff, outputs=similarity_score)
    return similarity_model
