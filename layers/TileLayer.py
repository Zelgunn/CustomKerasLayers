import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from typing import List


class TileLayer(Layer):
    def __init__(self, multiples: List[int], **kwargs):
        super(TileLayer, self).__init__(trainable=False, **kwargs)
        self.multiples = multiples

    def call(self, inputs, **kwargs):
        outputs = tf.tile(inputs, multiples=[1, *self.multiples])
        return outputs

    def get_config(self):
        config = super(TileLayer, self).get_config()
        config.update({"multiples": self.multiples})
        return config
