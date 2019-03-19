from tensorflow.python.keras.engine.base_layer import Layer
from typing import Tuple


class CompositeLayer(Layer):
    def build_sub_layer(self, layer: Layer, input_shape: Tuple):
        layer.build(input_shape)

        self._trainable_weights += layer._trainable_weights
        self._non_trainable_weights += layer._non_trainable_weights
