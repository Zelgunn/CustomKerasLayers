import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from typing import Union, List


class ExpandDims(Layer):
    def __init__(self, dims: Union[int, List[int]], **kwargs):
        super(ExpandDims, self).__init__(trainable=False, **kwargs)

        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        self.dims = dims

    def call(self, inputs, **kwargs):
        for dim in self.dims:
            if dim >= 0:
                dim += 1
            inputs = tf.expand_dims(inputs, dim)

        return inputs

    def get_config(self):
        config = super(ExpandDims, self).get_config()
        config.update({"dims": self.dims})
        return config
