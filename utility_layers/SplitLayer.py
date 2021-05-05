import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from typing import Union, List


class SplitLayer(Layer):
    def __init__(self,
                 num_or_size_splits: Union[int, List[int]],
                 axis=0,
                 num=None,
                 **kwargs):
        super(SplitLayer, self).__init__(trainable=False, **kwargs)

        self.num_or_size_splits = num_or_size_splits
        self.axis = axis
        self.num = num

    def call(self, inputs, **kwargs):
        outputs = tf.split(inputs, num_or_size_splits=self.num_or_size_splits, axis=self.axis, num=self.num)
        return outputs

    def get_config(self):
        config = super(SplitLayer, self).get_config()
        config.update({
            "num_or_size_splits": self.num_or_size_splits,
            "axis": self.axis,
            "num": self.num,
        })
        return config
