import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from typing import Union, List

from CustomKerasLayers.layers.MaskedConv import MaskedConv


class MaskedConvStack(Layer):
    """ Convolutional Autoregressive Model
    """

    def __init__(self,
                 rank: int,
                 filters: List[int],
                 mask_first_center: bool,
                 kernel_size: Union[int, List[int]],
                 **kwargs):
        super(MaskedConvStack, self).__init__(**kwargs)

        self.rank = rank
        self.filters = filters
        self.mask_first_center = mask_first_center
        self.kernel_size = kernel_size

        self.layers = [
            MaskedConv(rank=rank,
                       filters=filters[i],
                       kernel_size=kernel_size,
                       mask_center=mask_first_center and (i == 0),
                       strides=1,
                       padding="same")
            for i in range(len(filters))
        ]

    def call(self, inputs, **kwargs):
        outputs = []
        for layer in self.layers:
            outputs.append(layer(inputs))
        outputs = tf.concat(outputs, axis=-1)
        return outputs

    def get_config(self):
        return {
            "rank": self.rank,
            "filters": self.filters,
            "mask_first_center": self.mask_first_center,
            "kernel_size": self.kernel_size,
        }


class MaskedConv1DStack(Layer):
    def __init__(self,
                 filters: List[int],
                 mask_first_center: bool,
                 kernel_size: Union[int, List[int]],
                 **kwargs):
        super(MaskedConv1DStack, self).__init__(rank=1,
                                                filters=filters,
                                                mask_first_center=mask_first_center,
                                                kernel_size=kernel_size,
                                                **kwargs)


class MaskedConv2DStack(Layer):
    def __init__(self,
                 filters: List[int],
                 mask_first_center: bool,
                 kernel_size: Union[int, List[int]],
                 **kwargs):
        super(MaskedConv2DStack, self).__init__(rank=2,
                                                filters=filters,
                                                mask_first_center=mask_first_center,
                                                kernel_size=kernel_size,
                                                **kwargs)


class MaskedConv3DStack(Layer):
    def __init__(self,
                 filters: List[int],
                 mask_first_center: bool,
                 kernel_size: Union[int, List[int]],
                 **kwargs):
        super(MaskedConv3DStack, self).__init__(rank=3,
                                                filters=filters,
                                                mask_first_center=mask_first_center,
                                                kernel_size=kernel_size,
                                                **kwargs)
