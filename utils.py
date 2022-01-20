# from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.layers import Conv1D, Conv2D, Conv3D
from typing import Type, Union

ConvND = Union[Conv1D, Conv2D, Conv3D]


def to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def get_conv_layer_type(rank: int) -> Type[ConvND]:
    if rank not in (1, 2, 3):
        raise ValueError("Rank must either be 1, 2 or 3. Received {}.".format(rank))
    conv_layer_types = [None, Conv1D, Conv2D, Conv3D]
    return conv_layer_types[rank]
