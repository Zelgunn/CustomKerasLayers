import tensorflow as tf
from tensorflow.python.keras.layers import InputSpec, Activation, Layer
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.layers import Conv1D, Conv2D, Conv3D
# from tensorflow.python.keras.layers import Conv2DTranspose, Conv3DTranspose
# from tensorflow.python.keras.utils import conv_utils
# from tensorflow.python.ops.init_ops import Constant, VarianceScaling
# from tensorflow.python.keras import activations, regularizers, constraints
from typing import Tuple, List, Union, AnyStr, Callable, Dict, Optional, Type


from CustomKerasLayers.utils import get_conv_layer_type, ConvND
# from CustomKerasLayers.layers.Conv1DTranspose import Conv1DTranspose

class ConvMixerND(Layer):
    def __init__(self,
                 rank: int,
                 filters: int,
                 patch_size: Union[int, Tuple, List],
                 use_bias: bool,
                 **kwargs):
        # region Check parameters
        if rank not in (1, 2, 3):
            raise ValueError("Rank must either be 1, 2 or 3. Received {}.".format(rank))

        # endregion

        super(ConvMixerND, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.patch_size = patch_size
        self.use_bias = use_bias

        self.patch_convolutional_layer = self._make_patch_convolutional_layer(filters, patch_size, use_bias)

    def _make_patch_convolutional_layer(self,
                                        filters: int = None,
                                        patch_size: int = None,
                                        use_bias: bool = None):
        filters = self.filters if filters is None else filters
        patch_size = self.patch_size if patch_size is None else patch_size
        use_bias = self.use_bias if use_bias is None else use_bias
        return self._make_conv_layer(filters=filters, kernel_size=patch_size, strides=patch_size, padding="valid",
                                     use_bias=use_bias)

    def _make_conv_layer(self,
                         filters: int,
                         kernel_size: Union[int, Tuple, List],
                         strides: Union[int, Tuple, List],
                         padding: str,
                         use_bias: bool,
                         ):
        return self.conv_layer_type(filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    activation=None,
                                    use_bias=use_bias)

    @property
    def conv_layer_type(self) -> Type[ConvND]:
        return get_conv_layer_type(self.rank)
