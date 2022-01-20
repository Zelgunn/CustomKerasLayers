import tensorflow as tf
from tensorflow.python.keras.layers import InputSpec, Layer  # , Activation
from tensorflow.python.keras.layers.convolutional import Conv
# from tensorflow.python.keras.layers import Conv1D, Conv2D, Conv3D
# from tensorflow.python.keras.layers import Conv2DTranspose, Conv3DTranspose
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras import regularizers, constraints, initializers  # , activations
from typing import Tuple, List, Union, AnyStr, Callable, Dict, Optional, Type

from CustomKerasLayers.utils import get_conv_layer_type, ConvND


class DepthwiseConvolutionND(Layer):
    def __init__(self,
                 rank: int,
                 channels_multiplier: int,
                 kernel_size: Union[int, Tuple, List],
                 strides: Union[int, Tuple, List],
                 padding: str,
                 use_bias: bool,
                 data_format: Optional[AnyStr],
                 dilation_rate: Union[int, Tuple, List],
                 kernel_initializer: Optional[Union[Dict, AnyStr, Callable]],
                 bias_initializer: Optional[Union[Dict, AnyStr, Callable]],
                 kernel_regularizer: Optional[Union[Dict, AnyStr, Callable]],
                 bias_regularizer: Optional[Union[Dict, AnyStr, Callable]],
                 activity_regularizer: Optional[Union[Dict, AnyStr, Callable]],
                 kernel_constraint: Optional[Union[Dict, AnyStr, Callable]],
                 bias_constraint: Optional[Union[Dict, AnyStr, Callable]],
                 **kwargs):
        # region Check parameters
        if rank not in (1, 2, 3):
            raise ValueError("Rank must either be 1, 2 or 3. Received {}.".format(rank))

        if not isinstance(channels_multiplier, int) or channels_multiplier <= 0:
            raise ValueError("Channels multiplier must be a strictly positive integer. Received {}."
                             .format(channels_multiplier))
        # endregion

        super(DepthwiseConvolutionND, self).__init__(**kwargs)
        self.rank = rank
        self.channels_multiplier = channels_multiplier
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias

        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, "dilation_rate")

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.conv_layer = self.conv_layer_type(filters=self.channels_multiplier,
                                               kernel_size=self.kernel_size,
                                               strides=self.strides,
                                               padding=self.padding,
                                               use_bias=False,
                                               data_format="channels_last",
                                               dilation_rate=self.dilation_rate,
                                               kernel_initializer=self.kernel_initializer,
                                               kernel_regularizer=self.kernel_regularizer,
                                               activity_regularizer=self.activity_regularizer,
                                               kernel_constraint=self.kernel_constraint)
        self.bias = None
        self._input_channels: Optional[int] = None

    def build(self, input_shape):
        channels = input_shape[self.channel_axis]
        self._input_channels = channels
        if self.use_bias:
            self.bias = self.add_weight(shape=(channels * self.channels_multiplier,),
                                        initializer=self.bias_initializer,
                                        name="bias",
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.input_spec = InputSpec(ndim=self.rank + 2, axes={self.channel_axis: input_shape[self.channel_axis]})
        super(DepthwiseConvolutionND, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs_shape = inputs.shape
        batch_size = inputs_shape[0]
        input_channels = inputs_shape[self.channel_axis]

        if self.data_format == "channels_first":
            input_dimensions = inputs_shape[2:]
        else:
            input_dimensions = inputs_shape[1:-1]
            dimensions_axes = list(range(1, self.rank + 1))
            inputs = tf.transpose(inputs, perm=[0, self.rank + 1, *dimensions_axes])

        print("a)", inputs.shape)
        inputs = tf.reshape(inputs, shape=[batch_size * input_channels, *input_dimensions, 1])
        print("b)", inputs.shape)
        outputs = self.conv_layer(inputs)
        print("c)", outputs.shape)

        output_dimensions = outputs.shape[1:-1]
        if self.data_format == "channels_first":
            dimensions_axes = list(range(1, self.rank + 1))
            if self.channels_multiplier > 1:
                outputs = tf.transpose(outputs, perm=[0, self.rank + 1, *dimensions_axes])
                print("d)", outputs.shape)
            outputs = tf.reshape(outputs, shape=[batch_size, self.output_channels, *output_dimensions])
            print("e)", outputs.shape)
            if self.use_bias:
                outputs += tf.reshape(self.bias, [1, self.output_channels] + [1] * self.rank)
        else:
            outputs = tf.reshape(outputs, shape=[batch_size, input_channels, *output_dimensions,
                                                 self.channels_multiplier])
            print("d)", outputs.shape)
            dimensions_axes = list(range(2, self.rank + 2))
            outputs = tf.transpose(outputs, perm=[0, *dimensions_axes, 1, self.rank + 2])
            print("e)", outputs.shape)
            outputs = tf.reshape(outputs, shape=[batch_size, *output_dimensions, self.output_channels])
            print("f)", outputs.shape)
            if self.use_bias:
                outputs += tf.reshape(self.bias, [1] * (self.rank + 1) + [self.output_channels])

        return outputs

    @property
    def conv_layer_type(self) -> Type[ConvND]:
        return get_conv_layer_type(self.rank)

    @property
    def output_channels(self) -> int:
        if self._input_channels is None:
            raise ValueError("You must build this layer before accessing the number of output channels.")
        return self._input_channels * self.channels_multiplier

    @property
    def channel_axis(self):
        if self.data_format == "channels_first":
            return 1
        else:
            return -1


def main():
    layer = DepthwiseConvolutionND(rank=2,
                                   channels_multiplier=2,
                                   kernel_size=3,
                                   strides=2,
                                   padding="same",
                                   use_bias=True,
                                   data_format="channels_last",
                                   dilation_rate=1,
                                   kernel_initializer="he_normal",
                                   bias_initializer="zeros",
                                   kernel_regularizer=None,
                                   bias_regularizer=None,
                                   activity_regularizer=None,
                                   kernel_constraint=None,
                                   bias_constraint=None
                                   )
    x = tf.random.normal(shape=[5, 32, 32, 4])
    # x = tf.random.normal(shape=[5, 4, 32, 32])
    y = layer(x)
    print(y.shape)


if __name__ == "__main__":
    main()
