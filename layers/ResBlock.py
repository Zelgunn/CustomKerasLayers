from keras.layers import InputSpec
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import Deconv2D, Deconv3D
from keras.initializers import VarianceScaling
from keras.utils import conv_utils
from keras.utils.generic_utils import to_list
from keras import activations, initializers, regularizers, constraints
from keras import backend as K
from typing import Tuple, List, Union, AnyStr, Callable, Dict
import numpy as np

from layers import CompositeLayer


# region Basic blocks
class ResBasicBlockND(CompositeLayer):
    def __init__(self,
                 rank: int,
                 filters: int,
                 depth: int,
                 kernel_size: Union[int, Tuple, List],
                 strides: Union[int, Tuple, List],
                 data_format: Union[None, AnyStr],
                 dilation_rate: Union[int, Tuple, List],
                 activation: Union[None, AnyStr, Callable],
                 use_bias: bool,
                 kernel_initializer: Union[Dict, AnyStr, Callable],
                 bias_initializer: Union[Dict, AnyStr, Callable],
                 kernel_regularizer: Union[None, Dict, AnyStr, Callable],
                 bias_regularizer: Union[None, Dict, AnyStr, Callable],
                 activity_regularizer: Union[None, Dict, AnyStr, Callable],
                 kernel_constraint: Union[None, Dict, AnyStr, Callable],
                 bias_constraint: Union[None, Dict, AnyStr, Callable],
                 **kwargs):

        assert rank in [1, 2, 3]
        assert depth > 0

        super(ResBasicBlockND, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.depth = depth

        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, "kernel_size")
        self.strides = conv_utils.normalize_tuple(strides, rank, "strides")

        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, "dilation_rate")
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.conv_layers = []
        self.projection_layer = None
        self.residual_multiplier = None
        self.conv_biases = []
        self.activation_biases = []
        self.residual_bias = None

        self.input_spec = InputSpec(ndim=self.rank + 2)

    def get_conv_layer_type(self):
        return Conv1D if self.rank is 1 else Conv2D if self.rank is 2 else Conv3D

    def init_layers(self, input_shape):
        conv_layer_type = self.get_conv_layer_type()
        for i in range(self.depth):
            strides = self.strides if (i == 0) else 1
            kernel_initializer = self.kernel_initializer if (i == 0) else K.zeros
            conv_layer = conv_layer_type(filters=self.filters,
                                         kernel_size=self.kernel_size,
                                         strides=strides,
                                         padding="same",
                                         data_format=self.data_format,
                                         dilation_rate=self.dilation_rate,
                                         use_bias=False,
                                         kernel_initializer=kernel_initializer,
                                         kernel_regularizer=self.kernel_regularizer,
                                         activity_regularizer=self.activity_regularizer,
                                         kernel_constraint=self.kernel_constraint,
                                         bias_constraint=self.bias_constraint)
            self.conv_layers.append(conv_layer)

        if self.use_projection(input_shape):
            projection_kernel_size = conv_utils.normalize_tuple(1, self.rank, "projection_kernel_size")
            self.projection_layer = conv_layer_type(filters=self.filters,
                                                    kernel_size=projection_kernel_size,
                                                    strides=self.strides,
                                                    padding="same",
                                                    data_format=self.data_format,
                                                    dilation_rate=self.dilation_rate,
                                                    use_bias=False,
                                                    kernel_initializer=self.kernel_initializer,
                                                    kernel_regularizer=self.kernel_regularizer,
                                                    activity_regularizer=self.activity_regularizer,
                                                    kernel_constraint=self.kernel_constraint,
                                                    bias_constraint=self.bias_constraint)

    def build(self, input_shape):
        self.init_layers(input_shape)
        intermediate_shape = input_shape

        with K.name_scope("residual_basic_block_weights"):
            for i in range(self.depth):
                self.build_sub_layer(self.conv_layers[i], intermediate_shape)
                intermediate_shape = self.conv_layers[i].compute_output_shape(intermediate_shape)

            if self.projection_layer is not None:
                self.build_sub_layer(self.projection_layer, input_shape)

            self.residual_multiplier = self.add_weight(name="residual_multiplier", shape=[], dtype=K.floatx(),
                                                       initializer=K.ones)
            if self.use_bias:
                for i in range(self.depth):
                    conv_bias = self.add_weight(name="conv_bias", shape=[], dtype=K.floatx(), initializer=K.zeros)
                    self.conv_biases.append(conv_bias)

                    if i < (self.depth - 1):
                        activation_bias = self.add_weight(name="activation_bias", shape=[], dtype=K.floatx(),
                                                          initializer=K.zeros)
                        self.activation_biases.append(activation_bias)

                self.residual_bias = self.add_weight(name="residual_bias", shape=[], dtype=K.floatx(),
                                                     initializer=K.zeros)

        self.input_spec = InputSpec(ndim=self.rank + 2, axes={self.channel_axis: input_shape[self.channel_axis]})
        super(ResBasicBlockND, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = inputs
        for i in range(self.depth):
            if self.use_bias:
                outputs = outputs + self.conv_biases[i]
            outputs = self.conv_layers[i](outputs)

            if i < (self.depth - 1):
                if self.activation is not None:
                    if self.use_bias:
                        outputs = outputs + self.activation_biases[i]
                    outputs = self.activation(outputs)

        if self.use_projection(K.int_shape(inputs)):
            inputs = self.projection_layer(inputs)

        # x_k+1 = x_k + a*f(x_k) + b
        outputs = outputs * self.residual_multiplier
        if self.use_bias:
            outputs = outputs + self.residual_bias
        outputs = inputs + outputs

        outputs = self.activation(outputs)
        return outputs

    def use_projection(self, input_shape):
        strides = to_list(self.strides, allow_tuple=True)
        for stride in strides:
            if stride != 1:
                return True

        return input_shape[self.channel_axis] != self.filters

    def compute_output_shape(self, input_shape):
        def get_new_space(space):
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding="same",
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tuple(new_space)

        if self.channels_first:
            return (input_shape[0], self.filters) + get_new_space(input_shape[2:])
        else:
            return (input_shape[0],) + get_new_space(input_shape[1:-1]) + (self.filters,)

    @property
    def channel_axis(self):
        if self.data_format == "channels_first":
            return 1
        else:
            return -1

    @property
    def channels_first(self):
        return self.data_format == "channels_first"

    def get_config(self):
        config = \
            {
                "rank": self.rank,
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "depth": self.depth,
                "strides": self.strides,
                "padding": "same",
                "data_format": self.data_format,
                "dilation_rate": self.dilation_rate,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "activity_regularizer": regularizers.serialize(self.activity_regularizer),
                "kernel_constraint": constraints.serialize(self.kernel_constraint),
                "bias_constraint": constraints.serialize(self.bias_constraint)
            }
        base_config = super(ResBasicBlockND, self).get_config()
        return {**base_config, **config}


class ResBasicBlock1D(ResBasicBlockND):
    def __init__(self, filters,
                 depth=2,
                 kernel_size=3,
                 strides=1,
                 data_format=None,
                 dilation_rate=1,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer="he_normal",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ResBasicBlock1D, self).__init__(rank=1,
                                              filters=filters, depth=depth, kernel_size=kernel_size,
                                              strides=strides, data_format=data_format,
                                              dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                              kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                              activity_regularizer=activity_regularizer,
                                              kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                              **kwargs)

    def get_config(self):
        config = super(ResBasicBlock1D, self).get_config()
        config.pop("rank")
        return config


class ResBasicBlock2D(ResBasicBlockND):
    def __init__(self, filters,
                 depth=2,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 data_format=None,
                 dilation_rate=1,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer="he_normal",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ResBasicBlock2D, self).__init__(rank=2,
                                              filters=filters, depth=depth, kernel_size=kernel_size,
                                              strides=strides, data_format=data_format,
                                              dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                              kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                              activity_regularizer=activity_regularizer,
                                              kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                              **kwargs)

    def get_config(self):
        config = super(ResBasicBlock2D, self).get_config()
        config.pop("rank")
        return config


class ResBasicBlock3D(ResBasicBlockND):
    def __init__(self, filters,
                 depth=2,
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 data_format=None,
                 dilation_rate=1,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer="he_normal",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ResBasicBlock3D, self).__init__(rank=3,
                                              filters=filters, depth=depth, kernel_size=kernel_size,
                                              strides=strides, data_format=data_format,
                                              dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                              kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                              activity_regularizer=activity_regularizer,
                                              kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                              **kwargs)

    def get_config(self):
        config = super(ResBasicBlock3D, self).get_config()
        config.pop("rank")
        return config


class ResBasicBlockNDTranspose(ResBasicBlockND):
    def get_conv_layer_type(self):
        assert self.rank in [2, 3]
        return Deconv2D if self.rank is 2 else Deconv3D

    def compute_output_shape(self, input_shape):
        def get_new_space(space):
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.deconv_length(
                    space[i],
                    self.strides[i],
                    self.kernel_size[i],
                    padding="same",
                    output_padding=None)
                new_space.append(new_dim)
            return tuple(new_space)

        if self.channels_first:
            return (input_shape[0], self.filters) + get_new_space(input_shape[2:])
        else:
            return (input_shape[0],) + get_new_space(input_shape[1:-1]) + (self.filters,)


class ResBasicBlock2DTranspose(ResBasicBlockNDTranspose):
    def __init__(self, filters,
                 depth=2,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 data_format=None,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer="he_normal",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ResBasicBlock2DTranspose, self).__init__(rank=2,
                                                       filters=filters, depth=depth, kernel_size=kernel_size,
                                                       strides=strides, data_format=data_format,
                                                       activation=activation, use_bias=use_bias,
                                                       kernel_initializer=kernel_initializer,
                                                       bias_initializer=bias_initializer,
                                                       kernel_regularizer=kernel_regularizer,
                                                       bias_regularizer=bias_regularizer,
                                                       activity_regularizer=activity_regularizer,
                                                       kernel_constraint=kernel_constraint,
                                                       bias_constraint=bias_constraint,
                                                       **kwargs)

    def get_config(self):
        config = super(ResBasicBlock2DTranspose, self).get_config()
        config.pop("rank")
        return config


class ResBasicBlock3DTranspose(ResBasicBlockNDTranspose):
    def __init__(self, filters,
                 depth=2,
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 data_format=None,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer="he_normal",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ResBasicBlock3DTranspose, self).__init__(rank=3,
                                                       filters=filters, depth=depth, kernel_size=kernel_size,
                                                       strides=strides, data_format=data_format,
                                                       activation=activation, use_bias=use_bias,
                                                       kernel_initializer=kernel_initializer,
                                                       bias_initializer=bias_initializer,
                                                       kernel_regularizer=kernel_regularizer,
                                                       bias_regularizer=bias_regularizer,
                                                       activity_regularizer=activity_regularizer,
                                                       kernel_constraint=kernel_constraint,
                                                       bias_constraint=bias_constraint,
                                                       **kwargs)

    def get_config(self):
        config = super(ResBasicBlock3DTranspose, self).get_config()
        config.pop("rank")
        return config


# endregion

# region Blocks (of basic blocks)

class ResBlockND(CompositeLayer):
    def __init__(self,
                 rank: int,
                 filters: int,
                 basic_block_count: int,
                 basic_block_depth: int,
                 kernel_size: Union[int, Tuple, List],
                 strides: Union[int, Tuple, List],
                 data_format: Union[None, AnyStr],
                 dilation_rate: Union[int, Tuple, List],
                 activation: Union[None, AnyStr, Callable],
                 use_bias: bool,
                 kernel_initializer: Union[Dict, AnyStr, Callable],
                 bias_initializer: Union[Dict, AnyStr, Callable],
                 kernel_regularizer: Union[None, Dict, AnyStr, Callable],
                 bias_regularizer: Union[None, Dict, AnyStr, Callable],
                 activity_regularizer: Union[None, Dict, AnyStr, Callable],
                 kernel_constraint: Union[None, Dict, AnyStr, Callable],
                 bias_constraint: Union[None, Dict, AnyStr, Callable],
                 **kwargs):
        assert rank in [1, 2, 3]
        assert basic_block_count > 0

        if "model_depth" in kwargs:
            model_depth = kwargs.pop("model_depth")
        else:
            model_depth = None

        super(ResBlockND, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.basic_block_count = basic_block_count
        self.basic_block_depth = basic_block_depth

        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, "kernel_size")
        self.strides = conv_utils.normalize_tuple(strides, rank, "strides")

        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, "dilation_rate")
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        if kernel_initializer == "from_model_depth":
            assert model_depth is not None
            self.kernel_initializer = VarianceScaling(scale=1.0 / np.sqrt(model_depth),
                                                      mode="fan_in",
                                                      distribution="normal")
        else:
            self.kernel_initializer = initializers.get(kernel_initializer)

        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.basic_blocks = []

        self.input_spec = InputSpec(ndim=self.rank + 2)

    def init_layers(self):
        for i in range(self.basic_block_count):
            strides = self.strides if (i == 0) else 1
            basic_block = ResBasicBlockND(rank=self.rank,
                                          filters=self.filters,
                                          depth=self.basic_block_depth,
                                          kernel_size=self.kernel_size,
                                          strides=strides,
                                          data_format=self.data_format,
                                          dilation_rate=self.dilation_rate,
                                          activation=self.activation,
                                          use_bias=self.use_bias,
                                          kernel_initializer=self.kernel_initializer,
                                          bias_initializer=self.bias_initializer,
                                          kernel_regularizer=self.kernel_regularizer,
                                          bias_regularizer=self.bias_regularizer,
                                          activity_regularizer=self.activity_regularizer,
                                          kernel_constraint=self.kernel_constraint,
                                          bias_constraint=self.bias_constraint)
            self.basic_blocks.append(basic_block)

    def build(self, input_shape):
        self.init_layers()

        intermediate_shape = input_shape

        with K.name_scope("residual_block_weights"):
            for i in range(self.basic_block_count):
                basic_block = self.basic_blocks[i]
                self.build_sub_layer(basic_block, intermediate_shape)
                intermediate_shape = basic_block.compute_output_shape(intermediate_shape)

        self.input_spec = InputSpec(ndim=self.rank + 2, axes={self.channel_axis: input_shape[self.channel_axis]})
        super(ResBlockND, self).build(input_shape)

    def call(self, inputs, **kwargs):
        layer = inputs
        with K.name_scope("residual_block"):
            for basic_block in self.basic_blocks:
                layer = basic_block(layer)
        return layer

    def compute_output_shape(self, input_shape):
        def get_new_space(space):
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding="same",
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tuple(new_space)

        if self.channels_first:
            return (input_shape[0], self.filters) + get_new_space(input_shape[2:])
        else:
            return (input_shape[0],) + get_new_space(input_shape[1:-1]) + (self.filters,)

    @property
    def channel_axis(self):
        if self.data_format == "channels_first":
            return 1
        else:
            return -1

    @property
    def channels_first(self):
        return self.data_format == "channels_first"

    def get_config(self):
        config = \
            {
                "rank": self.rank,
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "basic_block_count": self.basic_block_count,
                "basic_block_depth": self.basic_block_depth,
                "strides": self.strides,
                "padding": "same",
                "data_format": self.data_format,
                "dilation_rate": self.dilation_rate,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "activity_regularizer": regularizers.serialize(self.activity_regularizer),
                "kernel_constraint": constraints.serialize(self.kernel_constraint),
                "bias_constraint": constraints.serialize(self.bias_constraint)
            }
        base_config = super(ResBlockND, self).get_config()
        return {**base_config, **config}


class ResBlock1D(ResBlockND):
    def __init__(self,
                 filters,
                 basic_block_count: int,
                 basic_block_depth=2,
                 kernel_size=3,
                 strides=1,
                 data_format=None,
                 dilation_rate=1,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer="he_normal",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ResBlock1D, self).__init__(rank=1,
                                         filters=filters, basic_block_count=basic_block_count,
                                         basic_block_depth=basic_block_depth, kernel_size=kernel_size,
                                         strides=strides, data_format=data_format,
                                         dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                         activity_regularizer=activity_regularizer,
                                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                         **kwargs)

    def get_config(self):
        config = super(ResBlock1D, self).get_config()
        config.pop("rank")
        return config


class ResBlock2D(ResBlockND):
    def __init__(self,
                 filters,
                 basic_block_count: int,
                 basic_block_depth=2,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 data_format=None,
                 dilation_rate=1,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer="he_normal",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ResBlock2D, self).__init__(rank=2,
                                         filters=filters, basic_block_count=basic_block_count,
                                         basic_block_depth=basic_block_depth, kernel_size=kernel_size,
                                         strides=strides, data_format=data_format,
                                         dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                         activity_regularizer=activity_regularizer,
                                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                         **kwargs)

    def get_config(self):
        config = super(ResBlock2D, self).get_config()
        config.pop("rank")
        return config


class ResBlock3D(ResBlockND):
    def __init__(self,
                 filters,
                 basic_block_count: int,
                 basic_block_depth=2,
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 data_format=None,
                 dilation_rate=1,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer="he_normal",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ResBlock3D, self).__init__(rank=3,
                                         filters=filters, basic_block_count=basic_block_count,
                                         basic_block_depth=basic_block_depth, kernel_size=kernel_size,
                                         strides=strides, data_format=data_format,
                                         dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                         activity_regularizer=activity_regularizer,
                                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                         **kwargs)

    def get_config(self):
        config = super(ResBlock3D, self).get_config()
        config.pop("rank")
        return config


class ResBlockNDTranspose(ResBlockND):
    def init_layers(self):
        for i in range(self.basic_block_count):
            strides = self.strides if (i == 0) else 1
            basic_block = ResBasicBlockNDTranspose(rank=self.rank,
                                                   filters=self.filters,
                                                   depth=self.basic_block_depth,
                                                   kernel_size=self.kernel_size,
                                                   strides=strides,
                                                   data_format=self.data_format,
                                                   dilation_rate=self.dilation_rate,
                                                   activation=self.activation,
                                                   use_bias=self.use_bias,
                                                   kernel_initializer=self.kernel_initializer,
                                                   bias_initializer=self.bias_initializer,
                                                   kernel_regularizer=self.kernel_regularizer,
                                                   bias_regularizer=self.bias_regularizer,
                                                   activity_regularizer=self.activity_regularizer,
                                                   kernel_constraint=self.kernel_constraint,
                                                   bias_constraint=self.bias_constraint)
            self.basic_blocks.append(basic_block)

    def compute_output_shape(self, input_shape):
        def get_new_space(space):
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.deconv_length(
                    space[i],
                    self.strides[i],
                    self.kernel_size[i],
                    padding="same",
                    output_padding=None)
                new_space.append(new_dim)
            return tuple(new_space)

        if self.channels_first:
            return (input_shape[0], self.filters) + get_new_space(input_shape[2:])
        else:
            return (input_shape[0],) + get_new_space(input_shape[1:-1]) + (self.filters,)


class ResBlock2DTranspose(ResBlockNDTranspose):
    def __init__(self,
                 filters,
                 basic_block_count: int,
                 basic_block_depth=2,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 data_format=None,
                 dilation_rate=1,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer="he_normal",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ResBlock2DTranspose, self).__init__(rank=2,
                                                  filters=filters, basic_block_count=basic_block_count,
                                                  basic_block_depth=basic_block_depth, kernel_size=kernel_size,
                                                  strides=strides, data_format=data_format,
                                                  dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                                                  kernel_initializer=kernel_initializer,
                                                  bias_initializer=bias_initializer,
                                                  kernel_regularizer=kernel_regularizer,
                                                  bias_regularizer=bias_regularizer,
                                                  activity_regularizer=activity_regularizer,
                                                  kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                                  **kwargs)

    def get_config(self):
        config = super(ResBlock2DTranspose, self).get_config()
        config.pop("rank")
        return config


class ResBlock3DTranspose(ResBlockNDTranspose):
    def __init__(self,
                 filters,
                 basic_block_count: int,
                 basic_block_depth=2,
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 data_format=None,
                 dilation_rate=1,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer="he_normal",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ResBlock3DTranspose, self).__init__(rank=3,
                                                  filters=filters, basic_block_count=basic_block_count,
                                                  basic_block_depth=basic_block_depth, kernel_size=kernel_size,
                                                  strides=strides, data_format=data_format,
                                                  dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                                                  kernel_initializer=kernel_initializer,
                                                  bias_initializer=bias_initializer,
                                                  kernel_regularizer=kernel_regularizer,
                                                  bias_regularizer=bias_regularizer,
                                                  activity_regularizer=activity_regularizer,
                                                  kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                                  **kwargs)

    def get_config(self):
        config = super(ResBlock3DTranspose, self).get_config()
        config.pop("rank")
        return config

# endregion
