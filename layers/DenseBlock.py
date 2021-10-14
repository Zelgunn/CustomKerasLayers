import tensorflow as tf
from tensorflow.python.keras.layers import InputSpec, Layer
from tensorflow.python.keras.layers import Conv1D, Conv2D, Conv3D
from tensorflow.python.keras.layers import BatchNormalization, concatenate
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras import activations, initializers, regularizers, constraints
from typing import List, Optional


class CompositeFunctionBlock(Layer):
    def __init__(self,
                 rank,
                 kernel_size,
                 filters,
                 use_bottleneck,
                 bottleneck_filters_multiplier,
                 use_batch_normalization,
                 data_format,
                 activation,
                 use_bias,
                 kernel_initializer,
                 bias_initializer,
                 kernel_regularizer,
                 bias_regularizer,
                 activity_regularizer,
                 kernel_constraint,
                 bias_constraint,
                 **kwargs
                 ):
        self.rank = rank
        conv_layer_type = Conv1D if rank == 1 else Conv2D if rank == 2 else Conv3D
        self.filters = filters
        self.channel_axis = -1 if data_format == "channels_last" else 1
        self.activation = activation
        self.use_bottleneck = use_bottleneck
        self.use_batch_normalization = use_batch_normalization

        # region Main layers initialization
        self.batch_normalization_layer = BatchNormalization() if use_batch_normalization else None
        self.conv_layer = conv_layer_type(filters, kernel_size, padding="same",
                                          data_format=data_format,
                                          use_bias=use_bias, kernel_initializer=kernel_initializer,
                                          bias_initializer=bias_initializer,
                                          kernel_regularizer=kernel_regularizer,
                                          bias_regularizer=bias_regularizer,
                                          activity_regularizer=activity_regularizer,
                                          kernel_constraint=kernel_constraint,
                                          bias_constraint=bias_constraint
                                          )

        # endregion

        # region Bottleneck layers initialization
        self.bottleneck_batch_normalization_layer = None
        self.bottleneck_conv_layer = None

        if use_bottleneck:
            if use_batch_normalization:
                self.bottleneck_batch_normalization_layer = BatchNormalization()

            bottleneck_filters = bottleneck_filters_multiplier * filters
            self.bottleneck_conv_layer = conv_layer_type(bottleneck_filters, kernel_size=1, padding="same",
                                                         data_format=data_format,
                                                         use_bias=use_bias,
                                                         kernel_initializer=kernel_initializer,
                                                         bias_initializer=bias_initializer,
                                                         kernel_regularizer=kernel_regularizer,
                                                         bias_regularizer=bias_regularizer,
                                                         activity_regularizer=activity_regularizer,
                                                         kernel_constraint=kernel_constraint,
                                                         bias_constraint=bias_constraint
                                                         )
        # endregion

        super(CompositeFunctionBlock, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        layer = inputs

        if self.use_bottleneck:
            if self.use_batch_normalization:
                layer = self.bottleneck_batch_normalization_layer(layer)

            if self.activation is not None:
                layer = self.activation(layer)

            layer = self.bottleneck_conv_layer(layer)

        if self.use_batch_normalization:
            layer = self.batch_normalization_layer(layer)

        if self.activation is not None:
            layer = self.activation(layer)

        layer = self.conv_layer(layer)

        return layer

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.channel_axis] = self.filters
        return tuple(output_shape)


class DenseBlockND(Layer):
    def __init__(self, rank,
                 kernel_size,
                 growth_rate,
                 depth,
                 output_filters=None,
                 use_bottleneck=True,
                 bottleneck_filters_multiplier=4,
                 use_batch_normalization=True,
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

        if rank not in [1, 2, 3]:
            raise ValueError("`rank` must be in [1, 2, 3]. Got {}".format(rank))

        super(DenseBlockND, self).__init__(**kwargs)

        self.rank = rank
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, "kernel_size")
        self.output_filters = output_filters
        self.growth_rate = growth_rate

        if use_bottleneck:
            if (depth % 2) != 0:
                raise ValueError("Depth must be a multiple of 2 when using bottlenecks. Got {}.".format(depth))

        self._depth = depth // 2 if use_bottleneck else depth
        self.use_bottleneck = use_bottleneck
        self.bottleneck_filters_multiplier = bottleneck_filters_multiplier

        self.use_batch_normalization = use_batch_normalization

        self.data_format = conv_utils.normalize_data_format(data_format)
        self.channel_axis = -1 if self.data_format == "channels_last" else 1

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.composite_function_blocks: Optional[List[CompositeFunctionBlock]] = None
        self.transition_layer = None

        self.input_spec = InputSpec(ndim=self.rank + 2)

    def init_layers(self):
        self.composite_function_blocks: List[CompositeFunctionBlock] = []
        for i in range(self._depth):
            composite_function_block = CompositeFunctionBlock(self.rank, self.kernel_size, self.growth_rate,
                                                              self.use_bottleneck, self.bottleneck_filters_multiplier,
                                                              self.use_batch_normalization, self.data_format,
                                                              self.activation, self.use_bias,
                                                              self.kernel_initializer, self.bias_initializer,
                                                              self.kernel_regularizer, self.bias_regularizer,
                                                              self.activity_regularizer,
                                                              self.kernel_constraint, self.bias_constraint)
            self.composite_function_blocks.append(composite_function_block)

        if self.output_filters is not None:
            conv_layer_type = Conv1D if self.rank == 1 else Conv2D if self.rank == 2 else Conv3D
            self.transition_layer = conv_layer_type(filters=self.output_filters, kernel_size=1,
                                                    use_bias=self.use_bias,
                                                    data_format=self.data_format,
                                                    kernel_initializer=self.kernel_initializer,
                                                    bias_initializer=self.bias_initializer,
                                                    kernel_regularizer=self.kernel_regularizer,
                                                    bias_regularizer=self.bias_regularizer,
                                                    activity_regularizer=self.activity_regularizer,
                                                    kernel_constraint=self.kernel_constraint,
                                                    bias_constraint=self.bias_constraint,
                                                    name="transition_layer")

    def build(self, input_shape):
        self.init_layers()

        input_dim = input_shape[self.channel_axis]
        self.input_spec = InputSpec(ndim=self.rank + 2, axes={self.channel_axis: input_dim})

        super(DenseBlockND, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs_list = [inputs]
        layer = inputs

        with tf.name_scope("dense_block"):
            for i in range(self._depth):
                layer = self.composite_function_blocks[i](layer)

                inputs_list.append(layer)
                layer = concatenate(inputs_list, axis=-1)

        if self.transition_layer is not None:
            layer = self.transition_layer(layer)

        return layer

    def compute_output_shape(self, input_shape):
        input_dim = input_shape[self.channel_axis]
        output_shape = list(input_shape)
        if self.output_filters is None:
            output_dim = input_dim + self._depth * self.growth_rate
            output_shape[self.channel_axis] = output_dim
        else:
            output_shape[self.channel_axis] = self.output_filters
        return tuple(output_shape)

    @property
    def depth(self):
        return self._depth * 2 if self.use_bottleneck else self._depth

    def get_config(self):
        config = \
            {
                "rank": self.rank,
                "kernel_size": self.kernel_size,
                "growth_rate": self.growth_rate,
                "output_filters": self.output_filters,
                "depth": self.depth,
                "use_bottleneck": self.use_bottleneck,
                "bottleneck_filters_multiplier": self.bottleneck_filters_multiplier,
                "use_batch_normalization": self.use_batch_normalization,
                "data_format": self.data_format,
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

        base_config = super(DenseBlockND, self).get_config()
        return {**base_config, **config}


class DenseBlock1D(DenseBlockND):
    def __init__(self,
                 kernel_size,
                 growth_rate,
                 depth,
                 output_filters=None,
                 use_bottleneck=True,
                 bottleneck_filters_multiplier=4,
                 use_batch_normalization=True,
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
        super(DenseBlock1D, self).__init__(rank=1,
                                           kernel_size=kernel_size,
                                           growth_rate=growth_rate,
                                           depth=depth,
                                           output_filters=output_filters,
                                           use_bottleneck=use_bottleneck,
                                           bottleneck_filters_multiplier=bottleneck_filters_multiplier,
                                           use_batch_normalization=use_batch_normalization,
                                           data_format=data_format,
                                           activation=activation,
                                           use_bias=use_bias,
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer,
                                           kernel_regularizer=kernel_regularizer,
                                           bias_regularizer=bias_regularizer,
                                           activity_regularizer=activity_regularizer,
                                           kernel_constraint=kernel_constraint,
                                           bias_constraint=bias_constraint,
                                           **kwargs)

    def get_config(self):
        config = super(DenseBlock1D, self).get_config()
        config.pop("rank")
        return config


class DenseBlock2D(DenseBlockND):
    def __init__(self,
                 kernel_size,
                 growth_rate,
                 depth,
                 output_filters=None,
                 use_bottleneck=True,
                 bottleneck_filters_multiplier=4,
                 use_batch_normalization=True,
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
        super(DenseBlock2D, self).__init__(rank=2,
                                           kernel_size=kernel_size,
                                           growth_rate=growth_rate,
                                           depth=depth,
                                           output_filters=output_filters,
                                           use_bottleneck=use_bottleneck,
                                           bottleneck_filters_multiplier=bottleneck_filters_multiplier,
                                           use_batch_normalization=use_batch_normalization,
                                           data_format=data_format,
                                           activation=activation,
                                           use_bias=use_bias,
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer,
                                           kernel_regularizer=kernel_regularizer,
                                           bias_regularizer=bias_regularizer,
                                           activity_regularizer=activity_regularizer,
                                           kernel_constraint=kernel_constraint,
                                           bias_constraint=bias_constraint,
                                           **kwargs)

    def get_config(self):
        config = super(DenseBlock2D, self).get_config()
        config.pop("rank")
        return config


class DenseBlock3D(DenseBlockND):
    def __init__(self,
                 kernel_size,
                 growth_rate,
                 depth,
                 output_filters=None,
                 use_bottleneck=True,
                 bottleneck_filters_multiplier=4,
                 use_batch_normalization=True,
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
        super(DenseBlock3D, self).__init__(rank=3,
                                           kernel_size=kernel_size,
                                           growth_rate=growth_rate,
                                           depth=depth,
                                           output_filters=output_filters,
                                           use_bottleneck=use_bottleneck,
                                           bottleneck_filters_multiplier=bottleneck_filters_multiplier,
                                           use_batch_normalization=use_batch_normalization,
                                           data_format=data_format,
                                           activation=activation,
                                           use_bias=use_bias,
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer,
                                           kernel_regularizer=kernel_regularizer,
                                           bias_regularizer=bias_regularizer,
                                           activity_regularizer=activity_regularizer,
                                           kernel_constraint=kernel_constraint,
                                           bias_constraint=bias_constraint,
                                           **kwargs)

    def get_config(self):
        config = super(DenseBlock3D, self).get_config()
        config.pop("rank")
        return config
