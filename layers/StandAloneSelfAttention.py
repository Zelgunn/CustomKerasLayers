import tensorflow as tf
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras import activations, initializers, regularizers, constraints
from tensorflow.python.keras.utils import conv_utils
from typing import Union, Optional, Tuple, List, AnyStr, Callable, Dict

from CustomKerasLayers.utility_layers.Unfold import Unfold


class StandAloneSelfAttention(Layer):
    def __init__(self,
                 rank: int,
                 head_size: int,
                 head_count: int,
                 kernel_size: Union[int, Tuple, List],
                 strides: Union[int, Tuple, List],
                 # data_format: Optional[AnyStr],
                 dilation_rate: Union[int, Tuple, List],
                 activation: Optional[Union[AnyStr, Callable]],
                 use_bias: bool,
                 kernel_initializer: Optional[Union[Dict, AnyStr, Callable]],
                 bias_initializer: Optional[Union[Dict, AnyStr, Callable]],
                 embeddings_initializer: Optional[Union[Dict, AnyStr, Callable]],
                 kernel_regularizer: Optional[Union[Dict, AnyStr, Callable]],
                 bias_regularizer: Optional[Union[Dict, AnyStr, Callable]],
                 activity_regularizer: Optional[Union[Dict, AnyStr, Callable]],
                 kernel_constraint: Optional[Union[Dict, AnyStr, Callable]],
                 bias_constraint: Optional[Union[Dict, AnyStr, Callable]],
                 trainable=True,
                 name=None,
                 **kwargs):
        activity_regularizer = regularizers.get(activity_regularizer)
        super(StandAloneSelfAttention, self).__init__(trainable=trainable,
                                                      name=name,
                                                      activity_regularizer=activity_regularizer,
                                                      **kwargs)

        # region Utils (normalizing tuples, data format and getting initializers/regularizers/constraints)
        kernel_size = conv_utils.normalize_tuple(kernel_size, rank, "kernel_size")
        strides = conv_utils.normalize_tuple(strides, rank, "strides")
        # data_format = conv_utils.normalize_data_format(data_format)
        dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, "dilation_rate")
        activation = activations.get(activation)
        kernel_initializer = initializers.get(kernel_initializer)
        bias_initializer = initializers.get(bias_initializer)
        if embeddings_initializer == "random_normal":
            embeddings_initializer = initializers.RandomNormal(stddev=1.0)
        embeddings_initializer = initializers.get(embeddings_initializer)
        kernel_regularizer = regularizers.get(kernel_regularizer)
        bias_regularizer = regularizers.get(bias_regularizer)
        kernel_constraint = constraints.get(kernel_constraint)
        bias_constraint = constraints.get(bias_constraint)
        # endregion

        # region Base attributes
        self.rank = rank
        self.head_size = head_size
        self.head_count = head_count
        self.kernel_size = kernel_size
        self.strides = strides
        # self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.embeddings_initializer = embeddings_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        # endregion

        # region Queries/Keys/Values conv layers
        common_parameters = {
            "rank": self.rank,
            "filters": self.filters,
            "kernel_size": 1,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "activity_regularizer": self.activity_regularizer,
            "kernel_constraint": self.kernel_constraint,
            "bias_constraint": self.bias_constraint,
        }
        self.queries_layer = Conv(name="Conv_Queries{}".format(self.name), **common_parameters)
        self.keys_layer = Conv(name="Conv_Keys{}".format(self.name), **common_parameters)
        self.values_layer = Conv(name="Conv_Values{}".format(self.name), **common_parameters)
        # endregion

        # region Queries/Keys/Values unfold layers
        self.queries_unfold = Unfold(kernel_size=1, strides=strides, name="Unfold_Queries_{}".format(self.name))
        self.keys_unfold = Unfold(kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate, padding="SAME",
                                  name="Unfold_Keys_{}".format(self.name))
        self.values_unfold = Unfold(kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate,
                                    padding="SAME", name="Unfold_Values_{}".format(self.name))
        # endregion

        # region Time/Height/Width embeddings
        conv_embeddings = []
        for i in range(rank):
            dim_embeddings_size = self.filters // rank
            if i == 0:
                dim_embeddings_size += self.filters % rank

            dim_embeddings_shape = (dim_embeddings_size, *[1] * i, kernel_size[i], *[1] * (rank - i - 1))
            dim_embeddings = self.add_weight(name="dim_{}_embeddings".format(i + 1),
                                             shape=dim_embeddings_shape,
                                             dtype=tf.float32,
                                             initializer=self.embeddings_initializer)
            conv_embeddings.append(dim_embeddings)
        self.conv_embeddings = conv_embeddings
        # endregion

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)

    def call(self, inputs, **kwargs):
        # region Inputs shape
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        # endregion

        # region Queries
        queries = self.queries_layer(inputs)
        # queries shape : [batch_size, *input_dims, filters]
        queries = self.queries_unfold(queries)
        # queries shape : [batch_size, *output_dims, filters]
        output_dims = tf.shape(queries)[1:-1]
        queries = tf.reshape(queries, [batch_size, -1, self.head_count, self.head_size, 1])
        # queries shape : [batch_size, patch_count, head_count, head_size, 1]
        queries = tf.transpose(queries, perm=[0, 1, 2, 4, 3])
        # queries shape : [batch_size, patch_count, head_count, 1, head_size]
        # endregion

        # region Keys
        keys = self.keys_layer(inputs)
        # keys shape : [batch_size, *input_dims, filters]
        keys = self.keys_unfold(keys)
        # keys shape : [batch_size, *output_dims, filters]
        keys = tf.reshape(keys, [batch_size, -1, self.head_count, self.head_size, self.kernel_dim])
        # keys shape : [batch_size, patch_count, head_count, head_size, kernel_dim]
        keys = self.add_embeddings(keys)
        # endregion

        # region Attention
        attention = tf.nn.softmax(queries @ keys, axis=-1)
        # attention shape : [batch_size, patch_count, head_count, 1, kernel_dim]
        attention = tf.transpose(attention, perm=[0, 1, 2, 4, 3])
        # attention shape : [batch_size, patch_count, head_count, kernel_dim, 1]
        # endregion

        # region Values
        values = self.values_layer(inputs)
        # values shape : [batch_size, *input_dims, filters]
        values = self.values_unfold(values)
        # values shape : [batch_size, *output_dims, filters]
        values = tf.reshape(values, [batch_size, -1, self.head_count, self.head_size, self.kernel_dim])
        # values shape : [batch_size, patch_count, head_count, head_size, kernel_dim]
        # endregion

        # region Outputs
        outputs = values @ attention
        # outputs shape : [batch_size, patch_count, head_count, head_size, 1]
        output_shape = tf.concat([[batch_size], output_dims, [self.filters]], axis=0)
        outputs = tf.reshape(outputs, output_shape)
        # outputs shape : [batch_size, *outputs_dims, self.filters]
        outputs.set_shape(self.compute_output_shape(inputs.shape))
        outputs = self.activation(outputs)
        # endregion

        return outputs

    def compute_output_shape(self, input_shape):
        batch_size, *dims, _ = input_shape
        output_dims = []
        for i in range(self.rank):
            if dims[i] is None:
                output_dims.append(None)
            else:
                output_dims.append(dims[i] // self.strides[i])
        return tf.TensorShape((batch_size, *output_dims, self.filters))

    @tf.function
    def add_embeddings(self, keys: tf.Tensor) -> tf.Tensor:
        conv_embeddings = []
        for i in range(self.rank):
            dim_embeddings_tile = (1, *self.kernel_size[:i], 1, *self.kernel_size[i + 1:])
            dim_embeddings = tf.tile(self.conv_embeddings[i], dim_embeddings_tile)
            conv_embeddings.append(dim_embeddings)

        conv_embeddings = tf.concat(conv_embeddings, axis=0, name="conv_embeddings")
        conv_embeddings = tf.reshape(conv_embeddings, [1, 1, self.head_count, self.head_size, self.kernel_dim])
        return keys + conv_embeddings

    @property
    def filters(self) -> int:
        return self.head_size * self.head_count

    @property
    def kernel_dim(self) -> int:
        dim = 1
        for value in self.kernel_size:
            dim *= value
        return dim

    def get_config(self):
        base_config = super(StandAloneSelfAttention, self).get_config()
        config = {
            **base_config,
            "rank": self.rank,
            "head_size": self.head_size,
            "head_count": self.head_count,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            # "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "embeddings_initializer": self.embeddings_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "kernel_constraint": self.kernel_constraint,
            "bias_constraint": self.bias_constraint,
            "seed": self.seed,
        }
        return config


# region Subclasses (1D/2D/3D)

class StandAloneSelfAttention1D(StandAloneSelfAttention):
    def __init__(self,
                 head_size: int,
                 head_count: int,
                 kernel_size: Union[int, Tuple, List] = 3,
                 strides: Union[int, Tuple, List] = 1,
                 dilation_rate: Union[int, Tuple, List] = 1,
                 activation: Optional[Union[AnyStr, Callable]] = "linear",
                 use_bias: bool = True,
                 kernel_initializer: Optional[Union[Dict, AnyStr, Callable]] = "he_normal",
                 bias_initializer: Optional[Union[Dict, AnyStr, Callable]] = "zeros",
                 embeddings_initializer: Optional[Union[Dict, AnyStr, Callable]] = "random_normal",
                 kernel_regularizer: Optional[Union[Dict, AnyStr, Callable]] = None,
                 bias_regularizer: Optional[Union[Dict, AnyStr, Callable]] = None,
                 activity_regularizer: Optional[Union[Dict, AnyStr, Callable]] = None,
                 kernel_constraint: Optional[Union[Dict, AnyStr, Callable]] = None,
                 bias_constraint: Optional[Union[Dict, AnyStr, Callable]] = None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(StandAloneSelfAttention1D, self).__init__(rank=1, head_size=head_size, head_count=head_count,
                                                        kernel_size=kernel_size, strides=strides,
                                                        dilation_rate=dilation_rate, activation=activation,
                                                        use_bias=use_bias,
                                                        kernel_initializer=kernel_initializer,
                                                        bias_initializer=bias_initializer,
                                                        embeddings_initializer=embeddings_initializer,
                                                        kernel_regularizer=kernel_regularizer,
                                                        bias_regularizer=bias_regularizer,
                                                        activity_regularizer=activity_regularizer,
                                                        kernel_constraint=kernel_constraint,
                                                        bias_constraint=bias_constraint,
                                                        trainable=trainable,
                                                        name=name,
                                                        **kwargs
                                                        )


class StandAloneSelfAttention2D(StandAloneSelfAttention):
    def __init__(self,
                 head_size: int,
                 head_count: int,
                 kernel_size: Union[int, Tuple, List] = 3,
                 strides: Union[int, Tuple, List] = 1,
                 dilation_rate: Union[int, Tuple, List] = 1,
                 activation: Optional[Union[AnyStr, Callable]] = "linear",
                 use_bias: bool = True,
                 kernel_initializer: Optional[Union[Dict, AnyStr, Callable]] = "he_normal",
                 bias_initializer: Optional[Union[Dict, AnyStr, Callable]] = "zeros",
                 embeddings_initializer: Optional[Union[Dict, AnyStr, Callable]] = "random_normal",
                 kernel_regularizer: Optional[Union[Dict, AnyStr, Callable]] = None,
                 bias_regularizer: Optional[Union[Dict, AnyStr, Callable]] = None,
                 activity_regularizer: Optional[Union[Dict, AnyStr, Callable]] = None,
                 kernel_constraint: Optional[Union[Dict, AnyStr, Callable]] = None,
                 bias_constraint: Optional[Union[Dict, AnyStr, Callable]] = None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(StandAloneSelfAttention2D, self).__init__(rank=3, head_size=head_size, head_count=head_count,
                                                        kernel_size=kernel_size, strides=strides,
                                                        dilation_rate=dilation_rate, activation=activation,
                                                        use_bias=use_bias,
                                                        kernel_initializer=kernel_initializer,
                                                        bias_initializer=bias_initializer,
                                                        embeddings_initializer=embeddings_initializer,
                                                        kernel_regularizer=kernel_regularizer,
                                                        bias_regularizer=bias_regularizer,
                                                        activity_regularizer=activity_regularizer,
                                                        kernel_constraint=kernel_constraint,
                                                        bias_constraint=bias_constraint,
                                                        trainable=trainable,
                                                        name=name,
                                                        **kwargs
                                                        )


class StandAloneSelfAttention3D(StandAloneSelfAttention):
    def __init__(self,
                 head_size: int,
                 head_count: int,
                 kernel_size: Union[int, Tuple, List] = 3,
                 strides: Union[int, Tuple, List] = 1,
                 dilation_rate: Union[int, Tuple, List] = 1,
                 activation: Optional[Union[AnyStr, Callable]] = "linear",
                 use_bias: bool = True,
                 kernel_initializer: Optional[Union[Dict, AnyStr, Callable]] = "he_normal",
                 bias_initializer: Optional[Union[Dict, AnyStr, Callable]] = "zeros",
                 embeddings_initializer: Optional[Union[Dict, AnyStr, Callable]] = "random_normal",
                 kernel_regularizer: Optional[Union[Dict, AnyStr, Callable]] = None,
                 bias_regularizer: Optional[Union[Dict, AnyStr, Callable]] = None,
                 activity_regularizer: Optional[Union[Dict, AnyStr, Callable]] = None,
                 kernel_constraint: Optional[Union[Dict, AnyStr, Callable]] = None,
                 bias_constraint: Optional[Union[Dict, AnyStr, Callable]] = None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(StandAloneSelfAttention3D, self).__init__(rank=3, head_size=head_size, head_count=head_count,
                                                        kernel_size=kernel_size, strides=strides,
                                                        dilation_rate=dilation_rate, activation=activation,
                                                        use_bias=use_bias,
                                                        kernel_initializer=kernel_initializer,
                                                        bias_initializer=bias_initializer,
                                                        embeddings_initializer=embeddings_initializer,
                                                        kernel_regularizer=kernel_regularizer,
                                                        bias_regularizer=bias_regularizer,
                                                        activity_regularizer=activity_regularizer,
                                                        kernel_constraint=kernel_constraint,
                                                        bias_constraint=bias_constraint,
                                                        trainable=trainable,
                                                        name=name,
                                                        **kwargs
                                                        )

# endregion
