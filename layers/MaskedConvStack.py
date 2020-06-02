import tensorflow as tf
from tensorflow.python.keras import activations, initializers, regularizers, constraints
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn_ops
import numpy as np
from typing import Union, Tuple, Optional

from CustomKerasLayers.layers.MaskedConv import get_kernel_mask


def get_stack_kernel_mask(kernel_size, stack_size, mask_center: bool):
    base_mask = get_kernel_mask(kernel_size=kernel_size, mask_center=True)

    base_mask = np.expand_dims(base_mask, axis=-3)
    base_mask = np.tile(base_mask, [1] * len(kernel_size) + [stack_size, 1, 1])

    mask = np.empty([stack_size, *kernel_size, stack_size, 1, 1])
    offset = 0 if mask_center else 1
    center_index = tuple(dim // 2 for dim in kernel_size)
    if not mask_center:
        base_mask[(*center_index, 0)] = 1
    mask[0] = base_mask

    for i in range(stack_size - 1):
        base_mask[(*center_index, i + offset)] = 1
        mask[i + 1] = base_mask

    return mask


def get_stack_padding(kernel_size):
    base = ((dim // 2, dim // 2) for dim in kernel_size)
    return ((0, 0), *base, (0, 0), (0, 0))


class MaskedConvStack(Layer):
    """
    Stack of Masked Convolution.
    """

    def __init__(self,
                 rank: int,
                 filters: int,
                 mask_center: bool,
                 kernel_size: Union[int, Tuple[int], Tuple[int, int]],
                 activation: Union[str, Layer],
                 use_bias: bool,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MaskedConvStack, self).__init__(**kwargs)

        self.rank = rank
        self.filters = filters
        self.mask_center = mask_center
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank - 1, "kernel_size")
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.kernel: Optional[tf.Variable] = None
        self.bias: Optional[tf.Variable] = None
        self.kernel_mask: Optional[tf.Tensor] = None
        self._convolution_op: Optional[nn_ops.Convolution] = None

    def build(self, input_shape):
        stack_size, input_dim = input_shape[-2:]
        if stack_size is None or input_dim is None:
            raise ValueError("The two last dimensions of the inputs should be defined. Found `None`.")

        kernel_shape = (stack_size, *self.kernel_size, stack_size, input_dim, self.filters)
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)

        if self.use_bias:
            bias_shape = (stack_size, self.filters)
            self.bias = self.add_weight(
                name="bias",
                shape=bias_shape,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

        kernel_mask = get_stack_kernel_mask(self.kernel_size, stack_size, self.mask_center)
        self.kernel_mask = tf.constant(kernel_mask, dtype=tf.float32, name="kernel_mask")

        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={-1: input_dim, -2: stack_size})

        strides = conv_utils.normalize_tuple(1, self.rank, "strides")
        dilation_rate = conv_utils.normalize_tuple(1, self.rank, "dilation_rate")
        self._convolution_op = nn_ops.Convolution(input_shape=input_shape,
                                                  filter_shape=tf.TensorShape(kernel_shape[1:]),
                                                  dilation_rate=dilation_rate,
                                                  strides=strides,
                                                  padding=get_stack_padding(self.kernel_size),
                                                  data_format=conv_utils.convert_data_format("channels_last",
                                                                                             self.rank + 2))

    def call(self, inputs, **kwargs):
        def loop_cond(i, _):
            return i < self.stack_size

        def loop_body(i, array: tf.TensorArray):
            step_output = self.single_convolution(inputs, i)
            array = array.write(i, step_output)
            i += 1
            return i, array

        loop_vars = [
            tf.constant(0),
            tf.TensorArray(dtype=self.dtype,
                           size=self.stack_size,
                           name="convolutions_outputs_array")
        ]

        _, output_array = tf.while_loop(cond=loop_cond,
                                        body=loop_body,
                                        loop_vars=loop_vars,
                                        parallel_iterations=self.stack_size)
        outputs: tf.Tensor = output_array.stack()
        outputs.set_shape((self.stack_size, *outputs.shape[1:]))

        outputs = tf.squeeze(outputs, axis=-2)
        perm = [*range(1, self.rank + 1), 0, self.rank + 1]
        outputs = tf.transpose(outputs, perm)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    @tf.function
    def single_convolution(self, inputs, index):
        outputs = self._convolution_op(inputs, self.kernel[index] * self.kernel_mask[index])

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias[index], data_format="NHWC")

        return outputs

    def get_config(self):
        return {
            "rank": self.rank,
            "filters": self.filters,
            "mask_center": self.mask_center,
            "kernel_size": self.kernel_size,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias
        }

    @property
    def stack_size(self) -> int:
        if self.kernel is None:
            raise AttributeError("This layer has not been built yet. `stack_size` is unknown.")
        return self.kernel.shape[0]


class MaskedConv2DStack(Layer):
    def __init__(self,
                 filters: int,
                 mask_center: bool,
                 kernel_size: Union[int, Tuple[int]],
                 activation: Union[str, Layer],
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MaskedConv2DStack, self).__init__(rank=2,
                                                filters=filters,
                                                mask_center=mask_center,
                                                kernel_size=kernel_size,
                                                activation=activation,
                                                use_bias=use_bias,
                                                kernel_initializer=kernel_initializer,
                                                bias_initializer=bias_initializer,
                                                kernel_regularizer=kernel_regularizer,
                                                bias_regularizer=bias_regularizer,
                                                kernel_constraint=kernel_constraint,
                                                bias_constraint=bias_constraint,
                                                **kwargs)


class MaskedConv3DStack(Layer):
    def __init__(self,
                 filters: int,
                 mask_center: bool,
                 kernel_size: Union[int, Tuple[int, int]],
                 activation: Union[str, Layer],
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MaskedConv3DStack, self).__init__(rank=3,
                                                filters=filters,
                                                mask_center=mask_center,
                                                kernel_size=kernel_size,
                                                activation=activation,
                                                use_bias=use_bias,
                                                kernel_initializer=kernel_initializer,
                                                bias_initializer=bias_initializer,
                                                kernel_regularizer=kernel_regularizer,
                                                bias_regularizer=bias_regularizer,
                                                kernel_constraint=kernel_constraint,
                                                bias_constraint=bias_constraint,
                                                **kwargs)
