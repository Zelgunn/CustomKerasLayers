import tensorflow as tf
from tensorflow.python.keras.layers import Conv2DTranspose, UpSampling1D, InputSpec
from tensorflow.python.keras import activations, initializers, regularizers, constraints
from tensorflow.python.keras.utils import conv_utils


class Conv1DTranspose(Conv2DTranspose):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding="valid",
                 output_padding=None,
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if output_padding is not None:
            output_padding = (1, output_padding)
        else:
            output_padding = extract_singleton(output_padding)

        kernel_size = extract_singleton(kernel_size)
        strides = extract_singleton(strides)
        dilation_rate = extract_singleton(dilation_rate)

        super(Conv1DTranspose, self).__init__(
            filters=filters,
            kernel_size=(1, kernel_size),
            strides=(1, strides),
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=(1, dilation_rate),
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)

        self._input_dim = None

    def build(self, input_shape):
        input_shape = expand_input_shape(input_shape)
        super(Conv1DTranspose, self).build(input_shape)

        self._input_dim = int(input_shape[self.channel_axis + 1])

    def call(self, inputs):
        inputs = tf.expand_dims(inputs, axis=1)
        outputs = super(Conv1DTranspose, self).call(inputs)
        outputs = tf.squeeze(outputs, axis=1)

        return outputs

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 4:
            if input_shape[1] == 1:
                return super(Conv1DTranspose, self).compute_output_shape(input_shape)

        input_shape = expand_input_shape(input_shape)
        output_shape = super(Conv1DTranspose, self).compute_output_shape(input_shape)
        output_shape = squeeze_output_shape(output_shape)

        return output_shape

    def get_config(self):
        config = super(Conv1DTranspose, self).get_config()
        config["output_padding"] = self.output_padding
        return config

    @property
    def channel_axis(self):
        return 1 if self.data_format == "channels_first" else 2

    @property
    def input_spec(self):
        return InputSpec(ndim=3, axes={self.channel_axis: self._input_dim})

    @input_spec.setter
    def input_spec(self, value):
        pass


def extract_singleton(value):
    if not (isinstance(value, list) or isinstance(value, tuple)):
        return value

    if len(value) != 1:
        raise ValueError("Value is not a singleton : length = {}".len(value))

    return value[0]


def expand_input_shape(input_shape):
    if len(input_shape) != 3:
        raise ValueError("Rank of `input_shape` must be 3, got {}.".format(input_shape))

    batch_size, length, channels = input_shape
    input_shape = (batch_size, 1, length, channels)
    return tf.TensorShape(input_shape)


def squeeze_output_shape(output_shape):
    if len(output_shape) != 4:
        raise ValueError("Rank of `output_shape` must be 4, got {}.".format(output_shape))

    batch_size, _, length, channels = output_shape
    output_shape = tf.TensorShape([batch_size, length, channels])
    return tf.TensorShape(output_shape)
