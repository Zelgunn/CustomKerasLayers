import tensorflow as tf
from tensorflow_core.python.keras.layers.convolutional import Conv
import numpy as np
from typing import List


def get_kernel_mask(kernel_size: List[int], mask_center: bool) -> np.ndarray:
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if any([(dim % 2) == 0 for dim in kernel_size]):
        raise ValueError("Even numbers are not supported yet. Received `{}`.".format(kernel_size))

    total_count = np.prod(kernel_size)
    one_count = total_count // 2

    if not mask_center:
        one_count += 1

    mask = np.concatenate([np.ones([one_count]), np.zeros([total_count - one_count])], axis=0)
    mask = np.reshape(mask, [*kernel_size, 1, 1])

    return mask


class MaskedConv(Conv):
    """
    Same layer as Conv, except that the kernel is masked so that a "pixel" only sees previous "pixels".
    See "Conditional Image Generation with PixelCNN Decoders".

    :param rank: Integer, the rank of the convolution, e.g. "2" for 2D convolution.
    :param filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
    :param kernel_size: An integer or tuple/list of n integers, specifying the length of the convolution window.
    :param mask_center: A boolean, if False then the center of the kernel is not masked.
    :param kernel_mask: (Optional) If specified, replaces the default mask.
    :param strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
    :param padding: One of `"valid"`,  `"same"` (case-insensitive).
    :param data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, ...)`.
    :param dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
    :param activation: Activation function. Set it to None to maintain a linear activation.
    :param use_bias: Boolean, whether the layer uses a bias.
    :param kernel_initializer: An initializer for the convolution kernel.
    :param bias_initializer: An initializer for the bias vector. If None, the default initializer will be used.
    :param kernel_regularizer: Optional regularizer for the convolution kernel.
    :param bias_regularizer: Optional regularizer for the bias vector.
    :param activity_regularizer: Optional regularizer function for the output.
    :param kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    :param bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.

    Input shape:
        (rank+2)D tensor with shape:
            `(samples, channels, *conv_dims)` if data_format='channels_first'
        or (rank+2)D tensor with shape:
            `(samples, *conv_dims, channels)` if data_format='channels_last'.

    Output shape:
        (rank+2)D tensor with shape:
            `(samples, filters, *new_conv_dims)` if data_format='channels_first'
        or (rank+2)D tensor with shape:
            `(samples, *new_conv_dims, filters)` if data_format='channels_last'.
            `new_conv_dims` values might have changed due to padding.
    """

    def __init__(self,
                 rank: int,
                 filters,
                 kernel_size,
                 mask_center=True,
                 kernel_mask=None,
                 strides=(1, 1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MaskedConv, self).__init__(
            rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
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

        self.mask_center = mask_center

        if kernel_mask is None:
            kernel_mask = get_kernel_mask(self.kernel_size, self.mask_center)

        if not isinstance(kernel_mask, tf.Tensor):
            kernel_mask = tf.constant(kernel_mask,
                                      name="kernel_mask",
                                      dtype=tf.float32)

        self.kernel_mask = kernel_mask

    def call(self, inputs):
        outputs = self._convolution_op(inputs, self.kernel * self.kernel_mask)

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    bias = tf.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    # region Keras utility
    def get_config(self):
        base_config = super(MaskedConv, self).get_config()
        return {
            **base_config,
            "mask_center": self.mask_center,
        }

    # endregion


class MaskedConv1D(MaskedConv):
    """
    Same layer as Conv1D, except that the kernel is masked so that a "pixel" only sees previous "pixels".
    See "Conditional Image Generation with PixelCNN Decoders".

    :param filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
    :param kernel_size: An integer, specifying the length of the convolution window.
    :param mask_center: A boolean, if False then the center of the kernel is not masked.
    :param kernel_mask: (Optional) If specified, replaces the default mask.
    :param strides: An integer, specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying any `dilation_rate` value != 1.
    :param padding: One of `"valid"`,  `"same"` and `"causal"` (case-insensitive).
    :param data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch, ..., channels)`
        while `channels_first` corresponds to inputs with shape `(batch, channels, ...)`.
    :param dilation_rate: An integer, specifying the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
    :param activation: Activation function. Set it to None to maintain a linear activation.
    :param use_bias: Boolean, whether the layer uses a bias.
    :param kernel_initializer: An initializer for the convolution kernel.
    :param bias_initializer: An initializer for the bias vector. If None, the default initializer will be used.
    :param kernel_regularizer: Optional regularizer for the convolution kernel.
    :param bias_regularizer: Optional regularizer for the bias vector.
    :param activity_regularizer: Optional regularizer function for the output.
    :param kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    :param bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.

    Input shape:
        3D tensor with shape:
            `(samples, channels, conv_dim)` if data_format='channels_first'
        or 3D tensor with shape:
            `(samples, conv_dim, channels)` if data_format='channels_last'.

    Output shape:
        3D tensor with shape:
            `(samples, filters, new_conv_dim)` if data_format='channels_first'
        or 3D tensor with shape:
            `(samples, new_conv_dim, filters)` if data_format='channels_last'.
            `new_conv_dims` values might have changed due to padding.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 mask_center=True,
                 kernel_mask=None,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MaskedConv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            mask_center=mask_center,
            kernel_mask=kernel_mask,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
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

    def call(self, inputs):
        if self.padding == "causal":
            inputs = tf.pad(inputs, self._compute_causal_padding())
        return super(MaskedConv1D, self).call(inputs)


class MaskedConv2D(MaskedConv):
    """
    Same layer as Conv2D, except that the kernel is masked so that a "pixel" only sees previous "pixels".
    See "Conditional Image Generation with PixelCNN Decoders".

    :param filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
    :param kernel_size: An integer or tuple/list of n integers, specifying the length of the convolution window.
    :param mask_center: A boolean, if False then the center of the kernel is not masked.
    :param kernel_mask: (Optional) If specified, replaces the default mask.
    :param strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
    :param padding: One of `"valid"`,  `"same"` (case-insensitive).
    :param data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, ...)`.
    :param dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
    :param activation: Activation function. Set it to None to maintain a linear activation.
    :param use_bias: Boolean, whether the layer uses a bias.
    :param kernel_initializer: An initializer for the convolution kernel.
    :param bias_initializer: An initializer for the bias vector. If None, the default initializer will be used.
    :param kernel_regularizer: Optional regularizer for the convolution kernel.
    :param bias_regularizer: Optional regularizer for the bias vector.
    :param activity_regularizer: Optional regularizer function for the output.
    :param kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    :param bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.

    Input shape:
        4D tensor with shape:
            `(samples, channels, conv_dim1, conv_dim_2)` if data_format='channels_first'
        or 4D tensor with shape:
            `(samples, conv_dim1, conv_dim2, channels)` if data_format='channels_last'.

    Output shape:
        4D tensor with shape:
            `(samples, filters, new_conv_dim1, new_conv_dim2)` if data_format='channels_first'
        or 4D tensor with shape:
            `(samples, new_conv_dim1, new_conv_dim2, filters)` if data_format='channels_last'.
            `new_conv_dims` values might have changed due to padding.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 mask_center=True,
                 kernel_mask=None,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MaskedConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            mask_center=mask_center,
            kernel_mask=kernel_mask,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
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


class MaskedConv3D(MaskedConv):
    """
    Same layer as Conv2D, except that the kernel is masked so that a "pixel" only sees previous "pixels".
    See "Conditional Image Generation with PixelCNN Decoders".

    :param filters: Integer, the dimensionality of the output space
    (i.e. the number of filters in the convolution).
    :param kernel_size: An integer or tuple/list of n integers, specifying the length of the convolution window.
    :param mask_center: A boolean, if False then the center of the kernel is not masked.
    :param kernel_mask: (Optional) If specified, replaces the default mask.
    :param strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
    :param padding: One of `"valid"`,  `"same"` (case-insensitive).
    :param data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, ...)`.
    :param dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
    :param activation: Activation function. Set it to None to maintain a linear activation.
    :param use_bias: Boolean, whether the layer uses a bias.
    :param kernel_initializer: An initializer for the convolution kernel.
    :param bias_initializer: An initializer for the bias vector. If None, the default initializer will be used.
    :param kernel_regularizer: Optional regularizer for the convolution kernel.
    :param bias_regularizer: Optional regularizer for the bias vector.
    :param activity_regularizer: Optional regularizer function for the output.
    :param kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    :param bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.

    Input shape:
        4D tensor with shape:
            `(samples, channels, conv_dim1, conv_dim2, conv_dim3)` if data_format='channels_first'
        or 4D tensor with shape:
            `(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if data_format='channels_last'.

    Output shape:
        5D tensor with shape:
            `(samples, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)` if data_format='channels_first'
        or 5D tensor with shape:
            `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)` if data_format='channels_last'.
            `new_conv_dims` values might have changed due to padding.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 mask_center=True,
                 kernel_mask=None,
                 strides=(1, 1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MaskedConv3D, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            mask_center=mask_center,
            kernel_mask=kernel_mask,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
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


if __name__ == "__main__":
    layer = MaskedConv3D(filters=1,
                         kernel_size=(3, 5, 5),
                         mask_center=True,
                         padding="same",
                         use_bias=False,
                         kernel_initializer="ones")

    x = tf.ones([1, 5, 9, 9, 1])
    y = layer(x)

    # print(tf.squeeze(y))
    print(tf.squeeze(layer.kernel_mask))
