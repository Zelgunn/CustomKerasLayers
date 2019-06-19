import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import Model

import numpy as np
from typing import Tuple, Union


class SpatialTransformer(Layer):
    """Spatial Transformer class, inherits from Keras class `Layer`.

        You must provide a `localisation_net` which is a Model or a Layer that takes regular inputs and outputs a tensor
        with total output dimension 6 (without batch). If the dimension if different from 6, the Spatial Transformer
        will add a Dense layer reducing dimensionality to 6.

        The Spatial Transformer uses this tensor of size 6 as the affine transformation matrix for bilinear sampling.

        The outputs of the Spatial Transformer are the transformed input images.
        If `output_theta` is provided when calling the layer (and is true), then the Spatial Transformer
        will also output `theta`, the affine transformation matrix.

    """
    def __init__(self, localisation_net: Union[Model, Layer], output_size=None, data_format=None, **kwargs):
        super(SpatialTransformer, self).__init__(**kwargs)
        self.localisation_net: Model = localisation_net
        self.localisation_output = None
        self.data_format = data_format
        self.theta_kernel = None
        self.theta_bias = None
        self.output_size = output_size

    def build(self, input_shape):
        self.localisation_net.build(input_shape)

        localisation_output_shape = self.localisation_net.compute_output_shape(input_shape)
        localisation_output_shape_check = list(localisation_output_shape)
        localisation_output_shape_check.remove(None)
        assert np.all(np.greater(localisation_output_shape_check, 0)), \
            "Negative output shape from localisation net : {0}".format(localisation_output_shape)
        if len(localisation_output_shape) != 2:
            output_dim = np.prod(localisation_output_shape[1:])
        else:
            output_dim = localisation_output_shape[-1]

        if output_dim != 6:
            self.theta_kernel = self.add_weight(name="theta_kernel", shape=[output_dim, 6],
                                                initializer="he_normal")
            self.theta_bias = self.add_weight(name="theta_bias", shape=[6],
                                              initializer="zeros")

        super(SpatialTransformer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs)[0]
        theta = self.localisation_net(inputs)

        if self.theta_kernel is not None:
            with tf.name_scope("theta_regression"):
                if len(self.localisation_net.output_shape) != 2:
                    theta = tf.reshape(theta, [batch_size, -1])
                theta = theta @ self.theta_kernel + self.theta_bias
        theta = tf.reshape(theta, [batch_size, 2, 3])

        if self.channels_first:
            inputs = tf.transpose(inputs, [0, 2, 3, 1])

        outputs = spatial_transformation(inputs, theta, self.output_size)

        if self.channels_first:
            outputs = tf.transpose(inputs, [0, 3, 1, 2])

        if "output_theta" in kwargs and kwargs["output_theta"]:
            outputs = [outputs, theta]

        return outputs

    @property
    def channels_first(self):
        return self.data_format == "channels_first"

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4

        if self.output_size is None:
            return input_shape
        elif self.channels_first:
            return (*input_shape[:2], *self.output_size)
        else:
            return (input_shape[0], *self.output_size, input_shape[-1])

    def get_config(self):
        config = {
            "output_size": self.output_size,
            "data_format": self.data_format,
            "localisation_net_config": self.localisation_net.get_config()
        }
        base_config = super(SpatialTransformer, self).get_config()
        return {**config, **base_config}


def affine_grid(theta: tf.Tensor,
                size: Union[Tuple[int, int], tf.Tensor],
                name="affine_grid"
                ) -> tf.Tensor:
    """Computes the affine grid to perform bilinear sampling using the affine transformation matrix `theta`.

        Parameters:
            theta: A 2D or 3D tensor with shape [batch_size, 6] or [batch_size, 2, 3] and type `float32`.
            size: A 1D tensor with shape [2] and type `int32` or a tuple containing two ints.
            name: An optional string to specify the `name_scope` of this operation.
        Returns:
            A 4D tensor with shape [batch_size, height, width, 2] and type `float32`.
    """
    with tf.name_scope(name):
        batch_size = tf.shape(theta)[0]
        theta = tf.reshape(theta, [batch_size, 2, 3])
        if isinstance(size, tuple):
            width, height = size
        else:
            width, height = tf.unstack(size)
        # Normalized grid (2D)
        x = tf.linspace(-1.0, 1.0, width)
        if isinstance(size, tuple) and (width == height):
            y = x
        else:
            y = tf.linspace(-1.0, 1.0, height)
        x_grid, y_grid = tf.meshgrid(x, y)

        # Flatten
        x_grid = tf.reshape(x_grid, [-1])
        y_grid = tf.reshape(y_grid, [-1])

        # Sampling grid
        ones = tf.ones_like(x_grid)
        sampling_grid = tf.stack([x_grid, y_grid, ones])
        sampling_grid = tf.expand_dims(sampling_grid, axis=0)
        sampling_grid = tf.tile(sampling_grid, tf.stack([batch_size, 1, 1]))

        grid = theta @ sampling_grid
        grid = tf.reshape(grid, [batch_size, 2, height, width])
        grid = tf.transpose(grid, [0, 2, 3, 1])
        return grid


# region Helper functions
def floor_int(tensor: tf.Tensor, name="floor_int") -> tf.Tensor:
    """Returns element-wise largest integer not greater than `tensor` and casts it to `int32`.

        Arguments:
            tensor: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
            name: A name for the operation (optional).

        Returns:
            A `Tensor`. Has the same shape as `tensor` and has type `int32`.
    """
    return tf.cast(tf.floor(tensor), dtype=tf.int32, name=name)


def repeat_1d(inputs: tf.Tensor, count: Union[tf.Tensor, int], name="repeat_1d"):
    """Repeats each element of `inputs` `count` times in a row.

        '''python
        repeat_1d(tf.range(4), 2) -> 0, 0, 1, 1, 2, 2, 3, 3
        '''

        Parameters:
            inputs: A 1D tensor with shape [`size`] to be repeated.
            count: An integer, used to specify the number of time elements of `inputs` are repeated.
            name: An optional string to specify the `name_scope` of this operation.
        Returns:
            A 1D tensor with shape [`size` * `count`] and same type as `inputs`.

    """
    with tf.name_scope(name):
        outputs = tf.expand_dims(inputs, 1)
        outputs = tf.tile(outputs, [1, count])
        outputs = tf.reshape(outputs, [-1])

        return outputs


def xy_from_grid(grid: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Separates `x` and `y` coordinates from `grid`.

        Parameters:
            grid: A 4D tensor with shape [batch_size, height, width, 2].
        Returns:
            Two 3D tensors with shape [batch_size, height, width] and same type as `grid`.
    """
    x = tf.reshape(grid[:, :, :, 0], [-1])
    y = tf.reshape(grid[:, :, :, 1], [-1])
    return x, y


# endregion

def bilinear_sampling(images: tf.Tensor, grid: tf.Tensor, output_size: tf.Tensor, name="interpolate_image"):
    """Performs bilinear sampling on input `images` using the affine `grid`.

            Arguments:
                images: A 4D tensor with shape [batch_size, height, width, channels] and float32 dtype.
                    Images to be transformed.
                grid: A 4D tensor with shape [batch_size, height, width, 2] and float32 dtype.
                    The grid used to perform bilinear sampling, maps points to sample to their destination.
                output_size: A 1D tensor with shape [2] containing a couple of integers.
                name: An optional string for the `tf.name_scope` of this function.
            Returns:
                A 4D tensor with shape [batch_size, output_height, output_width, channels]. Images transformed by the
                    `grid`.

        """
    with tf.name_scope(name):
        # region Constants
        image_shape = tf.shape(images)
        batch_size, height, width, channels = tf.unstack(image_shape)
        output_height, output_width = tf.unstack(output_size)

        one_float = tf.constant(1.0, dtype=tf.float32, name="one_float")
        half_float = tf.constant(0.5, dtype=tf.float32, name="half_float")
        zero_int = tf.constant(0, dtype=tf.int32, name="zero_int")
        one_int = tf.constant(1, dtype=tf.int32, name="one_int")
        # endregion

        # region Get x0|x1|y0|y1 and clip them
        x, y = xy_from_grid(grid)

        x = (x + one_float) * tf.cast(width, tf.float32) * half_float
        y = (y + one_float) * tf.cast(height, tf.float32) * half_float

        x0 = floor_int(x)
        x1 = x0 + one_int
        y0 = floor_int(y)
        y1 = y0 + one_int

        x0 = tf.clip_by_value(x0, clip_value_min=zero_int, clip_value_max=width - one_int)
        x1 = tf.clip_by_value(x1, clip_value_min=zero_int, clip_value_max=width - one_int)
        y0 = tf.clip_by_value(y0, clip_value_min=zero_int, clip_value_max=height - one_int)
        y1 = tf.clip_by_value(y1, clip_value_min=zero_int, clip_value_max=height - one_int)
        # endregion

        # region Get gather indices
        base = repeat_1d(tf.range(batch_size) * width * height, output_height * output_width)
        base_y0 = base + y0 * width
        base_y1 = base + y1 * width
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1
        # endregion

        # region Gather params from image
        image_flat = tf.reshape(images, [-1, channels])
        gathered_a = tf.gather(params=image_flat, indices=indices_a)
        gathered_b = tf.gather(params=image_flat, indices=indices_b)
        gathered_c = tf.gather(params=image_flat, indices=indices_c)
        gathered_d = tf.gather(params=image_flat, indices=indices_d)
        # endregion

        x0 = tf.cast(x0, dtype=tf.float32)
        x1 = tf.cast(x1, dtype=tf.float32)
        y0 = tf.cast(y0, dtype=tf.float32)
        y1 = tf.cast(y1, dtype=tf.float32)

        w_a = (x1 - x) * (y1 - y)
        w_b = (x1 - x) * (y - y0)
        w_c = (x - x0) * (y1 - y)
        w_d = (x - x0) * (y - y0)

        w_a = tf.expand_dims(w_a, axis=1)
        w_b = tf.expand_dims(w_b, axis=1)
        w_c = tf.expand_dims(w_c, axis=1)
        w_d = tf.expand_dims(w_d, axis=1)

        output_flat_image = tf.add_n([w_a * gathered_a, w_b * gathered_b, w_c * gathered_c, w_d * gathered_d])
        output_image = tf.reshape(output_flat_image, [batch_size, output_height, output_width, channels])
        return output_image


def adjust_theta(theta: tf.Tensor) -> tf.Tensor:
    """Adjusts theta to use it with the affine grid for bilinear sampling.

        Arguments:
            theta: A 2D or 3D tensor with shape [batch_size, 6] or [batch_size, 2, 3] and type `float32`.

        Returns:
            A tensor with same shape and type as theta.
    """
    theta_shape = tf.shape(theta)
    batch_size = theta_shape[0]
    theta = tf.reshape(theta, [batch_size, 2, 3])

    mul_theta = theta[:, :, :2]
    mul_theta = tf.linalg.inv(mul_theta)

    add_theta = theta[:, :, -1] * -1
    height = mul_theta[:, 0, 0]
    width = mul_theta[:, 1, 1]
    x_center = (add_theta[:, 0] + 0.5) * height
    y_center = (add_theta[:, 1] + 0.5) * width
    center = tf.stack([y_center, x_center], axis=1) * 2 - 1
    center = tf.expand_dims(center, axis=-1)

    theta = tf.concat([mul_theta, center], axis=-1)
    theta = tf.reshape(theta, theta_shape)
    return theta


def spatial_transformation(inputs: tf.Tensor,
                           theta: tf.Tensor,
                           output_size: Tuple[int, int] = None
                           ) -> tf.Tensor:
    """Transform input images according to theta. If `output_size` is provided, also rescales images to this size.

        Arguments:
            inputs: A 4D tensor with shape [batch_size, height, width, channels] and float32 dtype.
                Images to be transformed.
            theta: A 2D or 3D tensor with shape [batch_size, 6] or [batch_size, 2, 3]  and float32 dtype.
                The affine transformation matrix.
            output_size: An optional 1D tensor with shape [2] or tuple containing a couple of integers.
        Returns:
            A 4D tensor with shape [batch_size, output_height, output_width, channels]. Images transformed by the
                affine transformation matrix Theta.

    """
    with tf.name_scope("spatial_transformation"):
        if output_size is None:
            output_shape = inputs.shape
            output_size = tf.shape(inputs)[1:3]
        else:
            output_shape = [inputs.shape[0], *output_size, inputs.shape[3]]
            output_size = tf.constant(output_size, dtype=tf.int32)

        theta = adjust_theta(theta)
        grid = affine_grid(theta, output_size)
        outputs = bilinear_sampling(inputs, grid, output_size, name="bilinear_sampler")
        outputs.set_shape(output_shape)
        return outputs
