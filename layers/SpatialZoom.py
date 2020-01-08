import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import Model
from typing import Union

from CustomKerasLayers import SpatialTransformer
from CustomKerasLayers.layers.SpatialTransformer import spatial_transformation


class SpatialZoom(SpatialTransformer):
    """Spatial Zoom class, inherits from class `SpatialTransformer`.

        You must provide a `localisation_net` which is a Model or a Layer that takes regular inputs and outputs a tensor
        with total output dimension 2 (without batch). If the dimension if different from 2, the Spatial Transformer
        will add a Dense layer reducing dimensionality to 2.

        The Spatial Transformer uses this tensor of size 2 as the affine transformation matrix for bilinear sampling.

        The outputs of the Spatial Transformer are the transformed input images.
        If `output_theta` is provided when calling the layer (and is true), then the Spatial Transformer
        will also output `theta`, the affine transformation matrix.

    """

    def __init__(self,
                 localisation_net: Union[Model, Layer],
                 # zoom_factor=2.0,
                 output_size=None,
                 data_format=None,
                 **kwargs):
        super(SpatialZoom, self).__init__(localisation_net=localisation_net,
                                          output_size=output_size,
                                          data_format=data_format,
                                          **kwargs)
        zoom_factor = 2.0
        self.zoom_factor = zoom_factor
        self.constant_theta = tf.constant([zoom_factor], dtype=tf.float32, shape=[1, 1])

    def get_theta(self, inputs):
        batch_size = tf.shape(inputs)[0]
        theta = self.localisation_net(inputs)
        theta = self._project_theta(theta, batch_size)
        x, y = tf.split(theta, num_or_size_splits=2, axis=-1)
        constant_theta = tf.tile(self.constant_theta, [batch_size, 1])
        zeros = tf.zeros_like(constant_theta)
        theta = tf.concat([constant_theta, zeros, x, zeros, constant_theta, y], axis=-1)
        theta = tf.reshape(theta, [batch_size, 2, 3])
        return theta

    def _project_theta(self, theta, batch_size):
        theta = super(SpatialZoom, self)._project_theta(theta, batch_size)
        theta = - tf.nn.sigmoid(theta)
        return theta

    def transform_inputs(self, inputs, theta):
        output_size = self.compute_output_size(inputs.shape)
        return spatial_transformation(inputs, theta, output_size)

    def compute_output_size(self, input_shape):
        if isinstance(input_shape, (tuple, list)) and len(input_shape) == 2:
            input_shape = input_shape[1]

        if self.output_size == "zoom":
            if self.channels_first:
                height, width = input_shape[2:4]
            else:
                height, width = input_shape[1:3]
            height = int(height / self.zoom_factor)
            width = int(width / self.zoom_factor)
            output_size = (height, width)
        else:
            output_size = self.output_size

        return output_size

    def compute_output_shape(self, input_shape):
        tmp_output_size = self.output_size
        self.output_size = self.compute_output_size(input_shape)
        output_shape = self.compute_output_shape(input_shape)
        self.output_size = tmp_output_size
        return output_shape

    @property
    def localisation_dim(self) -> int:
        return 2

    def bias_identity_init(self, shape, dtype=None):
        return tf.ones([self.localisation_dim]) * 0.5

    def get_config(self):
        config = {
            "zoom_factor": self.zoom_factor,
        }
        base_config = super(SpatialTransformer, self).get_config()
        return {**config, **base_config}
