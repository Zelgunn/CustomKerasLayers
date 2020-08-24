import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from typing import Tuple


class VideoPatchExtractor(Layer):
    def __init__(self,
                 patch_size: int,
                 pick_random_during_training=True,
                 **kwargs):
        super(VideoPatchExtractor, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.original_height = tf.Variable(initial_value=-1, trainable=False, name="original_height")
        self.original_width = tf.Variable(initial_value=-1, trainable=False, name="original_width")
        self.pick_random_during_training = pick_random_during_training

    def call(self, inputs, **kwargs):
        training = kwargs["training"] if "training" in kwargs else False

        # region Shape variables
        inputs_shape = tf.shape(inputs)
        batch_size, length, original_height, original_width, channels = tf.unstack(inputs_shape)
        patch_count = (original_height // self.patch_size) * (original_width // self.patch_size)
        self.original_height.assign(original_height)
        self.original_width.assign(original_width)
        # endregion

        # inputs : [batch_size, length, original_height, original_width, channels]
        inputs = tf.reshape(inputs, [batch_size * length, original_height, original_width, channels])
        patches = tf.image.extract_patches(inputs, sizes=self.patch_shape, strides=self.patch_shape,
                                           rates=(1, 1, 1, 1), padding="VALID")

        # patches : [batch_size * length, n_height, n_width, height * width * channels]
        patches = tf.reshape(patches, [batch_size, length, patch_count, self.patch_size, self.patch_size, channels])

        if training:
            patches = patches[:, :, ]
        else:
            patches = tf.transpose(patches, perm=[0, 2, 1, 3, 4, 5])
            patches = tf.reshape(patches, [batch_size * patch_count, length, self.patch_size, self.patch_size,
                                           channels])

        return patches

    @property
    def patch_shape(self) -> Tuple[int, int, int, int]:
        return 1, self.patch_size, self.patch_size, 1
