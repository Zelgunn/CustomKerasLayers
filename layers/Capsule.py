import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.ops.init_ops import VarianceScaling, Constant
# from tensorflow.python.keras.layers import Conv1D, Conv2D, Conv3D, Dense
# from tensorflow.python.keras.layers import BatchNormalization, concatenate
from tensorflow.python.keras.utils import conv_utils
# from tensorflow.python.keras import activations, initializers, regularizers, constraints
from typing import Optional, Tuple, Union

# from misc_utils.math_utils import squash

# NOTE : The code for this layer (Capsule) is incomplete.

class CapsuleND(Layer):
    def __init__(self,
                 rank: int,
                 kernel_size: Union[int, Tuple[int, int], Tuple[int, int, int]],
                 strides: Union[int, Tuple[int, int], Tuple[int, int, int]],
                 padding: str,
                 output_dim: int,
                 output_atoms: int,
                 **kwargs
                 ):
        super(CapsuleND, self).__init__(**kwargs)

        self.rank = rank
        self.kernel_size: Tuple = conv_utils.normalize_tuple(kernel_size, rank, "kernel_size")
        self.strides: Tuple = conv_utils.normalize_tuple(strides, rank, "strides")
        self.padding = conv_utils.normalize_padding(padding)
        self.output_dim = output_dim
        self.output_atoms = output_atoms

        self.input_dim: Optional[int] = None
        self.input_atoms: Optional[int] = None
        self.kernel: Optional[tf.Variable] = None
        self.bias: Optional[tf.Variable] = None

    def build(self, input_shape):
        batch_size, input_dim, input_atoms = input_shape

        self.kernel = self.add_weight(name="kernel", initializer=VarianceScaling(scale=0.1),
                                      shape=[*self.kernel_size, input_atoms, self.output_dim * self.output_atoms])
        self.bias = self.add_weight(name="bias", initializer=Constant(value=0.1),
                                    shape=[self.output_dim, self.output_atoms])

        self.input_dim = input_dim
        self.input_atoms = input_atoms

    def call(self,
             inputs: tf.Tensor,
             **kwargs
             ) -> tf.Tensor:
        votes = self.get_votes(inputs)
        outputs = self.update_routing(votes)
        return outputs

    def get_votes(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs_shape = tf.shape(inputs)
        batch_size, input_dim, input_atoms, *conv_dims = tf.unstack(inputs_shape)
        conv_inputs_shape = [batch_size * input_dim, input_atoms, *conv_dims]
        inputs = tf.reshape(inputs, shape=conv_inputs_shape)

        votes = self.convolution(inputs)

        # conv_dims = tf.unstack(tf.shape(votes)[2:])
        # votes_shape = [batch_size, input_dim, self.output_dim, self.output_atoms, *conv_dims]

        return votes

    def convolution(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.rank == 1:
            return tf.nn.conv1d(
                input=inputs,
                filters=self.kernel,
                stride=[1, 1, *self.strides],
                padding=self.padding,
                data_format="NCW"
            )
        elif self.rank == 2:
            return tf.nn.conv2d(
                input=inputs,
                filters=self.kernel,
                strides=[1, 1, *self.strides],
                padding=self.padding,
                data_format="NCHW"
            )
        elif self.rank == 3:
            return tf.nn.conv3d(
                input=inputs,
                filters=self.kernel,
                strides=[1, 1, *self.strides],
                padding=self.padding,
                data_format="NCDHW"
            )
        else:
            raise AttributeError("No operation found for rank={}".format(self.rank))

    # def update_routing(self, votes: tf.Tensor, ) -> tf.Tensor:
    #     def loop_cond(i, _, __):
    #         return i < 0
    #
    #     def loop_body(i, logits, outputs_array):
    #         pass
