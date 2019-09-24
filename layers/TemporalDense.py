import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Dense


class TemporalDense(Layer):
    def __init__(self,
                 units,
                 axis=0,
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
        super(TemporalDense, self).__init__(**kwargs)

        self.dense = Dense(units=units,
                           activation=activation,
                           use_bias=use_bias,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           activity_regularizer=activity_regularizer,
                           kernel_constraint=kernel_constraint,
                           bias_constraint=bias_constraint)
        self.axis = axis + 1

    def call(self, inputs, **kwargs):
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        time = inputs_shape[self.axis]

        axis = self.axis
        if axis < 0:
            axis = inputs.shape.rank + self.axis

        if axis != 1:
            perm = list(range(inputs.shape.rank))
            perm.remove(axis)
            perm.insert(1, axis)
            inputs = tf.transpose(inputs, perm=perm)

        inputs = tf.reshape(inputs, [batch_size, time, -1])
        outputs = self.dense(inputs)
        return outputs
