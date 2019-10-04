import tensorflow as tf
from tensorflow.python.keras import backend, activations, Model
from tensorflow.python.keras.layers import Layer
from typing import Union, List

from CustomKerasLayers.layers.MaskedConvStack import MaskedConvStack


def compute_autoregression_loss(y_true, y_pred_distribution):
    y_pred_distribution_shape = tf.shape(y_pred_distribution)
    batch_size = y_pred_distribution_shape[0]
    n_bins = y_pred_distribution_shape[-1]
    epsilon = backend.epsilon()

    y_pred_distribution = tf.nn.softmax(y_pred_distribution, axis=1)
    y_pred_distribution = tf.reshape(y_pred_distribution, [batch_size, -1, n_bins])
    y_pred_distribution = tf.clip_by_value(y_pred_distribution, epsilon, 1.0 - epsilon)
    y_pred_distribution = tf.math.log(y_pred_distribution)

    n_bins = tf.cast(n_bins, tf.float32)
    y_true = tf.reshape(y_true, [batch_size, -1, 1])
    y_true = tf.clip_by_value(y_true * n_bins, 0.0, n_bins - 1.0)
    y_true = tf.cast(y_true, tf.int32)

    selected_bins = tf.gather(y_pred_distribution, indices=y_true, batch_dims=-1)
    loss = - tf.reduce_mean(selected_bins)
    return loss


class ConvAM(Model):
    """
    Convolutional Autoregressive Model.
    """

    def __init__(self,
                 rank: int,
                 filters: List[int],
                 intermediate_activation: Union[str, Layer],
                 output_activation: Union[str, Layer],
                 **kwargs
                 ):
        super(ConvAM, self).__init__(**kwargs)

        self.rank = rank
        self.filters = filters
        self.intermediate_activation = activations.get(intermediate_activation)
        self.output_activation = activations.get(output_activation)

        self.conv_layers = [
            MaskedConvStack(rank=rank,
                            filters=filters[i],
                            mask_center=i == 0,
                            kernel_size=(3,),
                            activation=self.intermediate_activation
                            if (i != (len(filters) - 1))
                            else self.output_activation,
                            use_bias=True,
                            )
            for i in range(len(filters))
        ]

    def call(self, inputs, training=None, mask=None):
        outputs = tf.expand_dims(inputs, axis=-1)
        for layer in self.conv_layers:
            outputs = layer(outputs)
        return outputs

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(inputs)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    @tf.function
    def compute_loss(self, inputs):
        y_pred_distribution = self(inputs)
        loss = compute_autoregression_loss(inputs, y_pred_distribution)
        return loss
