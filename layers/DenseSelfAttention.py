import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from typing import Union

from transformers.core.MultiHeadAttention import MultiHeadAttention


class DenseSelfAttention(Layer):
    def __init__(self,
                 head_count: int,
                 head_size: int,
                 output_size: int,
                 activation: Union[str, Layer],
                 use_mask: bool = True,
                 seed: int = None,
                 **kwargs):
        super(DenseSelfAttention, self).__init__(**kwargs)

        self.attention_layer = MultiHeadAttention(head_count=head_count, keys_size=head_size, values_size=head_size,
                                                  output_size=output_size, activation=activation, seed=seed)
        self.use_mask = use_mask

    def call(self, inputs, **kwargs):
        if self.use_mask:
            mask_length = tf.shape(inputs)[-2]
            mask = 1.0 - tf.linalg.band_part(tf.ones([mask_length, mask_length]), 0, -1)
            mask = tf.expand_dims(mask, axis=0)
        else:
            mask = None

        outputs, _ = self.attention_layer([inputs, inputs, inputs], mask=mask)
        return outputs

    def get_config(self):
        base_config = super(DenseSelfAttention, self).get_config()
        return {
            **base_config,
            "head_count": self.attention_layer.head_count,
            "head_size": self.attention_layer.values_size,
            "output_size": self.attention_layer.output_size,
            "activation" : self.activation.activation,
            "use_mask": self.use_mask,
            "seed": self.attention_layer.seed,
        }
