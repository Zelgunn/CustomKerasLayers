import tensorflow as tf
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops.array_ops import extract_image_patches, extract_volume_patches
from typing import Union, List, Tuple, Any, Optional, Dict


class Unfold(Layer):
    def __init__(self,
                 kernel_size: Union[int, Tuple[int, ...], List[int]],
                 strides: Union[int, Tuple[int, ...], List[int]] = 1,
                 padding: str = "VALID",
                 dilation_rate: Union[int, Tuple[int, ...], List[int]] = 1,
                 **kwargs):
        super(Unfold, self).__init__(**kwargs)

        # region Get rank (if possible)
        rank = kwargs["rank"] if "rank" in kwargs else None
        for value in (kernel_size, padding, dilation_rate):
            value_rank = get_rank_from(value)
            if rank is None:
                rank = value_rank
            elif (value_rank is not None) and (rank != value_rank):
                raise ValueError("Could not determine rank for Unfold, received values with different ranks : {} and {}"
                                 "".format(rank, value_rank))

        if (rank is not None) and (rank not in (1, 2, 3)):
            raise ValueError("Parameters' rank is invalid. Expected 1, 2 or 3. Found {}.".format(rank))

        self.rank = rank
        # endregion

        self._kernel_size = kernel_size
        self._strides = strides
        self._dilation_rate = dilation_rate
        self.padding = self.normalize_padding(padding)

    def build(self, input_shape: tf.TensorShape):
        if self.rank is None:
            if input_shape.rank not in (3, 4, 5):
                raise ValueError("Inputs' rank is invalid. Expected 1, 2 or 3. Found {}.".format(input_shape.rank))
            self.rank = input_shape.rank - 2

        self.input_spec = InputSpec(shape=input_shape)
        # input_channel = input_shape[-1]
        # self.input_spec = InputSpec(ndim=self.rank + 2, axes={-1: input_channel})

    def call(self, inputs, **kwargs):
        # region Getting parameters for extraction
        if self.rank == 1:
            sizes = (1, 1, *self.kernel_size, 1)
            strides = (1, 1, *self.strides, 1)
            rates = (1, 1, *self.dilation_rate, 1)
            inputs = tf.expand_dims(inputs, axis=1)
        else:
            sizes = (1, *self.kernel_size, 1)
            strides = (1, *self.strides, 1)
            rates = (1, *self.dilation_rate, 1)
        # endregion

        # region Extraction
        if self.rank == 1:
            outputs = extract_image_patches(inputs, sizes=sizes, strides=strides, rates=rates, padding=self.padding)
            outputs = tf.squeeze(outputs, axis=1)
        elif self.rank == 2:
            outputs = extract_image_patches(inputs, sizes=sizes, strides=strides, rates=rates, padding=self.padding)
        elif self.rank == 3:
            outputs = extract_volume_patches(inputs, ksizes=sizes, strides=strides, padding=self.padding)
        else:
            raise AttributeError("Invalid rank : self.rank is {}.".format(self.rank))
        # endregion

        if len(outputs.shape) != len(inputs.shape):
            raise ValueError(outputs.shape, inputs.shape, self.name)

        return outputs

    @property
    def kernel_size(self) -> Tuple:
        if not isinstance(self._kernel_size, (tuple, list)):
            self._kernel_size = conv_utils.normalize_tuple(self._kernel_size, self.rank, "kernel_size")
        return self._kernel_size

    @property
    def strides(self) -> Tuple:
        if not isinstance(self._strides, (tuple, list)):
            self._strides = conv_utils.normalize_tuple(self._strides, self.rank, "strides")
        return self._strides

    @property
    def dilation_rate(self) -> Tuple:
        if not isinstance(self._dilation_rate, (tuple, list)):
            self._dilation_rate = conv_utils.normalize_tuple(self._dilation_rate, self.rank, "dilation_rate")
        return self._dilation_rate

    def get_config(self) -> Dict[str, Any]:
        base_config = super(Unfold, self).get_config()
        config = {
            **base_config,
            "rank": self.rank,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "dilation_rate": self.dilation_rate,
        }
        return config

    @staticmethod
    def normalize_padding(value):
        if not isinstance(value, str):
            raise ValueError("Value must be a string. Received {}.".format(value))
        value = value.upper()
        if value not in {"VALID", "SAME"}:
            raise ValueError("Value for the attr `padding` of {} is not in the list of allowed values `SAME`, `VALID`."
                             .format(value))
        return value


def get_rank_from(value: Union[Any, Tuple, List]) -> Optional[int]:
    if isinstance(value, (tuple, list)):
        return len(value)
    else:
        return None
