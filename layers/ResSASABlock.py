from tensorflow.python.keras import layers
from tensorflow.python.keras.initializers.initializers_v2 import RandomNormal
from typing import Union, Optional, Tuple, List, AnyStr, Callable, Dict, Type

from CustomKerasLayers.layers.StandAloneSelfAttention import StandAloneSelfAttention
from CustomKerasLayers.layers.ResBlock import ResBasicBlockND, ResBlockND


class ResSASABasicBlock(ResBasicBlockND):
    def __init__(self,
                 rank: int,
                 head_size: int,
                 head_count: int,
                 depth: int,
                 kernel_size: Union[int, Tuple, List],
                 strides: Union[int, Tuple, List],
                 dilation_rate: Union[int, Tuple, List],
                 kernel_regularizer: Optional[Union[Dict, AnyStr, Callable]],
                 bias_regularizer: Optional[Union[Dict, AnyStr, Callable]],
                 activity_regularizer: Optional[Union[Dict, AnyStr, Callable]],
                 kernel_constraint: Optional[Union[Dict, AnyStr, Callable]],
                 bias_constraint: Optional[Union[Dict, AnyStr, Callable]],
                 seed: Optional[int],
                 **kwargs):
        filters = head_count * head_size
        self.head_size = head_size
        self.head_count = head_count
        self.embeddings_initializer = RandomNormal(stddev=1.0, seed=seed)

        super(ResSASABasicBlock, self).__init__(rank=rank, filters=filters, depth=depth, kernel_size=kernel_size,
                                                strides=strides, data_format=None, dilation_rate=dilation_rate,
                                                kernel_regularizer=kernel_regularizer,
                                                bias_regularizer=bias_regularizer,
                                                activity_regularizer=activity_regularizer,
                                                kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                                seed=seed, **kwargs)

    def get_conv_layer_type(self) -> Type[StandAloneSelfAttention]:
        return StandAloneSelfAttention

    def init_layer(self, use_strides: bool, projection_layer: bool) -> StandAloneSelfAttention:
        strides = self.strides if (use_strides or projection_layer) else 1
        kernel_size = self.projection_kernel_size if projection_layer else self.kernel_size
        return StandAloneSelfAttention(rank=self.rank,
                                       head_size=self.head_size,
                                       head_count=self.head_count,
                                       kernel_size=kernel_size,
                                       strides=strides,
                                       dilation_rate=self.dilation_rate,
                                       activation=None,
                                       use_bias=not projection_layer,
                                       kernel_initializer=self.kernel_initializer,
                                       bias_initializer="zeros",
                                       embeddings_initializer=self.embeddings_initializer,
                                       kernel_regularizer=self.kernel_regularizer,
                                       bias_regularizer=self.bias_regularizer,
                                       activity_regularizer=self.activity_regularizer,
                                       kernel_constraint=self.kernel_constraint,
                                       bias_constraint=self.bias_constraint)

    def build(self, input_shape):
        super(ResSASABasicBlock, self).build(input_shape)
        self.input_spec = layers.InputSpec(shape=input_shape)

    def get_config(self):
        config = \
            {
                **super(ResSASABasicBlock, self).get_config(),
                "head_size": self.head_size,
                "head_count": self.head_count,
            }

        return config


class ResSASABlock(ResBlockND):
    def __init__(self,
                 rank: int,
                 head_size: int,
                 head_count: int,
                 basic_block_count=1,
                 basic_block_depth=1,
                 kernel_size: Union[int, Tuple, List] = 3,
                 strides: Union[int, Tuple, List] = 1,
                 data_format: AnyStr = None,
                 dilation_rate: Union[int, Tuple, List] = 1,
                 activation: Union[None, AnyStr, Callable] = "relu",
                 projection_kernel_initializer: Union[Dict, AnyStr, Callable] = None,
                 kernel_regularizer: Union[Dict, AnyStr, Callable] = None,
                 bias_regularizer: Union[Dict, AnyStr, Callable] = None,
                 activity_regularizer: Union[Dict, AnyStr, Callable] = None,
                 kernel_constraint: Union[Dict, AnyStr, Callable] = None,
                 bias_constraint: Union[Dict, AnyStr, Callable] = None,
                 seed: Optional[int] = None,
                 **kwargs):
        filters = head_count * head_size
        self.head_size = head_size
        self.head_count = head_count

        super(ResSASABlock, self).__init__(rank=rank, filters=filters, basic_block_count=basic_block_count,
                                           basic_block_depth=basic_block_depth, kernel_size=kernel_size,
                                           strides=strides, data_format=data_format, dilation_rate=dilation_rate,
                                           activation=activation,
                                           projection_kernel_initializer=projection_kernel_initializer,
                                           kernel_regularizer=kernel_regularizer,
                                           bias_regularizer=bias_regularizer,
                                           activity_regularizer=activity_regularizer,
                                           kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                           seed=seed, **kwargs)

    def init_layers(self):
        for i in range(self.basic_block_count):
            strides = self.strides if (i == 0) else 1
            basic_block = ResSASABasicBlock(rank=self.rank,
                                            head_size=self.head_size,
                                            head_count=self.head_count,
                                            depth=self.basic_block_depth,
                                            kernel_size=self.kernel_size,
                                            strides=strides,
                                            dilation_rate=self.dilation_rate,
                                            kernel_regularizer=self.kernel_regularizer,
                                            bias_regularizer=self.bias_regularizer,
                                            activity_regularizer=self.activity_regularizer,
                                            kernel_constraint=self.kernel_constraint,
                                            bias_constraint=self.bias_constraint,
                                            seed=self.seed)
            self.basic_blocks.append(basic_block)

    def build(self, input_shape):
        super(ResSASABlock, self).build(input_shape)
        self.input_spec = layers.InputSpec(shape=input_shape)

    def get_config(self):
        config = \
            {
                **super(ResSASABlock, self).get_config(),
                "head_size": self.head_size,
                "head_count": self.head_count,
            }

        return config
