# Custom Keras Layers
Now updated for TF2.0 (and tf.keras).

## Table of Contents
    1. Layers
    2. ResBlock
        a. Fixup Initialization
    3. Spatial Tranformer
        a. Usage

## Layers

Layers:
- ResBlock 
    - Versions : 1D, 2D, 3D, 2DTranspose/Deconv, 3DTranspose/Deconv
    - Basic blocks are a separate class
- DenseBlock
    - Versions : 1D, 2D, 3D
    - Composite Function Block as a separate class
- SpatialTransformer
    
Very basic tests on Cifar10 are provided in /tests for ResBlock and DenseBlock.

## ResBlock
Original paper (ResNet):

    @inproceedings{he2016deep,
      title={Deep residual learning for image recognition},
      author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
      booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
      pages={770--778},
      year={2016}
    }

#### Fixup Initialization
ResBlocks in this project use the Fixup Initialization, which allows to drop Batch Normalization layers.

To use fixup initialization, provide the ResBlocks with the initializer like so:

```python
from layers import ResBlock2D

fixup_initializer = ResBlock2D.get_fixup_initializer(total_depth)
block = ResBlock2D(kernel_initializer=fixup_initializer, ...)
```

![Fixup initialization](https://i.stack.imgur.com/T67F3.png)

    @article{zhang2019fixup,
      Title={Fixup Initialization: Residual Learning Without Normalization},
      Author={Zhang, Hongyi and Dauphin, Yann N and Ma, Tengyu},
      Journal={arXiv preprint arXiv:1901.09321},
      Year={2019}
    }
    
Authors shared their [article on Arxiv](https://arxiv.org/abs/1901.09321)

## SpatialTransformer
Original paper :

    @inproceedings{jaderberg2015spatial,
      title={Spatial transformer networks},
      author={Jaderberg, Max and Simonyan, Karen and Zisserman, Andrew and others},
      booktitle={Advances in neural information processing systems},
      pages={2017--2025},
      year={2015}
    }

#### Usage
You must provide a `Localisation network` (Layer or Model) to the Spatial Transformer.

If the dimension of the output of the `Localisation network` if not 6, the Spatial Transformer
will add one dense layer with an output of dimension 6 (and flatten its inputs).

Why 6 ? It is the dimension of `theta`, the affine 2x3 transformation matrix used to transform
 the input.
 
The transformation is performed through differentiable bilinear sampling.    

Here is a small example:
```python
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from layers import SpatialTransformer

# Build
input_shape = (128, 128, 3)
encoder_input_shape = (64, 64, 3)

localisation_net = Sequential(...)
spatial_transformer = SpatialTransformer(localisation_net,
                                         output_size=encoder_input_shape)
encoder = Sequential(...)
...

# Call
input_layer = Input(input_shape)

if output_theta:
    layer, theta = spatial_transformer(input_layer, output_theta=True)
else:
    layer = spatial_transformer(input_layer)
   
layer = encoder(layer)
...

```