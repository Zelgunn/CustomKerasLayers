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
    
Currently implemented paper (Batch Normalization Biases Deep Residual Networks Towards Shallow Paths):

    @article{de2020batch,
        title={Batch normalization biases deep residual networks towards shallow paths},
        author={De, Soham and Smith, Samuel L},
        journal={arXiv preprint arXiv:2002.10444},
        year={2020}
    }

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
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input
from CustomKerasLayers import SpatialTransformer

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

output_theta = False
if output_theta:
    layer, theta = spatial_transformer(input_layer, output_theta=True)
else:
    layer = spatial_transformer(input_layer)
   
layer = encoder(layer)
...

```