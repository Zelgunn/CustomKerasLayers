# Custom Keras Layers
Layers:
- CompositeLayer
    - Base class for layers made of keras layers, only used to track layers weights
- ResBlock 
    - Versions : 1D, 2D, 3D, 2DTranspose/Deconv, 3DTranspose/Deconv
    - Basic blocks are a separate class
- DenseBlock
    - Versions : 1D, 2D, 3D
    - Composite Function Block as a separate class
    
Now updated for TF2.0 (and tf.keras).

Very basic tests on Cifar10 are provided in /tests.

## ResBlocks
### Fixup Initialization
ResBlocks in this project use the Fixup Initialization, which allows to drop Batch Normalization layers.

To use fixup initialization, provide the ResBlocks with the initializer like so:

``` python
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
