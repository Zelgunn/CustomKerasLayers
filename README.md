# Custom Keras Layers
Layers:
- CompositeLayer
    - Base class for layers made of keras layers, only used to track layers weights
- ResBlock 
    - Versions : 1D, 2D, 3D, 2DTranspose/Deconv, 3DTranspose/Deconv
    - Basic block as a separate class
- DenseBlock
    - Versions : 1D, 2D, 3D
    - Composite Function Block as a separate class

An alternative version is available in `tf_compat` if you need to use `tensorflow.python.keras` instead of `keras`. This is useful if you plan to use the `tf.data` API for example.

## ResBlocks
### Fixup Initialization
ResBlocks in this project use the Fixup Initialization, which allows to drop Batch Normalization layers.

![Fixup initialization](https://i.stack.imgur.com/T67F3.png)

    @article{zhang2019fixup,
      Title={Fixup Initialization: Residual Learning Without Normalization},
      Author={Zhang, Hongyi and Dauphin, Yann N and Ma, Tengyu},
      Journal={arXiv preprint arXiv:1901.09321},
      Year={2019}
    }
    
Authors shared their [article on Arxiv](https://arxiv.org/abs/1901.09321)
