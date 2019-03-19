# Custom Keras Layers
ResBlock and DenseBlock layers for Keras API.

ResBlocks use the Fixup Initialization, which allows to drop Batch Normalization layers.

![Fixup initialization](https://i.stack.imgur.com/T67F3.png)

    @article{zhang2019fixup,
      Title={Fixup Initialization: Residual Learning Without Normalization},
      Author={Zhang, Hongyi and Dauphin, Yann N and Ma, Tengyu},
      Journal={arXiv preprint arXiv:1901.09321},
      Year={2019}
    }
    
Authors shared their [article on Arxiv](https://arxiv.org/abs/1901.09321)
