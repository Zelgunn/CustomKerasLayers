import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import MaxPooling2D, Flatten, Dense
from tensorflow.python.ops.init_ops import VarianceScaling
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import os
from time import time

from CustomKerasLayers.layers.ResSASABlock import ResSASABlock
from CustomKerasLayers.layers.ResBlock import ResBlock2D


def main():
    tf.random.set_seed(42)

    block_count = 4
    basic_block_count = 8
    input_shape = (32, 32, 3)

    layers_params = {
        "rank": 2,
        "head_size": 8,
        "head_count": 8,
        "basic_block_count": basic_block_count,
        "kernel_size": 3,
        "strides": 1,
        "dilation_rate": 1,
        "activation": "relu",
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
    }

    layers = [
        ResBlock2D(filters=16, basic_block_count=basic_block_count, kernel_size=7, input_shape=input_shape),
        MaxPooling2D(4)
    ]
    for i in range(1, block_count):
        layer = ResSASABlock(**layers_params)
        layers.append(layer)
        layers_params["head_size"] *= 2
        layers.append(MaxPooling2D(2))

    layers.append(Flatten())
    layers.append(Dense(units=10, activation="softmax", kernel_initializer=VarianceScaling()))

    model = Sequential(layers=layers, name="StandAloneSelfAttentionBasedClassifier")
    model.summary()
    model.compile("adam", loss="categorical_crossentropy", metrics=["acc"])

    # region Data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    generator = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=5. / 32,
                                   height_shift_range=5. / 32,
                                   horizontal_flip=True)
    generator.fit(x_train)
    # endregion

    log_dir = "../../logs/tests/stand_alone_self_attention_cifar10/{}".format(int(time()))
    log_dir = os.path.normpath(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir, profile_batch="500,520")

    model.fit(generator.flow(x_train, y_train, batch_size=64),
              steps_per_epoch=100, epochs=300, validation_data=(x_test, y_test),
              validation_steps=100, verbose=1, callbacks=[tensorboard])


if __name__ == "__main__":
    main()
