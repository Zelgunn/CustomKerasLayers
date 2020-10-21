import tensorflow as tf
from tensorflow.python.keras.layers import Input, AveragePooling2D, Dense, Flatten
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import os
from time import time

from CustomKerasLayers import ResBlock2D


def main():
    tf.random.set_seed(42)

    total_depth = 36
    n_blocks = 3
    basic_block_count = total_depth // n_blocks

    # region Model
    input_layer = Input(shape=[32, 32, 3])
    layer = input_layer

    for k in range(n_blocks):
        strides = 2 if k < (n_blocks - 1) else 1
        layer = ResBlock2D(filters=16 * (2 ** k), basic_block_count=basic_block_count, strides=strides)(layer)

        if k == (n_blocks - 1):
            layer = AveragePooling2D(pool_size=8)(layer)

    layer = Flatten()(layer)
    layer = Dense(units=10, activation="softmax")(layer)
    model = Model(inputs=input_layer, outputs=layer)
    model.summary()

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    # endregion

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

    log_dir = "../logs/tests/res_block_cifar10/{}".format(int(time()))
    log_dir = os.path.normpath(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir, profile_batch=0)

    model.fit_generator(generator.flow(x_train, y_train, batch_size=64),
                        steps_per_epoch=100, epochs=300, validation_data=(x_test, y_test),
                        validation_steps=100, verbose=1, callbacks=[tensorboard])


if __name__ == "__main__":
    main()
