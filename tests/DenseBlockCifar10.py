from keras.layers import Input, AveragePooling2D, Dense, Reshape, Conv2D
from keras.models import Model
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import numpy as np

from layers import DenseBlock2D


def evaluate_on_cifar10():
    total_depth = 100
    n_blocks = 3
    depth = (total_depth - 4) // n_blocks
    growth_rate = 12
    filters = growth_rate * 2

    # region Model
    input_layer = Input(shape=[32, 32, 3])
    layer = input_layer
    layer = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")(layer)

    for k in range(n_blocks):
        layer = DenseBlock2D(rank=2, kernel_size=3, growth_rate=growth_rate, depth=depth,
                             use_batch_normalization=True)(layer)

        if k < (n_blocks - 1):
            filters += growth_rate * depth // 4
            layer = transition_block(layer, filters)
        else:
            layer = AveragePooling2D(pool_size=8)(layer)

    layer = Reshape([-1])(layer)
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
    generator.fit(x_train, seed=0)
    # endregion

    model.fit_generator(generator.flow(x_train, y_train, batch_size=100),
                        steps_per_epoch=100, epochs=300, validation_data=(x_test, y_test),
                        validation_steps=100, verbose=1)


def transition_block(layer, filters):
    layer = Conv2D(filters=filters, kernel_size=1, kernel_initializer="he_normal", use_bias=False,
                   kernel_regularizer=l2(1e-4))(layer)
    layer = AveragePooling2D(pool_size=2, strides=2)(layer)
    return layer


if __name__ == "__main__":
    evaluate_on_cifar10()
