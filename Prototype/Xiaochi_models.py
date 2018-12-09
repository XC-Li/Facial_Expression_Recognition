"""
Keras models
By: Xiaochi (George) Li
Dec.2018

"""
import tensorflow as tf
from tensorflow.keras import layers, Sequential


def light_vgg():
    """
    this model is inspired by vgg net, but lighter
    CV_Accuracy ~75% in 10 epoch, 12s per epoch

    """
    model = Sequential([
        # Block 1: in:100x100x3, out:50x50x64
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='b1_conv1'),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='b1_conv2'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), name='b1_maxpool'),

        # Block 2 in:50x50x64, out:25x25x128
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='b2_conv1'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='b2_conv2'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), name='b2_maxpool'),

        # Blcok 3 in:25x25x128, out:12x12x256
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='b3_conv1'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='b3_conv2'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), name='b3_maxpool'),

        # Full connect layer

        layers.Flatten(name='flatten'),
        layers.Dense(2048, activation='relu', name='fc1'),
        layers.Dropout(0.7, seed=42),  # Answer to the Ultimate Question of Life, the Universe, and Everything (42)
        layers.Dense(2048, activation='relu', name='fc2'),
        layers.Dropout(0.7, seed=42),
        layers.Dense(512, activation='relu', name='fc3'),
        layers.Dense(128, activation='relu', name='fc4'),
        layers.Dense(32, activation='relu', name='fc5'),

        layers.Dense(7, activation='softmax')
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics=[tf.keras.metrics.categorical_accuracy])
    return model