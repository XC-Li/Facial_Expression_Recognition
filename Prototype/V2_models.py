"""
Prototype V2
Different Keras Models
By: Xiaochi (George) Li
Nov.2018
"""

import tensorflow as tf
from tensorflow.keras import layers

# Accuracy around 65% (epoch=200)
def first_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(100,100,3)),
        layers.Conv2D(32, kernel_size=3, activation='relu'),
        layers.Flatten(),
        layers.Dense(7, activation='softmax')
    ])

    model.compile(optimizer=tf.train.AdadeltaOptimizer(0.001),
                  loss='categorical_crossentropy',
                 metrics=[tf.keras.metrics.categorical_accuracy])

    return model


# Adding the second model with the maxpooling layer
# Accuracy around 66% (epoch=200)
def second_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(100,100,3)),
        layers.Conv2D(32, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(7, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adadelta(0.001),
                  loss='categorical_crossentropy',
                 metrics=[tf.keras.metrics.categorical_accuracy])

    return model

# Adding the third model with modified filter number
# Accuracy around 66% (epoch=100)
def third_model():
    model = tf.keras.Sequential([
        layers.Conv2D(128, kernel_size=3, activation='relu', input_shape=(100,100,3)),
        layers.Conv2D(128, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(7, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adadelta(0.001),
                  loss='categorical_crossentropy',
                 metrics=[tf.keras.metrics.categorical_accuracy])

    return model

# Adding some dense and dropout layers
# Accuracy around
def fourth_model():
    model = tf.keras.Sequential([
        layers.Conv2D(128, kernel_size=3, activation='relu', input_shape=(100,100,3)),
        layers.Conv2D(128, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(7, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adadelta(0.001),
                  loss='categorical_crossentropy',
                 metrics=[tf.keras.metrics.categorical_accuracy])

    return model

