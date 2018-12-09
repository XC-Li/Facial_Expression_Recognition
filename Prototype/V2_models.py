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


# Adding the maxpooling layer
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

# Modified the filter number
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

# Adding the dense and dropout layers
# Accuracy around 66.62% (epoch=100)
def fourth_model():
    model = tf.keras.Sequential([
        layers.Conv2D(128, kernel_size=3, activation='relu', input_shape=(100,100,3)),
        layers.Conv2D(128, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.1),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(7, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adadelta(0.001),
                  loss='categorical_crossentropy',
                 metrics=[tf.keras.metrics.categorical_accuracy])

    return model

# Modified with Adam optimizer
# Accuracy around 96%/68.84% (epoch=100)
def fifth_model():
    model = tf.keras.Sequential([
        layers.Conv2D(128, kernel_size=3, activation='relu', input_shape=(100,100,3)),
        layers.Conv2D(128, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.1),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(7, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy',
                 metrics=[tf.keras.metrics.categorical_accuracy])

    return model

# Modified with Adadelta optimizer
# Accuracy around 99%/74% (epoch=100)
def Adadelta_model():
    model = tf.keras.Sequential([
        layers.Conv2D(128, kernel_size=3, activation='relu', input_shape=(100,100,3)),
        layers.Conv2D(128, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.1),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(7, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adadelta(lr=1.0),
                  loss='categorical_crossentropy',
                 metrics=[tf.keras.metrics.categorical_accuracy])

    return model

# Modified with RMSprop optimizer
# Accuracy around 95%/67% (epoch=100)
def rmsprop_model():
    model = tf.keras.Sequential([
        layers.Conv2D(128, kernel_size=3, activation='relu', input_shape=(100,100,3)),
        layers.Conv2D(128, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.1),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(7, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
                  loss='categorical_crossentropy',
                 metrics=[tf.keras.metrics.categorical_accuracy])

    return model

# Modified with SGD optimizer
# Accuracy around 99%/73% (epoch=100)
def sgd_model():
    model = tf.keras.Sequential([
        layers.Conv2D(128, kernel_size=3, activation='relu', input_shape=(100,100,3)),
        layers.Conv2D(128, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.1),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(7, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01),
                  loss='categorical_crossentropy',
                 metrics=[tf.keras.metrics.categorical_accuracy])

    return model