"""
Different MLP
By: Jia Chen
Dec 2018
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from keras import optimizers

hidden1_num = 50
hidden2_num = 50
hidden3_num = 50
hidden4_num = 50
hidden5_num = 50
output_num = 7


# --------------------------------- m_1 ------------------------------
def m_1():
    model = tf.keras.Sequential([
        layers.Dense(hidden1_num, activation='relu', input_dim=30000),
        layers.Dense(output_num, activation='softmax'),
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=[tf.keras.metrics.categorical_accuracy])

    return model


# --------------------------------- m_2 ------------------------------
def m_2():
    model = tf.keras.Sequential([
        layers.Dense(hidden1_num, activation='relu', input_dim=30000),
        layers.Dense(output_num, activation='softmax'),
    ])

    model.compile(optimizer=keras.optimizers.SGD(lr=0.001),
                  loss='categorical_crossentropy',
                 metrics=[tf.keras.metrics.categorical_accuracy])

    return model


# --------------------------------- m_3 ------------------------------
def m_3():
    model = tf.keras.Sequential([
        layers.Dense(hidden1_num, activation='relu', input_dim=30000),
        layers.Dropout(0.2),
        layers.Dense(output_num, activation='softmax'),
    ])

    model.compile(optimizer=keras.optimizers.SGD(lr=0.001),
                  loss='categorical_crossentropy',
                 metrics=[tf.keras.metrics.categorical_accuracy])

    return model


# --------------------------------- m_4------------------------------
def m_4():
    model = tf.keras.Sequential([
        layers.Dense(hidden1_num, activation='relu', input_dim=30000),
        layers.Dropout(0.2),
        layers.Dense(output_num, activation='softmax'),
    ])

    model.compile(optimizer=keras.optimizers.SGD(lr=0.01),
                  loss='categorical_crossentropy',
                 metrics=[tf.keras.metrics.categorical_accuracy])

    return model


# --------------------------------- m_5------------------------------
def m_5():
    model = tf.keras.Sequential([
        layers.Dense(50, activation='relu', input_dim=30000),
        layers.Dropout(0.2),
        layers.Dense(10, activation='relu'),
        layers.Dropout(0.2),
        # layers.Dense(10, activation='relu'),
        # layers.Dropout(0.2),
        # layers.Dense(10, activation='relu'),
        # layers.Dropout(0.2),
        layers.Dense(output_num, activation='softmax'),
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                 metrics=[tf.keras.metrics.categorical_accuracy])

    return model


# --------------------------------- m_6 ------------------------------
def m_6():
    model = tf.keras.Sequential([
        layers.Dense(50, activation='relu', input_dim=30000),
        layers.Dropout(0.2),
        layers.Dense(10, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(output_num, activation='softmax'),
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                 metrics=[tf.keras.metrics.categorical_accuracy])

    return model


# --------------------------------- m_7------------------------------
def m_7():
    model = tf.keras.Sequential([
        layers.Dense(10, activation='relu', input_dim=30000),
        layers.Dropout(0.2),
        layers.Dense(10, activation='relu'),
        layers.Dropout(0.2),
        # layers.Dense(10, activation='relu'),
        # layers.Dropout(0.2),
        # layers.Dense(10, activation='relu'),
        # layers.Dropout(0.2),
        layers.Dense(output_num, activation='softmax'),
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                 metrics=[tf.keras.metrics.categorical_accuracy])

    return model


# --------------------------------- m_8------------------------------
def m_8():
    model = tf.keras.Sequential([
        layers.Dense(10, activation='relu', input_dim=30000),
        layers.Dropout(0.2),
        layers.Dense(10, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(output_num, activation='softmax'),
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                 metrics=[tf.keras.metrics.categorical_accuracy])

    return model


# --------------------------------- m_9 ------------------------------
def m_9():
    model = tf.keras.Sequential([
        layers.Dense(100, activation='relu', input_dim=30000),
        layers.Dense(output_num, activation='softmax'),
    ])

    model.compile(optimizer=keras.optimizers.SGD(lr=0.001),
                  loss='categorical_crossentropy',
                 metrics=[tf.keras.metrics.categorical_accuracy])

    return model





