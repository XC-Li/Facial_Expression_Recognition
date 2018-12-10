"""
Keras models
By: Xiaochi (George) Li
Dec.2018

Unfortunately, non of them can work.
Last Error:FailedPreconditionError: Attempting to use uninitialized value training/TFOptimizer/beta1_power
"""
import tensorflow as tf
from keras import layers
from keras import optimizers
from keras.applications import inception_v3
from keras.applications import mobilenet
from keras.models import Model
from keras import backend as K
from keras import metrics



def first_model():
    """
    This model just makes sure that the code can work
    :return:
        model: keras model object
    """
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


def ft_inception_v3():
    """
    Use pretrained VGG16 as bottom

    :return:
    """
    base_model = inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
    base_out = base_model.output
    x = layers.Flatten()(base_out)
    # 'Concatenate' object has no attribute 'outbound_nodes'
    x = layers.Dense(4096, activation='relu')(x)  # Can't connect
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    predictions = layers.Dense(7, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers():
        layer.trainable = False

    model.compile(optimizer=tf.train.AdadeltaOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=[metrics.categorical_accuracy])

    return model

def ft_mobile_net():
    """
    Use pretrained mobile net as bottom

    :return:
    """
    # input = layers.Input((224,224,3), name='RGB')
    base_model = mobilenet.MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224,3))
    base_out = base_model.output
    x = layers.Flatten()(base_out)
    x = layers.Dense(1024, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros', name='Dense1024')(x)
    x = layers.Dense(256, activation='relu',kernel_initializer='random_uniform', bias_initializer='zeros', name='Dense256')(x)
    x = layers.Dense(64, activation='relu',kernel_initializer='random_uniform', bias_initializer='zeros', name='Dense64')(x)
    x = layers.Dense(16, activation='relu',kernel_initializer='random_uniform', bias_initializer='zeros', name='Dense16')(x)
    predictions = layers.Dense(7, activation='softmax',kernel_initializer='random_uniform', bias_initializer='zeros')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    with tf.Session() as sess:
        adam = tf.train.AdamOptimizer()
        print(adam.variables())
        sess.run(tf.variables_initializer(adam.variables()))
        model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=[metrics.categorical_accuracy])
        # model.compile(optimizer=sgd,
        #               loss='categorical_crossentropy',
        #               metrics=[tf.keras.metrics.categorical_accuracy])
        K.set_session(tf.Session(graph=model.output.graph))
        init = K.tf.global_variables_initializer()
        K.get_session().run(init)

    # tensorflow.python.framework.errors_impl.FailedPreconditionError: Attempting to use uninitialized value training/TFOptimizer/beta1_power
        return model
