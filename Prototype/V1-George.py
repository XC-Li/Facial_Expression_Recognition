import tensorflow as tf
print(tf.__version__)

import numpy as np

label2expression = {1: "Surprise",2: "Fear",3: "Disgust",4: "Happiness",
                    5: "Sadness", 6: "Anger",7: "Neutral"}

# create image label pair as a dictionary
label_file = "../RAFDB/list_patition_label.txt"
train_img_label_pair = {}
test_img_label_pair ={}
with open(label_file) as all_label:
    for label in all_label:
        label = label.rstrip()
        if label.startswith("train"):
            train_img_label_pair[label.split(" ")[0][:-4]+"_aligned.jpg"] = int(label.split(" ")[1])
        if label.startswith("test"):
            test_img_label_pair[label.split(" ")[0][:-4]+"_aligned.jpg"] = int(label.split(" ")[1])


folder = "../RAFDB/aligned/"

from PIL import Image
def load_to_numpy(img_label_pair):
    length = 100
    width = 100
    limit = len(img_label_pair)
    labels = np.zeros((limit, 7))
    imgs = np.empty((limit, length, width, 3))
    print("Loading Data:")

    i = 0
    for image_name in img_label_pair:
        img = Image.open(folder + image_name).convert('RGB')
        img = np.array(img).reshape((100,100,3))
        # imgs = np.append(imgs, img, axis=0)
        imgs[i] = img # faster approach! learning algorithm is useful
        #labels[i] = img_label_pair[image_name]
        labels[i, img_label_pair[image_name]-1] = 1
        i+=1

    one_hot_label = np.zeros(())
    return (imgs, labels)

train_img, train_label = load_to_numpy(train_img_label_pair)
test_img, test_label = load_to_numpy(test_img_label_pair)

# Keras model
from tensorflow.keras import layers
print(tf.keras.__version__)

model = tf.keras.Sequential([
    layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(100,100,3)),
    layers.Conv2D(32, kernel_size=3, activation='relu'),
    layers.Flatten(),
    layers.Dense(7, activation = 'softmax')
])

model.compile(optimizer=tf.train.AdadeltaOptimizer(0.001),
              loss='categorical_crossentropy',
             metrics=[tf.keras.metrics.categorical_accuracy])

num_epoch = 200

history = model.fit(train_img, train_label, epochs=num_epoch, batch_size=32)

import matplotlib.pyplot as plt
plt.plot(np.linspace(1, num_epoch, num_epoch), np.array(history.history["categorical_accuracy"]), label='Accuracy')
plt.legend()
plt.show()
plt.plot(np.linspace(1, num_epoch, num_epoch), np.array(history.history["loss"]), label='loss')
plt.legend()
plt.show()


loss, accuracy = model.evaluate(test_img, test_label,batch_size=32)
print("test loss:", loss)
print("test accuracy:", accuracy)