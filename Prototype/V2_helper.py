"""
Prototype V2 Helper
label loader, data loader and plot for accuracy and loss
By: Xiaochi (George) Li
Nov.2018
"""
from PIL import Image
import numpy as np
import time


def label_loader(label_file):
    train_img_label_pair = {}
    test_img_label_pair ={}
    with open(label_file) as all_label:
        for label in all_label:
            label = label.rstrip()
            if label.startswith("train"):
                train_img_label_pair[label.split(" ")[0][:-4]+"_aligned.jpg"] = int(label.split(" ")[1])
            if label.startswith("test"):
                test_img_label_pair[label.split(" ")[0][:-4]+"_aligned.jpg"] = int(label.split(" ")[1])
    return train_img_label_pair, test_img_label_pair


def load_to_numpy(img_label_pair, folder, shape):
    width = shape[0]
    height = shape[1]
    limit = len(img_label_pair)
    labels = np.zeros((limit, 7))
    imgs = np.empty((limit, width, height, 3))

    i = 0
    for image_name in img_label_pair:
        img = Image.open(folder + image_name).convert('RGB')
        img = img.resize((width, height))
        img = np.array(img).reshape((width, height, 3))
        # important: normalize to [0,1]
        img = img/255
        # imgs = np.append(imgs, img, axis=0)
        imgs[i] = img # faster approach! learning algorithm is useful
        # labels[i] = img_label_pair[image_name]
        labels[i, img_label_pair[image_name]-1] = 1
        i += 1
    return imgs, labels


def plot(history, log_name, num_epoch):
    import matplotlib.pyplot as plt
    import os
    if not os.path.exists("./imgs"):
        print("First run, make dir")
        os.makedirs("./imgs")
    plt.plot(np.linspace(1, num_epoch, num_epoch),
             np.array(history.history["categorical_accuracy"]), label='Accuracy', color='b')
    plt.plot(np.linspace(1, num_epoch, num_epoch),
             np.array(history.history["val_categorical_accuracy"]), label='Validation Accuracy', color='r')
    plt.legend()
    plt.title("Accuracy" + log_name + time.ctime())
    plt.savefig("./imgs/Accuracy " + log_name + " " + time.ctime())
    # plt.show()
    plt.close()
    plt.plot(np.linspace(1, num_epoch, num_epoch), np.array(history.history["loss"]), label='Loss', color='b')
    plt.plot(np.linspace(1, num_epoch, num_epoch),
             np.array(history.history["val_loss"]), label='Validation Loss', color='r')
    plt.legend()
    plt.title("Loss" + log_name + time.ctime())
    plt.savefig("./imgs/Loss " + log_name + " " + time.ctime())
    # plt.show()
    plt.close()