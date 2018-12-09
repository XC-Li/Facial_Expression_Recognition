#!/usr/bin/python3.5

"""
Prototype V2
Main Program
By: Xiaochi (George) Li
Nov.2018
"""


import V2_helper
import V2_models
import Xiaochi_models


num_epoch = 100
batch_size = 64
input_shape = (100, 100)

model = Xiaochi_models.light_vgg2()

log_name = input("What's the name of this run?:")

# create image label pair as a dictionary
label_file = "../RAFDB/list_patition_label.txt"
folder = "../RAFDB/aligned/"
train_img_label_pair, test_img_label_pair = V2_helper.label_loader(label_file)

# Use pickle to speed up reading data
import pickle
print("Loading Data")
try:
    train_img, train_label, test_img, test_label = pickle.load(open("processed_data_"+str(input_shape)+".pickle", 'rb'))
    print("Shape", train_img.shape)
except:
    train_img, train_label = V2_helper.load_to_numpy(train_img_label_pair, folder, input_shape)
    test_img, test_label = V2_helper.load_to_numpy(test_img_label_pair, folder, input_shape)
    print("Shape", train_img.shape)
    print("First time reading data, saving to pickle to speed up next time")
    try:
        pickle.dump((train_img, train_label, test_img, test_label), open("processed_data_"+str(input_shape)+".pickle", 'wb'))
    except:
        print("can't save, file too big")

# Keras model and Tensor Board
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time
tb = TensorBoard(log_dir="logs/" + log_name +" "+ time.ctime())
chk = ModelCheckpoint("./trained_model/"+log_name+"_chk.h5", monitor='val_categorical_accuracy', verbose=0,
                      save_best_only=True, save_weights_only=True, mode='max', period=1)
history = model.fit(train_img, train_label, epochs=num_epoch, batch_size=batch_size,
                    validation_data=(test_img, test_label), callbacks=[tb, chk])
model.summary()

V2_helper.plot(history, log_name, num_epoch)
import os
if not os.path.exists("./trained_model"):
    print("First run, make dir")
    os.makedirs("./trained_model")
model.save("./trained_model/"+log_name+".h5")
loss, accuracy = model.evaluate(test_img, test_label,batch_size=32)
print("test loss:", loss)
print("test accuracy:", accuracy)
