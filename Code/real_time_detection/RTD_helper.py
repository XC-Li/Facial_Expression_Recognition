"""
Real Time Facial Expression Detection Helper Function
By: Xiaochi (George) Li
Dec.2018

"""

from tensorflow.keras.models import load_model
from tensorflow.logging import set_verbosity
from tensorflow.logging import ERROR
import numpy as np

def prediction(img):
    """
    prediction function
    By: Xiaochi (George) Li
    Dec.2018
    :parameter
        img: size(100,100,3)
    :return
        predict: String of best predict
        detail: dict of probabilty of all emotions
    """
    model_dir = './'
    model_file = 'lightvgg2.h5'

    set_verbosity(ERROR)
    label2expression = {1: "Surprise", 2: "Fear", 3: "Disgust", 4: "Happiness",
                        5: "Sadness", 6: "Anger", 7: "Neutral"}
    model = load_model(model_dir + model_file)

    img = np.array(img).reshape((1, 100, 100, 3))
    img = img / 255

    prediction = model.predict(img)
    detail = {}
    for i in range(7):
        detail[label2expression[i+1]] = prediction[0][i]
    prediction_max = np.argmax(prediction)
    predict = label2expression[prediction_max + 1]
    return predict, detail
