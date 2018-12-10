"""
Real Time Facial Expression Detection
By: Xiaochi (George) Li
Dec.2018
Structure based on https://github.com/shantnu/Webcam-Face-Detect/blob/master/webcam_cv3.py
"""

import cv2
import RTD_helper

from time import sleep
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)
print("press Q to quit")
sleep(0.5)


while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        pass

    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.03,
        minNeighbors=5,
        minSize=(150,150)
    )

    for (x, y, w, h) in faces:
        cropped = RGB_frame[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (100, 100), interpolation=cv2.INTER_AREA)
        predict, detail = RTD_helper.prediction(resized)
        print(detail)
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 0, 0)
        if predict in ['Happiness','Neutral']:
            color = (0, 255, 0)  # Green
        elif predict in ['Surprise', 'Fear', 'Disgust', 'Sadness']:
            color = (255, 0, 0)  # Blue
        else:
            color = (0, 0, 255)  # Red
        cv2.putText(frame, predict, (x + w, y), font, 1, color, 2, cv2.LINE_AA)
        cv2.putText(frame, str(detail[predict]), (x + w, y+25), font, 1, color, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('Expression Detection', frame)
    # sleep(0.5)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
