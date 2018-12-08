import cv2
"""I don't know why, but the program just stop working"""
# from time import sleep
import matplotlib.pyplot as plt
print("1")
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)


while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.03,
        minNeighbors=5,
        minSize=(150,150)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cropped = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()