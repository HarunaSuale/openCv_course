import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('./faces.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

haar_face_detect = cv.face.LBPHFaceRecognizer_create()
haar_face_detect.read('./face_trained.yml')

img = cv.imread(r'C:\Users\Haruna Suale\Desktop\Computer Vision Projects\Resources\Faces\train\Elton John\7.jpg')
# cv.imshow('alex', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('bro', gray)

# Dectect faces in the img
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for(x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+h]
    label, confidence = haar_face_detect.predict(faces_roi)
    print(f'predicted label= {people[label]} and the confidence is = {confidence}')
    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)




cv.imshow("Detected face", img)
cv.waitKey(0)
