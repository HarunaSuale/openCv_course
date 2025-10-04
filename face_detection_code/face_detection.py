import cv2 as cv



path = "./resources/photos/group 1.jpg"

pic = cv.imread(path)
#cv.imshow("normal pic", pic)

gray_pic = cv.cvtColor(pic, cv.COLOR_BGR2GRAY)
#cv.imshow('gray_image', gray_pic)

haar_face_detect = cv.CascadeClassifier('./faces.xml')
face_detect_rec = haar_face_detect.detectMultiScale(gray_pic, scaleFactor=1.1, minNeighbors=3)
print(f'Number of faces detected = {len(face_detect_rec)}')

for (x, y, w, h) in face_detect_rec:
    face = cv.rectangle(pic, (x, y), (x+w, y+h), (0, 255, 0), thickness=3)

cv.imshow("faces detected", face)

cv.waitKey(0)