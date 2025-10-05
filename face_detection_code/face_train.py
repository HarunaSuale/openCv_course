import cv2 as cv
import numpy as np 
import os 

DIR = r'C:\Users\Haruna Suale\Desktop\Computer Vision Projects\Resources\Faces\train'
people = []
for i in os.listdir(r'C:\Users\Haruna Suale\Desktop\Computer Vision Projects\Resources\Faces\train'):
    people.append(i)


#haarcascade face detector
haar_face_detect = cv.CascadeClassifier('./faces.xml')

#print(people)
fearture = []
labels = []

def create_train():
    for person in people:
        # Get the path of each directory
        path = os.path.join(DIR, person)
        # Create a label for each person
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            # Now we can use image path to read imgs
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            face_detect_rec = haar_face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

            #crop the faces after detection
            for (x, y, w, h) in face_detect_rec:
                faces_roi = gray[y:y+h, x:x+w]
                
                fearture.append(faces_roi)
                labels.append(label)

create_train()
# print(f'number of faces = {len(fearture)}')
# print(f'Number of labels = {len(labels)}')
# print(labels)


print('Training ----------------------->')
fearture = np.array(fearture, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the model on the feature, labels
face_recognizer.train(fearture, labels)

face_recognizer.save('face_trained.yml')
np.save('feature.npy', fearture) 
np.save('label.npy', labels)

