##importing required libraries
import cv2 as cv
import numpy as np 
import os

##loading face haar cascade
face_haar = cv.CascadeClassifier('face_haar.xml')

##list of avengers' cast
avengers = ['Benedict Cumberbatch', 'Chadwick Boseman', 'Chris Evans', 'Chris Hemsworth', 'Elizabeth Olsen', 'Robert Downey Jr', 'Scarlett Johansson', 'Tom Holland']

##directory where images are saved for training
DIR = r'C:\Users\Admin\Face Recognition\images'

##empty lists
features = []
labels = []

##defining a function that preprocesses the data 
def preprocess():
    for avenger in avengers: #accessing individual person's folder
        path = os.path.join(DIR, avenger)
        label = avengers.index(avenger)

        for img in os.listdir(path): #accessing each image from an individual's folder
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path) #reading the image
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY) #converting to gray scale

            face_rect = face_haar.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3) #detecting faces
            for (x,y,w,h) in face_rect: 
                face_roi = gray[y: y + h, x: x + w]  #cropping region of interest (faces)
                features.append(face_roi)   #appending the roi 
                labels.append(label)        #appending the corresponding label for roi

##calling the function
preprocess()
features = np.array(features, dtype = 'object') #
labels = np.array(labels)

##training
face_train = cv.face.LBPHFaceRecognizer_create()
face_train.train(features, labels)
face_train.save('face_train.yml')
print('Training done..........')

print('no of features: ', len(features))
print('no of labels: ', len(labels))