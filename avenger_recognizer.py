##importing required libraries
import cv2 as cv
import numpy as np
import os

#loading face haar cascade
face_haar = cv.CascadeClassifier('face_haar.xml')

#list of avengers' cast
avengers = ['Benedict Cumberbatch', 'Chadwick Boseman', 'Chris Evans', 'Chris Hemsworth', 'Elizabeth Olsen', 'Robert Downey Jr', 'Scarlett Johansson', 'Tom Holland']

#loading trained yml file
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_train.yml')

#directory where validation images are saved
DIR = r'C:\Users\Admin\Face Recognition\val'

#function that recognizes faces
def test():
    k = 1
    for img in os.listdir(DIR): #accessing all validation images
        img_path = os.path.join(DIR, img)
        img = cv.imread(img_path)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  #converting to gray scale

        #cv.imshow('gray', gray)

        face_rect = face_haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7) #detecting faces
        #l = 0
        for (x,y,w,h) in face_rect: 
            face_roi = gray[y:y + h, x:x+w]  #cropping roi
            label, confidence = face_recognizer.predict(face_roi) #recognizing faces
            
            cv.putText(img, str(avengers[label]), (x-20,y-20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=3 ) #writing the recognized name
            cv.rectangle(img, (x,y), (x+w, y+h), (0, 255,0), thickness=2)  #drawing a rectangle around the faces
            #l = l + 20
            

        cv.imshow('avenger ' + str(k) , img)  #showing the image with recognized face
        k = k + 1

#calling the function
test()

cv.waitKey(0)