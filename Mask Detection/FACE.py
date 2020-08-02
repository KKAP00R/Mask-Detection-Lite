import cv2
import numpy as np
import imutils
import time

video = cv2.VideoCapture(0) #INITIALISING WEBCAM
loop, img = video.read() #DECLARING VARIABLE TO CAPTURE FRAME
face_library = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
eye_library = cv2.CascadeClassifier('data/haarcascades/haarcascade_righteye_2splits.xml')
mouth_library = cv2.CascadeClassifier('data/haarcascades/haarcascade_smile.xml')
#ALLOWING CAMERA TO WARM UP AND USER TO PREPARE THE OBJECT FOR DETECTION
time.sleep(2)

while loop:
    #CONTINUING EXTRACTION OF FRAMES IN A LOOP
    loop, img = video.read()
    #FLIPPING IMAGE TO MAKE IT STRAIGHT AS PER THE USER
    flipped = cv2.flip(img, 1)
    #PREPROCESSING ON THE IMAGE TO SHARPEN THE BALL, PUT THE BALL IN FOCUS
    img_sized = imutils.resize(flipped, width = 800)
    img_gray = cv2.cvtColor(img_sized, cv2.COLOR_BGR2GRAY)
    output = img_sized.copy()
    faces = face_library.detectMultiScale(img_gray, scaleFactor = 1.2, minNeighbors = 5)
    eyes = eye_library.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 5)
    mouth = mouth_library.detectMultiScale(img_gray, scaleFactor = 2, minNeighbors = 22)
    face_no = len(faces)
    eye_no = len(eyes)
    mouth_no = len(mouth)
    if eye_no is not None:
        if mouth_no != 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(output,'NO MASK',(x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255),2, cv2.LINE_AA)
        if mouth_no == 0:
            for (x, y, w, h) in eyes:
                width1 = 2*w
                height1 = 2*w
                width2 = 4*w
                height2 = 4*h
                cv2.rectangle(output, (x - width1, y - height1), (x + width2, y + height2), (0, 255, 0), 2)
                cv2.putText(output,'MASK',(x - width1, y - height1), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2, cv2.LINE_AA)
        cv2.imshow("OUTPUT", np.hstack([output]))
    else:
        cv2.imshow("OUTPUT", output)
    key = cv2.waitKey(1)
    #USER CAN END THE PROGRAM BY PRESSING THE ESCAPE KEY
    if key == 27: 
        break
video.release()
cv2.destroyAllWindows()