import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('/Users/calap/Desktop/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/calap/Desktop/opencv-3.1.0/data/haarcascades/haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
#count = 0
while True:
    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor =1.3,
            minNeighbors = 10,
            minSize=(30,30)
        )
 
    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
              cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0, 255, 0), 2)
            
  
    cv2.imshow('video',frame)
    k = cv2.waitKey(30) & 0xFF == ord('q')
    if  k ==27:
 #   if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 #   count = count+1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
