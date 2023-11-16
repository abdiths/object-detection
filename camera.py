import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Camera is not openning!!")
    exit()
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame! Exiting")
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv.waitKey(1) ==  ord('q'):
        break

# when everything done, release the capture
cap.release()
cv.destroyAllWindows()
