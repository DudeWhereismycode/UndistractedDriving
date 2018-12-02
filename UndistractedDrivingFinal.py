import numpy as np
import cv2
import time as t
import math as m

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
lastTime = 0
setTime = False

def offset(ix,iy):
    xo = abs(firstFrame[0] - ix)
    yo = abs(firstFrame[1] - iy)
    return xo, yo
# from documentation function of screen capture that returns w & L
width = cap.get(3)
height = cap.get(4)
firstFrame = (width/2,height/2,0,0)
# we convert capture to grayscale to record data and apply functions from cascades
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for ex, ey, ew, eh in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (9, 255, 0), 2)


        if len(eyes) == 0:
            print("no eyes | " + t.ctime())
            t.sleep(0.1)
        avgx = x+(w/2)
        avgy = y+(h/2)
# if eyes travel more than 70 units from center of screen start counting
        if (not setTime) & (offset(avgx,avgy)[0] > 70):
            print("timer start")
            lastTime = t.process_time()
            setTime = True
            # if timer has been set and it has been 3 seconds or more start printing warings
        if setTime & (t.process_time() - lastTime > 7) & (offset(avgx,avgy)[0] > 70):
            print("PAY ATTENTION ! | xoff:"+str(offset(avgx,avgy)[0])+", yoff:"+str(offset(avgx,avgy)[1]))

        if setTime & (offset(avgx,avgy)[0] < 70):
            print("timer end")
            setTime = False
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
