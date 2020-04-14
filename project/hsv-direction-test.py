import cv2
import numpy as np
from collections import deque
import time


#cap = cv2.VideoCapture(R"C:\Users\rain\Desktop\OPENCV TEST VIDEOS\b1.mp4")
cap = cv2.VideoCapture(0)
time.sleep(3)
(dX, dY) = (0, 0)
direction = ''
buffer = 30
counter = 0
distance = 15
lastframe = 5
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

pts = deque(maxlen = buffer)
while 1:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(640,480))
    #frame = cv2.flip(frame,1)
    blur = cv2.GaussianBlur(frame, (15,15), 0)
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 100, 60]) 
    upper_blue = np.array([255, 255, 180]) 

    # lower_blue = np.array([161, 155, 84])
    # upper_blue = np.array([179, 255, 255])

    blue = cv2.inRange(hsv,lower_blue,upper_blue)

    # morfolojik islemler ve dilate
    blue = cv2.erode(blue,None,iterations=2)
    blue = cv2.dilate(blue,None,iterations=2)
    #blue_result = cv2.bitwise_and(frame,frame,mask = blue)

    # mavi rengi takip etme
    contours,hierarchy = cv2.findContours(blue,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:

        c = max(contours, key = cv2.contourArea)
        ((rx, ry), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if radius > 10:
            cv2.circle(frame, (int(rx), int(ry)), int(radius), (0,255,255), 2)
            cv2.circle(frame, center, 5, (0,255,255), -1)
            cv2.putText(frame,format("x:%d, y:%d" % (int(rx), int(ry))),(int(rx), int(ry)),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0))
            pts.append(center)

    for i in np.arange(1, len(pts)):
        #If no points are detected, move on.
        if(pts[i-1] == None or pts[i] == None):
            continue

        #If atleast 10 frames have direction change, proceed
        if counter >= 10 and i == 1 and pts[-lastframe] is not None:
            # su an ki noktayla buffer'in sonundaki nokta arasini hesaplar
            dX = pts[-lastframe][0] - pts[i][0]
            dY = pts[-lastframe][1] - pts[i][1]
            (dirX, dirY) = ('', '')

            
            if np.abs(dX) > distance:
                dirX = "Dogu" if np.sign(dX) == 1 else "Bati"
                #dirX = "Bati" if np.sign(dX) == 1 else "Dogu"
            if np.abs(dY) > distance:
                #dirY = "Kuzey" if np.sign(dY) == 1 else "Guney"
                dirY = "Guney" if np.sign(dY) == 1 else "Kuzey"
            if dirX != "" and dirY != "":
                direction = "{}-{}".format(dirY, dirX)
            else:
                direction = dirX if dirX != "" else dirY
        thickness = int(np.sqrt(buffer / float(i + 1)) * 2)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.putText(frame, direction, (20,40), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,255), 3)

    cv2.imshow('direction', frame)
    cv2.imshow('blue',blue)
    key = cv2.waitKey(30) & 0xFF
    counter += 1
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()