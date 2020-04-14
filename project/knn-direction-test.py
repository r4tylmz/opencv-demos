import numpy as np 
import cv2 
from collections import deque
  
dilate_kernel = np.ones((5,5),np.uint8)
knn_subtractor = cv2.createBackgroundSubtractorKNN()
  
#cap = cv2.VideoCapture(R"C:\Users\rain\Desktop\OPENCV TEST VIDEOS\b4.mp4")
cap = cv2.VideoCapture(0)
(dX, dY) = (0, 0)
direction = ''
buffer = 5
counter = 0
distance = 3
pts = deque(maxlen = buffer)

while(1): 
    ret, frame = cap.read(); 
    frame = cv2.resize(frame, dsize=(640,480))
    # frame = cv2.flip(frame,1)
    mask = knn_subtractor.apply(frame)
    mask = cv2.GaussianBlur(mask,(13,13),0)
    mask = cv2.dilate(mask,dilate_kernel,iterations=3)

    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    center = None

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
        if counter >= 10 and i == 1 and pts[-2] is not None:
            # su an ki noktayla buffer'in sonundaki nokta arasini hesaplar
            dX = pts[-buffer][0] - pts[i][0]
            dY = pts[-buffer][1] - pts[i][1]
            (dirX, dirY) = ('', '')

            
            if np.abs(dX) > distance:
                #dirX = "Dogu" if np.sign(dX) == 1 else "Bati"
                dirX = "Bati" if np.sign(dX) == 1 else "Dogu"
            if np.abs(dY) > distance:
                dirY = "Kuzey" if np.sign(dY) == 1 else "Guney"
            if dirX != "" and dirY != "":
                direction = "{}-{}".format(dirY, dirX)
            else:
                direction = dirX if dirX != "" else dirY
        thickness = int(np.sqrt(buffer / float(i + 1)) * 1.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.putText(frame, direction, (20,40), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,255), 3)

    cv2.imshow('direction', frame)
    cv2.imshow('mask', mask)
    key = cv2.waitKey(10) & 0xFF
    counter += 1

    #If q is pressed, close the window
    if(key == ord('q')):
        break
  
cap.release()
cv2.destroyAllWindows()