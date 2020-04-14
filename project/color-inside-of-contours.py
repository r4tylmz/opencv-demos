import cv2
import numpy as np
from collections import deque

#cap = cv2.VideoCapture(R"C:\Users\rain\Desktop\OPENCV TEST VIDEOS\b4.mp4")
cap = cv2.VideoCapture(0)

first_frame = None
#dilate_kernel = np.ones((8,8),np.uint8)
(dX, dY) = (0, 0)
direction = ''
buffer = 30
counter = 0
pts = deque(maxlen = buffer)
while 1:
    _,frame = cap.read()
    frame = cv2.resize(frame,(640,480))
    frame = cv2.flip(frame,flipCode=1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    
    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame,gray)
    threshold_delta = cv2.threshold(delta_frame,60,255,cv2.THRESH_BINARY)[1]
    threshold_delta = cv2.erode(threshold_delta,None,iterations=4)
    threshold_delta = cv2.dilate(threshold_delta,None,iterations=4)
    contours,_ = cv2.findContours(threshold_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    final = np.zeros(frame.shape,np.uint8)
    mask = np.zeros(threshold_delta.shape,np.uint8)


    if len(contours) > 0:
        c = max(contours, key = cv2.contourArea)
        ((rx, ry), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if radius > 20:
            cv2.circle(frame, (int(rx), int(ry)), int(radius), (0,255,255), 2)
            cv2.circle(frame, center, 5, (0,255,255), -1)
            pts.append(center)
    for i in np.arange(1, len(pts)):
        #If no points are detected, move on.
        if(pts[i-1] == None or pts[i] == None):
            continue

        #If atleast 10 frames have direction change, proceed
        if counter >= 2 and i == 1 and pts[-2] is not None:
            #Calculate the distance between the current frame and 10th frame before
            dX = pts[-2][0] - pts[i][0]
            dY = pts[-2][1] - pts[i][1]
            (dirX, dirY) = ('', '')

            #If distance is greater than 100 pixels, considerable direction change has occured.
            if np.abs(dX) > 20:
                dirX = "Dogu" if np.sign(dX) == 1 else "Bati"
            # ensure there is significant movement in the
            # y-direction
            if np.abs(dY) > 20:
                dirY = "Kuzey" if np.sign(dY) == 1 else "Guney"
            # handle when both directions are non-empty
            if dirX != "" and dirY != "":
                direction = "{}-{}".format(dirY, dirX)
            # otherwise, only one direction is non-empty
            else:
                direction = dirX if dirX != "" else dirY

        #Draw a trailing red line to depict motion of the object.
        thickness = int(np.sqrt(buffer / float(i + 1)) * 1.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    #Write the detected direction on the frame.
    cv2.putText(frame, direction, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    #Show the output frame.
    cv2.imshow('direction test', frame)
    key = cv2.waitKey(20) & 0xFF
    #Update counter as the direction change has been detected.
    counter += 1

    if(key == ord('q')):
        break
  

    # for i in range(0,len(contours)):
    #     contour_area = cv2.contourArea(contours[i])
    #     cv2.imshow('mask',mask)
    #     if contour_area > 5000:
    #         #mask = np.zeros(diff.shape, dtype="uint8")
    #         #cv2.drawContours(mask,contours,i,255,-1)
    #         #mean = cv2.mean(diff,mask)
    #         #cv2.drawContours(final,contours,i,mean,-1)
    #         x,y,w,h = cv2.boundingRect(contours[i])
    #         frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)


    #cv2.imshow('frame',frame)
    #cv2.imshow('final',final)
    #cv2.imshow('diff',diff)

      
cap.release()
cv2.destroyAllWindows()  
            