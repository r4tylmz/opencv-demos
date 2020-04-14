import cv2
import numpy as np

#cap = cv2.VideoCapture(R"C:\Users\rain\Desktop\OPENCV TEST VIDEOS\b4.mp4")
cap = cv2.VideoCapture(0)

first_frame = None
#dilate_kernel = np.ones((8,8),np.uint8)

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
    cnts,_ = cv2.findContours(threshold_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    final = np.zeros(frame.shape,np.uint8)
    mask = np.zeros(threshold_delta.shape,np.uint8)

    if len(cnts) > 0:
        c = max(cnts,key=cv2.contourArea)
        (x,y,w,h) = cv2.boundingRect(c)
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)


    cv2.imshow('frame',frame)
    cv2.imshow('threshold',threshold_delta)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    
cap.release()
cv2.destroyAllWindows()  