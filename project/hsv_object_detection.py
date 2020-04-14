import cv2
import numpy as np


cap = cv2.VideoCapture(R"C:\Users\rain\Desktop\OPENCV TEST VIDEOS\b1.mp4")
# cap = cv2.VideoCapture(0)

while 1:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(640,480))
    frame = cv2.flip(frame,1)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 100, 60]) 
    upper_blue = np.array([255, 255, 180]) 

    blue = cv2.inRange(hsv,lower_blue,upper_blue)

    # morfolojik islemler ve dilate
    kernel = np.ones((5,5),"uint8")

    blue = cv2.dilate(blue,kernel)
    blue_result = cv2.bitwise_and(frame,frame,mask = blue)

    # mavi rengi takip etme
    contours,hierarchy = cv2.findContours(blue,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area>400:
            x,y,w,h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame,"blue",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0))
    
    cv2.imshow("b",frame)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()