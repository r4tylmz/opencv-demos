import cv2
import numpy as np

def n(x):
    pass

cap = cv2.VideoCapture(R"C:\Users\rain\Desktop\OPENCV TEST VIDEOS\blue_ball.mp4")
# cap = cv2.VideoCapture(0)
cv2.namedWindow("Settings")


cv2.createTrackbar("lh","Settings",0,180,n)
cv2.createTrackbar("ls","Settings",0,255,n)
cv2.createTrackbar("lv","Settings",0,255,n)
cv2.createTrackbar("uh","Settings",0,180,n)
cv2.createTrackbar("us","Settings",0,255,n)
cv2.createTrackbar("uv","Settings",0,255,n)

while 1:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(640,480))
    frame = cv2.flip(frame,1)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lh = cv2.getTrackbarPos("lh","Settings")
    ls = cv2.getTrackbarPos("ls","Settings")
    lv = cv2.getTrackbarPos("lv","Settings")
    uh = cv2.getTrackbarPos("uh","Settings")
    us = cv2.getTrackbarPos("us","Settings")
    uv = cv2.getTrackbarPos("uv","Settings")


    lower_blue = np.array([lh,ls,lv])
    upper_blue = np.array([uh,us,uv])
    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    # mavinin mavi oldugunu gormek icin yapiyoruz
    bitwise = cv2.bitwise_and(frame,frame,mask=mask)
    cv2.imshow("f",frame)
    cv2.imshow("m",mask)
    cv2.imshow("b",bitwise)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()