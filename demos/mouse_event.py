import cv2
import numpy as np

cap = cv2.VideoCapture(R"C:\Users\rain\Desktop\OpenCV\line.mp4")
circles = []
def mouse(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        circles.append((x,y))


cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame",mouse)

while 1:
    _,frame = cap.read()
    frame = cv2.resize(frame,(640,480))
    for center in circles:
        cv2.circle(frame,center,20,(0,255,0),-1)

    cv2.imshow("Frame",frame)

    key = cv2.waitKey(10)

    if key == 27:
        break
    elif key == ord("h"):
        circles = []

cap.release()
cv2.destroyAllWindows()