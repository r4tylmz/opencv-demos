import cv2
import numpy as np

cap = cv2.VideoCapture(R"C:\Users\rain\Desktop\OpenCV\dog.mp4")
subtractor = cv2.createBackgroundSubtractorMOG2(history=10,varThreshold=100,detectShadows=True)
_,first_frame = cap.read()
first_frame = cv2.resize(first_frame,(640,480))
while 1:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(640,480))
    diff = cv2.absdiff(first_frame,frame)
    mask = subtractor.apply(diff)
    cv2.imshow("m",mask)
    cv2.imshow("f",frame)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()