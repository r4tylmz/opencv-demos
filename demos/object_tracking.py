import cv2
import numpy as np

cap = cv2.VideoCapture(R"C:\Users\rain\Desktop\OpenCV\dog.mp4")

while 1:
    check,frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # white icin deger araliklari hazir alindi
    sensitivity = 15
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])

    # white icin deger aralıklarında maskeleme uygulandı
    mask = cv2.inRange(hsv,lower_white,upper_white)

    res = cv2.bitwise_and(frame,frame,mask,mask)

    cv2.imshow("frame",frame)
    # cv2.imshow("masked",mask)
    cv2.imshow("result",res)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()