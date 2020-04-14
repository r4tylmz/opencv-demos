import cv2
import numpy as np

cap = cv2.VideoCapture(R"C:\Users\rain\Desktop\OpenCV\car.mp4")

# aldigimiz sonuclari iyilestirmek icin gri tona cevirip blur uygulayacagiz
# ilk frame ile sonrakileri karsilastirip kazima islemi yapmayi deneyecegiz

_,f_frame = cap.read()
f_frame = cv2.resize(f_frame,(640,480))

f_gray = cv2.cvtColor(f_frame,cv2.COLOR_BGR2GRAY)
f_gray = cv2.GaussianBlur(f_gray,(5,5),0)

while 1:
    _,frame = cap.read()
    frame = cv2.resize(frame,(640,480))

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)

    # first gray ile anlik degisen gray arasindaki farki bulacak
    diff = cv2.absdiff(f_gray,gray)
    _,diff = cv2.threshold(diff,25,255,cv2.THRESH_BINARY)

    cv2.imshow("f",frame)
    cv2.imshow("d",diff)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
