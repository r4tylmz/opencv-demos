import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def findMaxContour(contours):
    max_i = 0
    max_area =0
    for i in range(len(contours)):
        area_face = cv2.contourArea(contours[i])
        
        if max_area<area_face:
            max_area=area_face
            max_i = i
        try:
            c = contours[max_i]            
        except:
            contours = [0]
            c = contours[0]
        return c  


while 1:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    # frame[y1:y2,x1:x2]
    roi = frame[50:250,250:450]

    cv2.rectangle(frame,(250,50),(450,250),(0,0,255),0)

    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    lower_color = np.array([0,45,79],dtype=np.uint8)
    upper_color = np.array([17,255,255],dtype=np.uint8)

    mask = cv2.inRange(hsv,lower_color,upper_color)
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations=1)
    mask = cv2.medianBlur(mask,15)

    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        try:
            c = findMaxContour(contours)
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])

            cv2.line(roi,extLeft,extTop,(255,0,0),2)
            cv2.line(roi,extTop,extRight,(255,0,0),2)
            cv2.line(roi,extRight,extBot,(255,0,0),2)
            cv2.line(roi,extBot,extLeft,(255,0,0),2)

            cv2.circle(roi,extLeft,5,(0,255,0),2)
            cv2.circle(roi,extRight,5,(0,255,0),2)
            cv2.circle(roi,extTop,5,(0,255,0),2)
        except:
            pass
    else:
        pass

    cv2.imshow("frame",frame)
    cv2.imshow("roi",roi)
    cv2.imshow("roi2",mask)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()