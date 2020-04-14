import numpy as np 
import cv2 
  
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilate_kernel = np.ones((5,5),np.uint8)
knn_subtractor = cv2.createBackgroundSubtractorKNN()
  
#cap = cv2.VideoCapture(R"C:\Users\rain\Desktop\OPENCV TEST VIDEOS\b4.mp4")
cap = cv2.VideoCapture(0)
while(1): 
    ret, img = cap.read(); 
    img = cv2.resize(img, dsize=(640,480))
    mask = knn_subtractor.apply(img)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask,(13,13),0)
    mask = cv2.dilate(mask,dilate_kernel,iterations=3)

    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    

    for cnt in contours:
        contour_area = cv2.contourArea(cnt)
        if contour_area > 5000:
            x,y,w,h = cv2.boundingRect(cnt)
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(img,format("x:%d, y:%d" % (x,y)),(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0))
            cv2.putText(img,format("area:%d" % contour_area),(x,y+30),cv2.FONT_HERSHEY_DUPLEX,1,(222,245,0))

    cv2.imshow('KNN', mask)
    cv2.imshow('frame', img)
      
    k = cv2.waitKey(1) & 0xff
    if k == 27: 
        break 
  
cap.release()
cv2.destroyAllWindows()