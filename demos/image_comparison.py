import cv2
import numpy as np

img = cv2.imread(R"C:\Users\rain\Desktop\OpenCV\aircraft.jpg")
img = cv2.resize(img,(640,480))
img2 = cv2.imread(R"C:\Users\rain\Desktop\OpenCV\palette.jpg")
img2 = cv2.resize(img2,(640,480))
diff = cv2.subtract(img,img2)
b,g,r = cv2.split(diff)


if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r):
    print("equal")

cv2.imshow("Aircraft",img)
cv2.imshow("diff",diff)
cv2.waitKey(0)
cv2.destroyAllWindows() 