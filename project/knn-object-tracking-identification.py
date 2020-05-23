import random
import time
from collections import OrderedDict, deque
import cv2
import numpy as np
from scipy.spatial import distance as dist


class CentroidTracker():
    def __init__(self, maxDisappeared=30):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids, metric="braycurtis")

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()


            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects

def get_random_color():
    r = random.randint(0,random.randint(100,255))
    g = random.randint(0,random.randint(120,255))
    b = random.randint(0,random.randint(150,255))
    return (b,g,r)

def filter_mask(frame):
    dilate_kernel = np.ones((5,5),np.uint8)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = knn_subtractor.apply(frame)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,morph_kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,morph_kernel)
    mask = cv2.GaussianBlur(mask,(7,7),0)
    mask = cv2.dilate(mask,dilate_kernel,iterations=2)
    return mask

def get_direction_name(dX,dY):
    (dirX, dirY) = ("", "")

    if np.abs(dX) > 10:
        dirX = "Bati" if np.sign(dX) == 1 else "Dogu"
    if np.abs(dY) > 10:
        dirY = "Kuzey" if np.sign(dY) == 1 else "Guney"
    
    if dirX != "" and dirY != "":
        direction_name = "{}-{}".format(dirY, dirX)
    else:
        direction_name = dirX if dirX != "" else dirY

    return direction_name

knn_subtractor = cv2.createBackgroundSubtractorKNN()
  
cap = cv2.VideoCapture(R"C:\Users\ylmz\Desktop\OPENCV TEST VIDEOS\test4.mp4")
# cap = cv2.VideoCapture(0)
time.sleep(1)
ct = CentroidTracker()
queue = OrderedDict()
colorDict = OrderedDict()
directionDict = OrderedDict()


while(1):

    ret, frame = cap.read(); 
    if ret is False:
        break
    frame = cv2.resize(frame, (640,480))
    # frame = cv2.flip(frame,1)
    mask = filter_mask(frame)
    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rects = []

    for cnt in contours:
        if cv2.contourArea(cnt) >= 3000:
            (startX, startY, endX, endY) = np.array(cv2.boundingRect(cnt)).astype("int")
            box = np.array([startX, startY, (startX+endX), (startY+endY)])
            rects.append(box.astype(int))
            cv2.rectangle(frame, (startX, startY), (startX+endX, startY+endY),(255, 0, 0), 2)
        
    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():
        if objectID not in queue:
            queue[objectID] = deque(maxlen=32)
            colorDict[objectID] = get_random_color()
        queue[objectID].appendleft((centroid[0],centroid[1]))

        direction_name = ""
        if len(queue[objectID]) >= 5:
            lx,ly= queue[objectID][-1]
            x,y= queue[objectID][1]
            direction_name = get_direction_name(lx-x,ly-y)

        text = f"id: {objectID}"
        coordinates = f"{centroid}"
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(frame, direction_name, (centroid[0]-10, centroid[1] + 20),cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        for i in range(1,len(queue[objectID])):
            cv2.line(frame,queue[objectID][i-1],queue[objectID][i],colorDict[objectID],2)

    cv2.imshow('frame', frame)
    cv2.imshow('mask',mask)

    if(cv2.waitKey(25) & 0xFF == ord('q')):
        break

cv2.destroyAllWindows()
cap.release()
