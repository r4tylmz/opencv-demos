import random
import time
from collections import OrderedDict, deque

import cv2
import numpy as np
from scipy.spatial import distance as dist
import math as m


class TrackableObject():
    def __init__(self,objectID,last_coordinate,last_direction_name):
        self.objectID = objectID
        self.last_coordinate = last_coordinate
        self.last_direction = last_direction_name

class CentroidTracker():
    def __init__(self, maxDisappeared=30):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.disappeared_list = []
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        trackable_object = TrackableObject(objectID,self.objects[objectID],None)
        self.disappeared_list.append(trackable_object)
        del self.objects[objectID]
        del self.disappeared[objectID]

    def get_disappeared_list(self):
        return self.disappeared_list

    def closest_node(self,node, nodes):
        closest_index = dist.cdist([node], np.array(nodes)).argmin()
        return closest_index,nodes[closest_index]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        inputRects = np.zeros((len(rects), 4), dtype="int")
        dis_list = OrderedDict()

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            inputRects[i] = (startX, startY, endX, endY)
        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # objelerin merkezlerinin tutuldugu liste ile
            # varolan objelerin merkezlerine olan uzakligi bulmaya calisiyoruz
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                

                closest_index,closest_node,objectID = None,None,objectIDs[row]
                # kaybolan nesnelerin son koodinatlariyla yeni gelen koordinatlari karsilastir
                if len(self.get_disappeared_list()) != 0:
                    for trackable in self.get_disappeared_list():
                        dis_list[trackable.objectID] = trackable.last_coordinate
                    closest_index,closest_node = self.closest_node(inputCentroids[col],list(dis_list.values()))
                    if m.hypot(closest_node[0] - inputCentroids[col][0], closest_node[1] - inputCentroids[col][1]) <= 100:
                        objectID = closest_index
                        self.disappeared_list = [x for x in self.get_disappeared_list() if x.objectID != closest_index]

                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                # usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # ekranda varolan nesne merkezlerinin sayisi yeni gelen
            # nesne merkezlerinden fazlaysa bazi nesneler kaybolmustur
                                                                                                 
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                self.register(inputCentroids[col])                        

        # return the set of trackable objects
        return self.objects

# this method gives randomized bgr color as a tuple. 
def get_random_color():
    r = random.randint(0,random.randint(100,255))
    g = random.randint(0,random.randint(120,255))
    b = random.randint(0,random.randint(150,255))
    return (b,g,r)

# this method filters frame that we are using
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
  
cap = cv2.VideoCapture(R"C:\Users\ylmz\Desktop\OPENCV TEST VIDEOS\test6.mp4")
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
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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

    disappeared_list = ct.get_disappeared_list()
    disappeared_objectids = [x.objectID for x in disappeared_list]

    cv2.imshow('frame', frame)
    cv2.imshow('mask',mask)

    if(cv2.waitKey(25) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
