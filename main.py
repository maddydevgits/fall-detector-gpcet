import numpy as np
import argparse
import imutils
import time
import cv2
import os

import streamlit as st
from constants import *

# load the COCO class labels our YOLO model was trained on
LABELS = open(YOLOV3_LABELS_PATH).read().strip().split('\n')

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

weightsPath = YOLOV3_WEIGHTS_PATH
configPath = YOLOV3_CFG_PATH

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()   #Changes here
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

def app():
    st.title('Fall Alert System')
    run = st.checkbox('Run Camera')
    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    vs = cv2.VideoCapture(0)
    writer = None
    (W, H) = (None, None)
    FRAME_WINDOW = st.image([])
    cnt = 0 
    fno = 0
    # ------------------FRAME PART-----------------------------------------
    counter = 0
    while run:
        start1 = time.time()
        (grabbed, frame) = vs.read()
        fno+=1
        if fno%2!=0:
            continue
        print("Frame No:", fno)
        if not grabbed:
            break
	    # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        boxes = []
        confidences = []
        classIDs = []

	    # loop over each of the layer outputs
        for output in layerOutputs:
		    # loop over each of the detections
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

			    # filter out weak predictions by ensuring the detected
			    # probability is greater than the minimum probability
                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))     #Top-Left Co-ordinates
                    y = int(centerY - (height / 2))

				    # update our list of bounding box coordinates,
				    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)			

	    # apply non-maxima suppression to suppress weak, overlapping
	    # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
	    #idxs contains indices that can be used from boxes, classIDs lists
	    # ensure at least one detection exists
        if len(idxs) > 0:
            idArray = []
            for j in idxs.flatten():
                if classIDs[j]==0:
                    idArray.append(j)
	    # 	#Intersection of bounding box code
            print("Dimensions:")
	    # boxes, idxs, idArray
            print(f"For Boxes:: Length: {len(boxes)} Values: {boxes}")
            print(f"For idxs:: Length: {idxs.size} Values: {idxs.flatten()}")
            print(f"For idArray:: Length: {len(idArray)} Values: {idArray}")
            for i in idArray:
		    # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                if(w>h):
			    # draw a bounding box rectangle and label on the frame
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
                    text = "FALL ALERT!! {}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                    cv2.putText(frame, text, (x, y - 5),
				    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.imwrite('fall/demo.png',frame)
                    print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
                else:
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]],
					confidences[i])
                    cv2.putText(frame, text, (x, y - 5),
				    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.imwrite("output-test1\\Frame{}.jpg".format(fno), frame)
                    print(f"Box[{i}]: {x} {y} {w} {h} Labels[{i}]: {LABELS[classIDs[i]]} classIDs[{i}]: {classIDs[i]}  confidences[{i}]: {confidences[i]}")			
        end1 = time.time()
	    # cv2.imwrite("output-image/frame{}.jpg".format(fno), frame)
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        print("Complete time for algorithm", (end1-start1))
        cnt+=1
        cv2.waitKey(2)
    print("[INFO] cleaning up...")
    vs.release()

app()