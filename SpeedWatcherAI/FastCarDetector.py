import numpy as np
from ultralytics import YOLO
import os
import cvzone
import cv2
import math
from sort import *
import time
from datetime import datetime
import csv

cap = cv2.VideoCapture('C:/Users/sivam/PycharmProjects/PythonProject/Project1/Videos_Testing/traffic1.mp4')
#video_folder = 'C:/Users/sivam/PycharmProjects/PythonProject/Project1/Videos_Testing'
#video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

log_file = "vehicle_log.csv"
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Entry Frame", "Exit Frame", "Speed", "Status", "Timestamp", "Location"])

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign",
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "bench",
              "parking meter",
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
              "kite", "baseball bat", "baseball glove", "skateboard",
              "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
              "apple", "sandwich", "orange", "broccoli", "carrot",
              "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
              "tvmonitor", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

model = YOLO('yolov8n.pt')
musk = cv2.imread('musk2.png')

#max_age = frames that are gone and time i need to put it back, higher its value longer it takes(better it is)
tracker = Sort(max_age=20,min_hits= 3, iou_threshold=0.3)
                                                                                        # ,iou = intersection over union, tracker for id
location = "Kolkata NH2_Tool_Plaza"
lineA = [480,300,783,300]
lineB = [400,400,850,400]

crossing_data = {}
vehicle_speeds = {}
fps = cap.get(cv2.CAP_PROP_FPS)

SPEED_LIMIT = 40

#for video_name in video_files:
    #video_path = os.path.join(video_folder, 'traffic2.mp4')
    #cap = cv2.VideoCapture(video_path)

    #print(f"\n🔁 Running on: {video_name}")
while True:
    success, img = cap.read()
    #print(img.shape)                                #The shape of img is (1080,1920,3)
    #print(musk.shape)                               #The shape of musk is (432,768,3)
    musk = cv2.resize(musk,(img.shape[1],img.shape[0]))
    imgRegion = cv2.bitwise_and(img,musk)

    results = model(imgRegion, stream=True)
    detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:  # bounding boxes
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "bus" or currentClass == "car" or currentClass == "truck" or currentClass == "motorbike" and conf > 0.5:
                '''cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1, offset = 3)
                cvzone.cornerRect(img, (x1, y1, w, h), l=5,rt=5)'''                  #rt = rectangle thickness,l=length
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))           #array.append

    resultsTracker = tracker.update(detections)
    #cv2.line(img,(limit[0],limit[1]),(limit[2],limit[3]),(0,0,255),3)
    cv2.line(img,(lineA[0],lineA[1]),(lineA[2],lineA[3]),(0,255,255),2)
    cv2.putText(img, "Line A", (lineA[0], lineA[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.line(img,(lineB[0],lineB[1]),(lineB[2],lineB[3]),(255,0,0),2)
    cv2.putText(img,"Line B",(lineB[0],lineB[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        print(result)
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        w,h = x2-x1,y2-y1
        id = int(id)
        cvzone.cornerRect(img,(x1,y1,w,h),l=4,rt=2,colorR = (255,0,0),colorC=(0,255,0))
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1,offset=3)

        # center
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        current_time = time.time()
        print(f"ID {id} center y (cy): {cy}, LineA_y: {lineA[1]}, LineB_y: {lineB[1]}")
        #check crossing A
        if abs(cy - lineA[1]) < 15:
            if id not in crossing_data:
                crossing_data[id] = {'entry_frame': cap.get(cv2.CAP_PROP_POS_FRAMES)}
                print(f'Id {id} crossed line A at {current_time:.2f}')

        #crossing B
        elif abs(cy - lineB[1]) < 15:
            if id in crossing_data and 'exit_frame' not in crossing_data[id]:
                crossing_data[id]['exit_frame'] = cap.get(cv2.CAP_PROP_POS_FRAMES)

                entry_frame = crossing_data[id]['entry_frame']
                exit_frame = crossing_data[id]['exit_frame']
                frames_diff = exit_frame - entry_frame
                distance_m = 15  # Real distance between Line A & B

                speed = (distance_m /(frames_diff/fps))* 3.6
                speed = int(speed)
                vehicle_speeds[id] = speed
                del crossing_data[id]
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if speed > SPEED_LIMIT:
                    # Crop and save vehicle region
                    cropped = img[y1:y2, x1:x2]
                    cv2.imwrite(f"violations/ID{id}_speed_{speed}.jpg", cropped)

                status = "Overspeed" if speed > SPEED_LIMIT else "OK"
                with open(log_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([id, int(entry_frame), int(exit_frame), speed, status, timestamp, location])

        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1, offset=3)
        if id in vehicle_speeds:
            cvzone.putTextRect(img, f'{vehicle_speeds[id]} km/h', (x1, y1 - 25), scale=0.8, thickness=1, offset=2,
                               colorR=(0, 255, 255))
        #.putTextRect(img,f'Count:- {len(totalCount)}',(40,35))

        if id in vehicle_speeds:
            speed = vehicle_speeds[id]
            color = (0, 255, 0) if speed <= SPEED_LIMIT else (0, 0, 255)  # green or red
            cvzone.putTextRect(img, f'{speed} km/h', (x1, y1 - 25), scale=0.8, thickness=1, offset=2, colorR=color)


    cv2.imshow("Image", img)
    #cv2.imshow("Image_Region",imgRegion)
    cv2.waitKey(1)