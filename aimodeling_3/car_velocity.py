import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
import time

model = YOLO("yolov8n.pt", task='detect')

def XY_coordinate(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        xy = [x, y]
        print(xy)

cv2.namedWindow('Car_velocity')
cv2.setMouseCallback('Car_velocity', XY_coordinate)

cap = cv2.VideoCapture('traffic_mov.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
# print(class_list)

count = 0

tracker = Tracker()

cy1 = 322
cy2 = 368
offset = 6

vh_down = {}
vh_up = {}
down_counter = []
up_counter = []
violation = []
violation_id = ''

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list1 = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if ('car' in c) or ('truck' in c):
            list1.append([x1, y1, x2, y2])
    bbox_id = tracker.update(list1)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int((x3 + x4) // 2)
        cy = int((y3 + y4) // 2)

        # going down
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            vh_down[id] = time.time()
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        if id in vh_down:
            if cy2 < (cy + offset) and cy2 > (cy - offset):
                elapsed_time = time.time() - vh_down[id]
                if down_counter.count(id) == 0:
                    down_counter.append(id)
                    distance = 10  # meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 255, 255), 2)
                    time.sleep(1)
                    if a_speed_kh > 50:
                        violation.append(id)

        # going up
        if cy2 < (cy + offset) and cy2 > (cy - offset):
            vh_up[id] = time.time()
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        if id in vh_up:
            if cy1 < (cy + offset) and cy1 > (cy - offset):
                elapsed_time = time.time() - vh_up[id]
                if up_counter.count(id) == 0:
                    up_counter.append(id)
                    distance = 10  # meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 255, 255), 2)
                    time.sleep(1)
                    if a_speed_kh > 50:
                        violation.append(id)

    cv2.line(frame, (267, cy1), (829, cy1), (255, 255, 255), 1)
    cv2.putText(frame, 'line', (274, 318), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.line(frame, (167, cy2), (932, cy2), (255, 255, 255), 1)
    cv2.putText(frame, 'line', (181, 363), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    down_count = len(down_counter)
    cv2.putText(frame, 'going down : ' + str(down_count), (60, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

    up_count = len(up_counter)
    cv2.putText(frame, 'going up : ' + str(up_count), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

    for i in range(len(violation)):
        violation_id += str(violation[i])
        violation_id += ' '

    cv2.putText(frame, 'violation : ' + str(violation_id), (700, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
    violation_id = ''

    cv2.imshow('Car_velocity', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
