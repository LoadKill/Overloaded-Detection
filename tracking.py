import cv2
import requests
from ultralytics import YOLO
from sort import Sort
import numpy as np
from dotenv import load_dotenv
import os

# 1. ITS API 호출
load_dotenv()
api_key = os.getenv('ITS_API_KEY')
api_url = f"https://openapi.its.go.kr:9443/cctvInfo?apiKey={api_key}&type=ex&cctvType=1&minX=126.8&maxX=127.2&minY=37.4&maxY=37.7&getType=json"
cctv_list = requests.get(api_url).json()['response']['data']

# 2. 첫 번째 CCTV 스트림 URL
url = cctv_list[0]['cctvurl']

print(f"열려는 CCTV URL: {url}")


# 3. OpenCV로 영상 스트림 열기
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("스트림을 열 수 없습니다.")
    exit()

model = YOLO("yolov8_n.pt")
model.to('cpu') # 추론 시 cpu 사용

tracker = Sort()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    results = model(frame, conf = 0.3)[0]
    car_detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        car_detections.append([x1, y1, x2, y2, conf])

    if len(car_detections) > 0:
        dets_np = np.array(car_detections)

    else:
        dets_np = np.empty((0, 5))

    
    tracks = tracker.update(dets_np)

    
    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
        
    # 여기서 YOLO → SORT 로 연동 가능
    cv2.imshow("ITS CCTV", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()