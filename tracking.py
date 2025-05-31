import cv2
import requests
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from dotenv import load_dotenv
import os
import sqlite3
from datetime import datetime
from PIL import Image
import onnxruntime
from torchvision import transforms


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

# 학습한 Detection 모델 load
model = YOLO("yolov8_n.pt")
model.to('cpu') # 추론 시 cpu 사용, GPU 사용 시 수정.

tracker = DeepSort(max_age=10, n_init=3)

# Classification 모델 load. 포맷 주의.
onnx_session = onnxruntime.InferenceSession("final_classification.onnx", providers=['CPUExecutionProvider'])  # 현재는 cpu로 추론. GPU 추론 시 변경.
onnx_input_name = onnx_session.get_inputs()[0].name

# 모델 input 맞춰서 ONNX 전처리
onnx_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

conn = sqlite3.connect('illegal_vehicle.db')
cursor = conn.cursor()


# 바운딩 박스 색상 지정
np.random.seed(42)
colors = np.random.randint(0, 255, size = (1000, 3), dtype = int)

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    results = model(frame, conf=0.3, iou = 0.3)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id != 1:
            continue  # illegal class만 추적
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w, h = x2 - x1, y2 - y1
        conf = float(box.conf[0])
        detections.append(([x1, y1, w, h], conf, cls_id))  # DeepSORT input 요구 -> [x, y, w, h]

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # ROI 추출 및 ONNX 재분류
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        input_tensor = onnx_transform(pil_img).unsqueeze(0).numpy()
        output = onnx_session.run(None, {onnx_input_name: input_tensor})
        logit = output[0][0][0]
        prob = 1 / (1 + np.exp(-logit))
        pred_class = int(prob > 0.5)  # 0: illegal, 1: legal

        if pred_class != 0:
            continue  # ONNX 기준 illegal 아님 패스

        # 이미 DB에 저장되어 있는지 확인 (중복 저장 방지)
        cursor.execute("SELECT 1 FROM illegal_vehicles WHERE track_id=?", (str(track_id),))
        if cursor.fetchone() is None:
            save_path = f"saved_illegal/illegal_{track_id}.jpg"
            cv2.imwrite(save_path, roi)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
                INSERT INTO illegal_vehicles (track_id, timestamp, class, x1, y1, x2, y2, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (str(track_id), timestamp, 'illegal', x1, y1, x2, y2, save_path))
            conn.commit()

        # 시각화
        color = colors[hash(track_id) % len(colors)].tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("ITS CCTV", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()